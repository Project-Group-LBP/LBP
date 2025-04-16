# Pytorch implementation of MADDPG algorithm
# for simple_spread_v3 cooperative environment of PettingZoo MPE (prev OpenAI MPE)
# https://pettingzoo.farama.org/environments/mpe/simple_spread/
# Implemented for a parallel environment with continuous actions

# Agents must learn to cover all the landmarks while avoiding collisions.
# More specifically, all agents are globally rewarded
# based on how far the closest agent is to each landmark (negative sum of the minimum distances).
# Locally, the agents are penalized if they collide with
# other agents (-1 for each collision).

# Additional features apart from core algorithm:
# - Gradient clipping
# - Normalization of rewards
# - OU noise for exploration with noise decay
# - Network updates at intervals
# - Saving and loading model weights
# - Plotting training and testing rewards

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
import os
import argparse


class OUActionNoise:
    def __init__(
        self,
        action_dim,
        mu=0.5,
        theta=0.15,
        sigma=0.3,
        dt=1e-2,
        decay=0.995,
        sigma_min=0.05,
    ):
        self.mu = np.ones(action_dim) * mu
        self.theta = theta
        self.sigma = np.ones(action_dim) * sigma
        self.sigma_decay = decay
        self.sigma_min = sigma_min
        self.dt = dt
        self.reset()

    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.state = x + dx
        return self.state.copy()

    def reset(self):
        self.state = np.copy(self.mu)
        self.sigma = np.maximum(self.sigma_min, self.sigma * self.sigma_decay)


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, is_actor=False):
        super(MLPNetwork, self).__init__()
        self.is_actor = is_actor
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)

    def forward(self, input):
        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float)
        x = nn.functional.relu(self.layer1(input))
        x = nn.functional.relu(self.layer2(x))
        if self.is_actor:  # Actor network outputs actions in range [0, 1]
            output = torch.sigmoid(self.layer3(x))
        else:
            output = self.layer3(x)
        return output


class Agent:
    def __init__(self, obs_dim, act_dim, global_dim, actor_lr, critic_lr):
        self.actor = MLPNetwork(obs_dim, act_dim, is_actor=True)
        self.target_actor = MLPNetwork(obs_dim, act_dim, is_actor=True)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic = MLPNetwork(global_dim, 1)
        self.target_critic = MLPNetwork(global_dim, 1)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)


class Buffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self._index = 0
        self.size = 0
        self.obs_buffer = np.zeros((capacity, obs_dim))
        self.action_buffer = np.zeros((capacity, act_dim))
        self.reward_buffer = np.zeros(capacity)
        self.next_obs_buffer = np.zeros((capacity, obs_dim))
        self.done_buffer = np.zeros(capacity, dtype=bool)

    def add(self, obs, action, reward, next_obs, done):
        self.obs_buffer[self._index] = obs
        self.action_buffer[self._index] = action
        self.reward_buffer[self._index] = reward
        self.next_obs_buffer[self._index] = next_obs
        self.done_buffer[self._index] = done
        self._index = (self._index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, indices):
        obs = self.obs_buffer[indices]
        action = self.action_buffer[indices]
        reward = self.reward_buffer[indices]
        next_obs = self.next_obs_buffer[indices]
        done = self.done_buffer[indices]
        obs = torch.from_numpy(obs).float()
        action = torch.from_numpy(action).float()
        reward = (reward - reward.mean()) / (reward.std() + 1e-7)  # normalization
        reward = torch.from_numpy(reward).float()
        next_obs = torch.from_numpy(next_obs).float()
        done = torch.from_numpy(done).float()
        return obs, action, reward, next_obs, done


class MADDPG:
    def __init__(
        self,
        epi_length,
        capacity,
        batch_size,
        gamma,
        tau,
        actor_lr,
        critic_lr,
        local_ratio,
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.env = simple_spread_v3.parallel_env(max_cycles=epi_length, continuous_actions=True, local_ratio=local_ratio)
        self.env.reset()
        first_agent = self.env.agents[0]
        obs_dim = self.env.observation_space(first_agent).shape[0]
        act_dim = self.env.action_space(first_agent).shape[0]
        global_obs_act_dim = self.env.num_agents * (obs_dim + act_dim)
        self.agents = {}
        self.buffers = {}
        self.noises = {}
        for agent_id in self.env.agents:
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim)
            self.noises[agent_id] = OUActionNoise(act_dim)

    def update_targets(self):  # Polyak averaging
        def soft_update(from_network, to_network):
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(self.tau * from_p.data + (1.0 - self.tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def sample(self):
        indices = np.random.choice(self.buffers["agent_0"].size, size=self.batch_size, replace=False)
        obs_dict, act_dict, rew_dict, done_dict = {}, {}, {}, {}
        glob_obs, glob_act, glob_next_obs, glob_next_act = [], [], [], []
        # Sample experience from buffers of all agents
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d = buffer.sample(indices)
            obs_dict[agent_id] = o
            glob_obs.append(o)
            act_dict[agent_id] = a
            glob_act.append(a)
            glob_next_obs.append(n_o)
            rew_dict[agent_id] = r
            done_dict[agent_id] = d
            # Calculate next actions using target network and next states
            next_actions = self.agents[agent_id].target_actor(n_o)
            glob_next_act.append(next_actions)
        return (
            obs_dict,
            act_dict,
            rew_dict,
            done_dict,
            glob_obs,
            glob_act,
            glob_next_obs,
            glob_next_act,
        )

    # Heart of the algorithm
    def learn(self):
        for agent_id, agent in self.agents.items():
            obs, act, reward, done, glob_obs, glob_act, glob_next_obs, glob_next_act = self.sample()

            # Update critic
            q = agent.critic(torch.cat(glob_obs + glob_act, 1)).squeeze(1)
            q_target = agent.target_critic(torch.cat(glob_next_obs + glob_next_act, 1)).squeeze(1)
            # Squeeze : [batch_size, 1] -> [batch_size]

            y = reward[agent_id] + self.gamma * q_target * (1 - done[agent_id])
            critic_loss = nn.MSELoss()(q, y)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)  # gradient clipping
            agent.critic_optimizer.step()

            # Update actor
            act[agent_id] = agent.actor(obs[agent_id])
            glob_act = list(act.values())
            q = agent.critic(torch.cat(glob_obs + glob_act, 1)).squeeze(1)
            actor_loss = -q.mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)  # gradient clipping
            agent.actor_optimizer.step()

    def select_train_action(self, obs, step, initial_steps):
        actions = {}
        if step < initial_steps:  # randomly explore initially
            for agent_id in self.env.agents:
                actions[agent_id] = self.env.action_space(agent_id).sample()
        else:  # select actions using actor networks
            for id, o in obs.items():
                o = torch.from_numpy(o).unsqueeze(0).float()
                action = self.agents[id].actor(o).squeeze(0).detach().numpy()  # Squeeze : [1, action_dim] -> [action_dim]
                actions[id] = np.clip(action + self.noises[id](), 0.0, 1.0).astype(np.float32)  # add noise
        return actions

    def select_test_action(self, obs):  # testing without noise
        actions = {}
        for agent_id, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float()
            actions[agent_id] = self.agents[agent_id].actor(o).squeeze(0).detach().numpy()
        return actions

    def train(self, episodes, initial_steps, learn_interval):
        step = 0
        self.env.reset()
        episode_rewards = {agent_id: np.zeros(episodes) for agent_id in self.env.agents}

        for epi in range(episodes):
            obs, _ = self.env.reset()
            if step > initial_steps:
                for noise in self.noises.values():
                    noise.reset()
            curr_epi_rewards = {agent_id: 0 for agent_id in self.env.agents}

            while self.env.agents:
                step += 1
                action = self.select_train_action(obs, step, initial_steps)
                next_obs, reward, term, trunc, _ = self.env.step(action)
                for agent_id in self.env.agents:
                    self.buffers[agent_id].add(
                        obs[agent_id],
                        action[agent_id],
                        reward[agent_id],
                        next_obs[agent_id],
                        term[agent_id] or trunc[agent_id],
                    )
                obs = next_obs

                for agent_id, r in reward.items():
                    curr_epi_rewards[agent_id] += r

                if step >= initial_steps and step % learn_interval == 0:
                    self.learn()
                    self.update_targets()

            for agent_id, r in curr_epi_rewards.items():
                episode_rewards[agent_id][epi] = r

            if (epi + 1) % 100 == 0:
                message = f"Episode {epi + 1}, "
                sum_reward = 0
                for agent_id, reward in curr_epi_rewards.items():
                    message += f"{agent_id}: {reward:>4f}; "
                    sum_reward += reward
                message += f"Sum: {sum_reward}"
                print(message)

        return episode_rewards

    def test(self, episodes):
        self.env.reset()
        episode_rewards = {agent_id: np.zeros(episodes) for agent_id in self.env.agents}

        for epi in range(episodes):
            obs, _ = self.env.reset()
            curr_epi_rewards = {agent_id: 0 for agent_id in self.env.agents}

            while self.env.agents:
                action = self.select_test_action(obs)
                next_obs, reward, _, _, _ = self.env.step(action)
                obs = next_obs

                for agent_id, r in reward.items():
                    curr_epi_rewards[agent_id] += r

            for agent_id, r in curr_epi_rewards.items():
                episode_rewards[agent_id][epi] = r

            if (epi + 1) % 5 == 0:
                message = f"Episode {epi + 1}, "
                sum_reward = 0
                for agent_id, reward in curr_epi_rewards.items():
                    message += f"{agent_id}: {reward:>4f}; "
                    sum_reward += reward
                message += f"Sum: {sum_reward}"
                print(message)

        return episode_rewards


# saving actor parameters for each agent
def save_model(maddpg, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    torch.save(
        {name: agent.actor.state_dict() for name, agent in maddpg.agents.items()},
        os.path.join(result_dir, "model.pt"),
    )


def get_running_reward(arr: np.ndarray, window=100):
    running_reward = np.zeros_like(arr)
    for i in range(window - 1):
        running_reward[i] = np.mean(arr[: i + 1])
    for i in range(window - 1, len(arr)):
        running_reward[i] = np.mean(arr[i - window + 1 : i + 1])
    return running_reward


def plot_graphs(num_episodes, episode_rewards, result_dir, test=False):
    _, ax = plt.subplots()
    x = range(1, num_episodes + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        if test == False:
            ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    if test:
        title = "Testing"
    else:
        title = "Training"
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))


def load_model(maddpg, episode_length, file):
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    data = torch.load(file, weights_only=True)
    maddpg.env = simple_spread_v3.parallel_env(max_cycles=episode_length, continuous_actions=True)
    maddpg.env.reset()
    for agent_id, agent in maddpg.agents.items():
        agent.actor.load_state_dict(data[agent_id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--test", action="store_true", help="Test the agent")
    parser.add_argument("--local_ratio", type=float, default=0.5)
    args = parser.parse_args()
    assert args.train or args.test, "Please provide either train or test flag"

    LEARN_INTERVAL = 100
    INITIAL_STEPS = 5000
    TAU = 0.02
    GAMMA = 0.95
    BUFFER_CAPACITY = 100000
    BATCH_SIZE = 1024
    ACTOR_LR = 0.01
    CRITIC_LR = 0.01
    NUM_TRAIN_EPISODES = 3000
    NUM_TEST_EPISODES = 50
    EPISODE_LENGTH = 25

    maddpg = MADDPG(
        EPISODE_LENGTH,
        BUFFER_CAPACITY,
        BATCH_SIZE,
        GAMMA,
        TAU,
        ACTOR_LR,
        CRITIC_LR,
        args.local_ratio,
    )
    result_dir = os.path.join("./results")

    if args.train:
        print("Training")
        episode_rewards = maddpg.train(NUM_TRAIN_EPISODES, INITIAL_STEPS, LEARN_INTERVAL)
        save_model(maddpg, result_dir)
        plot_graphs(NUM_TRAIN_EPISODES, episode_rewards, result_dir)

    if args.test:
        print("Testing")
        load_model(maddpg, EPISODE_LENGTH, os.path.join(result_dir, "model.pt"))
        episode_rewards = maddpg.test(NUM_TEST_EPISODES)
        plot_graphs(NUM_TEST_EPISODES, episode_rewards, result_dir, test=True)
