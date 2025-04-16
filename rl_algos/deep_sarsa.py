# Multi-agent independent deep SARSA
# where multiple agents collaborate for maximum coverage of the grid while avoiding obstacles
# (Reward of an agent is affected by positions of other agents)

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Parameters
GRID_SIZE = 10
NUM_EPISODES = 1000
MAX_STEPS = 50
LEARNING_RATE = 0.001
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.1

NUM_AGENTS = 3
NUM_OBSTACLES = 15
BATCH_SIZE = 32


class Environment:
    def __init__(self, grid_size, num_agents, num_obstacles):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_obstacles = num_obstacles
        self.reset()

    def reset(self):
        self.agent_positions = {i: self._random_position() for i in range(self.num_agents)}
        self.obstacle_positions = [self._random_position() for _ in range(self.num_obstacles)]
        self.visited = np.zeros((self.grid_size, self.grid_size))
        self.visited_counts = np.zeros((self.grid_size, self.grid_size))
        return self._get_state()

    def step(self, actions):
        rewards = [self._process_action(agent_id, action) for agent_id, action in enumerate(actions)]
        state = self._get_state()
        done = np.all(self.visited)
        return state, rewards, done

    def _random_position(self):
        return (
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1),
        )

    def _process_action(self, agent_id, action):
        new_pos = self._move(self.agent_positions[agent_id], action)
        if new_pos not in self.obstacle_positions:
            self.agent_positions[agent_id] = new_pos
            return self._calculate_reward(new_pos, agent_id)
        return -1

    def _calculate_reward(self, new_pos, agent_id):
        visit_count = self.visited_counts[new_pos]
        self.visited[new_pos] = 1
        self.visited_counts[new_pos] += 1

        distance_penalty = sum(-0.05 / (1 + np.linalg.norm(np.subtract(new_pos, pos))) for other_id, pos in self.agent_positions.items() if other_id != agent_id)
        sigmoid_penalty = -0.05 / (1 + np.exp(-visit_count + 2))

        return 10 * (self.visited_counts[new_pos] == 1) + 1 + distance_penalty + sigmoid_penalty

    def _move(self, position, action):
        x, y = position
        return [
            (max(x - 1, 0), y),
            (min(x + 1, self.grid_size - 1), y),
            (x, max(y - 1, 0)),
            (x, min(y + 1, self.grid_size - 1)),
        ][action]

    def _get_state(self):
        state = np.zeros((self.grid_size, self.grid_size))

        for pos in self.obstacle_positions:
            state[pos] = -1

        for agent_id, pos in self.agent_positions.items():
            state[pos] = agent_id + 1

        return state


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class MultiAgentDeepSARSA:
    def __init__(self, env):
        self.env = env
        self.q_networks = {agent_id: QNetwork(GRID_SIZE**2, 4) for agent_id in range(NUM_AGENTS)}
        self.target_networks = {agent_id: QNetwork(GRID_SIZE**2, 4) for agent_id in range(NUM_AGENTS)}
        self.optimizers = {agent_id: optim.Adam(self.q_networks[agent_id].parameters(), lr=LEARNING_RATE) for agent_id in range(NUM_AGENTS)}
        self.criterion = nn.MSELoss()
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

    def choose_action(self, state, agent_id):
        if np.random.rand() < self.epsilon:
            return random.choice(range(4))

        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_networks[agent_id](state_tensor)

        return torch.argmax(q_values).item()

    def update_q_network(self, state, action, reward, next_state, next_action, agent_id):
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state.flatten(), dtype=torch.float32)

        q_values = self.q_networks[agent_id](state_tensor)
        next_q_values = self.target_networks[agent_id](next_state_tensor).detach()

        target = q_values.clone()
        target[action] = reward + self.gamma * next_q_values[next_action]

        loss = self.criterion(q_values, target)
        self.optimizers[agent_id].zero_grad()
        loss.backward()
        self.optimizers[agent_id].step()

    def train(self, episodes=NUM_EPISODES):
        rewards_per_episode = []
        rolling_avg_rewards = []

        for _ in range(episodes):
            state = self.env.reset()
            actions = [self.choose_action(state, agent_id) for agent_id in range(NUM_AGENTS)]
            total_reward = 0

            for _ in range(MAX_STEPS):
                next_state, rewards, done = self.env.step(actions)
                next_actions = [self.choose_action(next_state, agent_id) for agent_id in range(NUM_AGENTS)]

                for agent_id in range(NUM_AGENTS):
                    self.update_q_network(
                        state,
                        actions[agent_id],
                        rewards[agent_id],
                        next_state,
                        next_actions[agent_id],
                        agent_id,
                    )

                state, actions = next_state, next_actions
                total_reward += sum(rewards)

                if done:
                    break

            rewards_per_episode.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # Calculate and store rolling average reward
            if len(rewards_per_episode) >= 50:
                rolling_avg_rewards.append(np.mean(rewards_per_episode[-50:]))  # Average of the last 100 rewards
            else:
                rolling_avg_rewards.append(np.mean(rewards_per_episode))

        return rewards_per_episode, rolling_avg_rewards

    def test(self, episodes=100):
        total_rewards = []

        for _ in range(episodes):
            state = self.env.reset()
            total_reward = 0

            for _ in range(MAX_STEPS):

                actions = []
                for agent_id in range(NUM_AGENTS):
                    state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        q_values = self.q_networks[agent_id](state_tensor)
                        actions.append(torch.argmax(q_values).item())

                next_state, rewards, done = self.env.step(actions)
                total_reward += sum(rewards)
                state = next_state
                if done:
                    break

            total_rewards.append(total_reward)

        return total_rewards


def plot_results(train_rewards, rolling_avg_rewards, test_rewards):
    avg_train_reward = np.mean(train_rewards)
    avg_test_reward = np.mean(test_rewards)

    _, axs = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [2, 1]})

    axs[0].plot(
        np.arange(1, len(train_rewards) + 1),
        train_rewards,
        label="Rewards per Episode",
        alpha=0.5,
    )
    axs[0].plot(
        np.arange(1, len(train_rewards) + 1),
        rolling_avg_rewards,
        label="Rolling Average Rewards (per 50 episodes)",
        color="red",
    )
    axs[0].axhline(
        y=avg_train_reward,
        color="green",
        linestyle="--",
        label=f"Overall Avg: {avg_train_reward:.2f}",
    )
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Total Reward")
    axs[0].set_title("Training Progress")
    axs[0].legend()
    axs[0].grid()

    axs[1].bar(
        ["Training", "Testing"],
        [avg_train_reward, avg_test_reward],
        color=["blue", "orange"],
    )
    axs[1].set_ylabel("Average Reward")
    axs[1].set_title("Avg Training vs Avg Testing Reward")

    # Annotate the bars with exact values
    for i, v in enumerate([avg_train_reward, avg_test_reward]):
        axs[1].text(i, v + 1, f"{v:.2f}", ha="center", fontsize=10, color="black")

    axs[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = Environment(GRID_SIZE, NUM_AGENTS, NUM_OBSTACLES)
    agents = MultiAgentDeepSARSA(env)

    print("Training in progress...")
    train_rewards, rolling_avg_rewards = agents.train(NUM_EPISODES)

    print("Testing in progress...")
    test_rewards = agents.test(100)

    print(f"Average Training Reward: {np.mean(train_rewards)}")
    print(f"Average Testing Reward: {np.mean(test_rewards)}")

    plot_results(train_rewards, rolling_avg_rewards, test_rewards)
