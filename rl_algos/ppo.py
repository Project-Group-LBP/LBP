# PPO implementation for continuous action space environments in PyTorch

import torch
from torch import nn
from torch.distributions import MultivariateNormal
import numpy as np
import gymnasium as gym


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)

    def forward(self, input):
        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float)
        activation1 = nn.functional.relu(self.layer1(input))
        activation2 = nn.functional.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output


env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


class PPOAgent:
    def __init__(self):
        self.gamma = 0.99
        self.clip = 0.2  # epsilon
        self.std_dev = 0.2
        self.cov_mat = torch.diag(torch.full(size=(action_dim,), fill_value=self.std_dev))
        self.steps_per_epoch = 2048
        self.max_steps_per_episode = 200
        self.update_pi_iters = 5
        self.update_v_iters = 5
        self.actor_network = FeedForwardNN(obs_dim, action_dim)
        self.critic_network = FeedForwardNN(obs_dim, 1)
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=3e-4)

    def get_action(self, state):
        mean = self.actor_network(state)
        distr = MultivariateNormal(mean, self.cov_mat)
        action = distr.sample()
        log_prob = distr.log_prob(action)
        action = action.detach().numpy()
        log_prob = log_prob.detach()
        return action, log_prob

    def compute_rtgs(self, epoch_rewards):
        # reversing makes it easier to compute rewards-to-go
        epoch_rtgs = []
        for epi_rewards in reversed(epoch_rewards):
            discounted_sum = 0
            for reward in reversed(epi_rewards):
                discounted_sum = reward + self.gamma * discounted_sum
                epoch_rtgs.append(discounted_sum)
        epoch_rtgs.reverse()
        epoch_rtgs = torch.tensor(epoch_rtgs, dtype=torch.float)
        return epoch_rtgs

    def get_trajectories(self):
        epoch_states = []
        epoch_actions = []
        epoch_log_probs = []
        epoch_rewards = []
        # in one epoch keep running episodes until you hit steps_per_epoch limit
        # but don't let any episode be longer than specified max_steps_per_episode
        epoch_steps = 0
        while epoch_steps < self.steps_per_epoch:
            state, _ = env.reset()
            epi_rewards = []
            for _ in range(self.max_steps_per_episode):
                action, log_prob = self.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                epoch_states.append(state)
                epoch_actions.append(action)
                epoch_log_probs.append(log_prob)
                epi_rewards.append(reward)
                state = next_state
                epoch_steps += 1
                if terminated or truncated:
                    break
            epoch_rewards.append(epi_rewards)
        epoch_states = torch.tensor(epoch_states, dtype=torch.float)
        epoch_actions = torch.tensor(epoch_actions, dtype=torch.float)
        epoch_log_probs = torch.tensor(epoch_log_probs, dtype=torch.float)
        epoch_rtgs = self.compute_rtgs(epoch_rewards)
        return (epoch_states, epoch_actions, epoch_log_probs, epoch_rtgs), epoch_rewards

    # Heart of algorithm
    def update(self, epoch_tuple):
        (epoch_states, epoch_actions, epoch_log_probs, epoch_rtgs) = epoch_tuple
        v_k = self.critic_network(epoch_states).squeeze()
        A_k = epoch_rtgs - v_k.detach()  # advantage estimate : rtgs - v
        # Normalization of advantage fn for stable training
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        for _ in range(self.update_pi_iters):
            mean = self.actor_network(epoch_states)
            distr = MultivariateNormal(mean, self.cov_mat)
            curr_log_probs = distr.log_prob(epoch_actions)
            ratios = torch.exp(curr_log_probs - epoch_log_probs)
            term1 = ratios * A_k
            term2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
            # Used minus for gradient ascent
            actor_loss = -torch.min(term1, term2).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        for _ in range(self.update_v_iters):
            v = self.critic_network(epoch_states).squeeze()
            critic_loss = nn.MSELoss()(v, epoch_rtgs)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def train(self, num_epochs):
        print("Training agent")
        for epoch in range(num_epochs):
            epoch_tuple, epoch_rewards = self.get_trajectories()
            self.update(epoch_tuple)
            mean_return = np.mean([np.sum(rewards) for rewards in epoch_rewards])
            print(f"Epoch {epoch+1}: Mean Return = {mean_return:.2f}")

    def test(self, num_episodes):
        print("\nTesting agent")
        for ep in range(num_episodes):
            state, _ = env.reset()
            episodic_reward = 0
            for _ in range(self.max_steps_per_episode):
                action = self.actor_network(state).detach().numpy()
                next_state, reward, terminated, truncated, _ = env.step(action)
                state = next_state
                episodic_reward += reward
                if terminated or truncated:
                    break
            print(f"Episode {ep+1}: Reward = {episodic_reward:.2f}")


agent = PPOAgent()
agent.train(num_epochs=50)
agent.test(num_episodes=10)
env.close()

"""
Sample Output:
Training agent
Epoch 1: Mean Return = -1178.02
Epoch 2: Mean Return = -1403.34
Epoch 3: Mean Return = -1264.46
Epoch 4: Mean Return = -1224.37
Epoch 5: Mean Return = -1182.62
Epoch 6: Mean Return = -1040.61
Epoch 7: Mean Return = -1152.82
Epoch 8: Mean Return = -1236.48
Epoch 9: Mean Return = -1067.14
Epoch 10: Mean Return = -1253.62
Epoch 11: Mean Return = -1088.39
Epoch 12: Mean Return = -1168.38
Epoch 13: Mean Return = -1056.22
Epoch 14: Mean Return = -1161.43
Epoch 15: Mean Return = -1216.85
Epoch 16: Mean Return = -1074.27
Epoch 17: Mean Return = -1131.71
Epoch 18: Mean Return = -1093.66
Epoch 19: Mean Return = -1019.46
Epoch 20: Mean Return = -1213.00
Epoch 21: Mean Return = -1069.61
Epoch 22: Mean Return = -1032.48
Epoch 23: Mean Return = -1207.05
Epoch 24: Mean Return = -1143.79
Epoch 25: Mean Return = -1100.03
Epoch 26: Mean Return = -1037.51
Epoch 27: Mean Return = -1093.75
Epoch 28: Mean Return = -1124.08
Epoch 29: Mean Return = -1130.29
Epoch 30: Mean Return = -1138.51
Epoch 31: Mean Return = -973.41
Epoch 32: Mean Return = -1112.20
Epoch 33: Mean Return = -1028.18
Epoch 34: Mean Return = -1064.85
Epoch 35: Mean Return = -1099.74
Epoch 36: Mean Return = -1137.13
Epoch 37: Mean Return = -1082.18
Epoch 38: Mean Return = -1084.51
Epoch 39: Mean Return = -1060.64
Epoch 40: Mean Return = -1033.65
Epoch 41: Mean Return = -1050.09
Epoch 42: Mean Return = -1016.56
Epoch 43: Mean Return = -1086.70
Epoch 44: Mean Return = -1134.78
Epoch 45: Mean Return = -1147.72
Epoch 46: Mean Return = -1102.80
Epoch 47: Mean Return = -932.00
Epoch 48: Mean Return = -1168.47
Epoch 49: Mean Return = -1043.41
Epoch 50: Mean Return = -1163.35

Testing agent
Episode 1: Reward = -1021.96
Episode 2: Reward = -751.86
Episode 3: Reward = -1048.60
Episode 4: Reward = -1025.15
Episode 5: Reward = -1018.41
Episode 6: Reward = -1018.77
Episode 7: Reward = -1313.00
Episode 8: Reward = -1000.26
Episode 9: Reward = -1013.23
Episode 10: Reward = -1042.33
"""
