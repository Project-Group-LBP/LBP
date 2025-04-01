import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNN
import numpy as np


# Fully connected MLP for Actor
# class ActorNetwork(nn.Module):
#     def __init__(self, obs_dim, action_dim, hidden_dim):
#         super(ActorNetwork, self).__init__()
#         self.fc1 = nn.Linear(obs_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.out = nn.Linear(hidden_dim, action_dim)

#     def forward(self, obs):
#         x = F.relu(self.fc1(obs))
#         x = F.relu(self.fc2(x))
#         action = torch.tanh(self.out(x))
#         return action

class ActorNetwork(nn.Module):
    def __init__(self, input_channels=3, action_dim=2, hidden_dim=600):
        super(ActorNetwork, self).__init__()

        # CNN feature extractor
        self.cnn = CNN(input_channels)
        feature_dim = self.cnn._calculate_conv_output_dim()

        # MLP layers
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        features = self.cnn(x)
        x = F.relu(self.fc1(features))
        # Output actions in [-1, 1] range using tanh
        actions = torch.tanh(self.fc2(x))
        return actions


class CriticNetwork(nn.Module):
    '''Centralized Critic: takes joint observations and joint actions'''
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, joint_obs, joint_actions):
        x = torch.cat([joint_obs, joint_actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.out(x)
        return q_value


# Soft Target Update Utility
def soft_update(target_net, source_net, tau):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


# Gaussian noise with decay for exploration
class GaussianNoise:
    def __init__(self, size, initial_scale=1.0, min_scale=0.1, decay_rate=0.995):
        self.size = size
        self.initial_scale = initial_scale
        self.min_scale = min_scale
        self.decay_rate = decay_rate
        self.scale = initial_scale
        self.reset()

    def reset(self):
        self.scale = self.initial_scale

    def sample(self):
        return self.scale * np.random.randn(self.size)

    def decay(self):
        """Reduce noise scale over time"""
        self.scale = max(self.min_scale, self.scale * self.decay_rate)
