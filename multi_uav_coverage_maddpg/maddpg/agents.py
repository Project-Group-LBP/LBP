import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .cnn import CNN


class ActorNetwork(nn.Module):
    """Actor Network that maps states to actions using CNN and MLP layers."""

    def __init__(self, action_dim=2, hidden_dim=160):
        super(ActorNetwork, self).__init__()

        # CNN feature extractor
        self.cnn = CNN()
        feature_dim = self.cnn.feature_dim

        # MLP layers
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        features = self.cnn(x)
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        actions = torch.sigmoid(self.out(x))  # Output actions in [0, 1] range
        return actions


class QuantileCriticNetwork(nn.Module):
    """
    Quantile Critic Network:
    This network takes state and action as input and outputs quantile values
    This framework helps to compute CVaR loss
    """

    def __init__(self, input_dim, action_dim, hidden_dim=160, num_quantiles=50):
        super(QuantileCriticNetwork, self).__init__()
        self.num_quantiles = num_quantiles
        self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.quantile_out = nn.Linear(hidden_dim, num_quantiles)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.quantile_out(x)  # Output: [batch_size, num_quantiles]


# Soft Target Update Utility
def soft_update(target_net, source_net, tau):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


# Gaussian noise with decay for exploration
class GaussianNoise:
    def __init__(self, size, initial_scale=1.0, min_scale=0.1, decay_rate=0.995, device='cpu'):
        self.size = size
        self.initial_scale = initial_scale
        self.min_scale = min_scale
        self.decay_rate = decay_rate
        self.scale = initial_scale
        self.device = device
        self.reset()

    def reset(self):
        self.scale = self.initial_scale

    def sample(self):
        return self.scale * torch.randn(self.size, device=self.device)

    def decay(self):
        """Reduce noise scale over time"""
        self.scale = max(self.min_scale, self.scale * self.decay_rate)
