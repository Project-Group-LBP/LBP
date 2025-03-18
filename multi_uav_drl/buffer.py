import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size, num_agents, obs_dim, action_dim):
        self.max_size = max_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.buffer = deque(maxlen=max_size)

    def add(self, obs, actions, rewards, next_obs, dones):
        """
        Store one experience tuple: (joint_obs, joint_actions, joint_rewards, joint_next_obs, joint_dones)
        """
        self.buffer.append((obs, actions, rewards, next_obs, dones))

    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        Returns: batches of (obs, actions, rewards, next_obs, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        obs_batch = np.array([item[0] for item in batch])      # Shape: [batch_size, num_agents, obs_dim]
        actions_batch = np.array([item[1] for item in batch])  # Shape: [batch_size, num_agents, action_dim]
        rewards_batch = np.array([item[2] for item in batch])  # Shape: [batch_size, num_agents]
        next_obs_batch = np.array([item[3] for item in batch]) # Shape: [batch_size, num_agents, obs_dim]
        dones_batch = np.array([item[4] for item in batch])    # Shape: [batch_size, num_agents]
        
        return obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch

    def __len__(self):
        return len(self.buffer)
