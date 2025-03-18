import torch
import numpy as np
import torch.nn.functional as F
from agents import ActorNetwork, CriticNetwork, soft_update, OUNoise
from buffer import ReplayBuffer

class MADDPG:
    def __init__(self, num_agents, obs_dim, action_dim, device='cpu'):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # One actor per UAV (Decentralized execution)
        self.actors = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.target_actors = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=0.001) for actor in self.actors]

        # One centralized critic (CTDE)
        total_obs_dim = num_agents * obs_dim
        total_action_dim = num_agents * action_dim
        self.critic = CriticNetwork(total_obs_dim, total_action_dim).to(device)
        self.target_critic = CriticNetwork(total_obs_dim, total_action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.002)

        # Copy actor and critic params to targets
        self._init_target_networks()

        # Replay Buffer
        self.buffer = ReplayBuffer(max_size=100000, num_agents=num_agents, obs_dim=obs_dim, action_dim=action_dim)

        # Noise for exploration
        self.noise = [OUNoise(action_dim) for _ in range(num_agents)]

        # Hyperparameters
        self.gamma = 0.95
        self.tau = 0.01

    def _init_target_networks(self):
        for target_actor, actor in zip(self.target_actors, self.actors):
            target_actor.load_state_dict(actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def select_action(self, obs, noise=True):
        """
        obs: shape (num_agents, obs_dim)
        Returns: actions (num_agents, action_dim)
        """
        actions = []
        for i, actor in enumerate(self.actors):
            obs_tensor = torch.tensor(obs[i], dtype=torch.float32).to(self.device)
            action = actor(obs_tensor).detach().cpu().numpy()
            if noise:
                action += self.noise[i].sample()
            actions.append(np.clip(action, -1, 1))
        return np.array(actions)

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return  # Not enough samples

        # Sample from buffer
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = self.buffer.sample(batch_size)

        # Convert to torch tensors
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)     # (B, N, obs_dim)
        act_tensor = torch.tensor(act_batch, dtype=torch.float32).to(self.device)     # (B, N, action_dim)
        next_obs_tensor = torch.tensor(next_obs_batch, dtype=torch.float32).to(self.device)
        rew_tensor = torch.tensor(rew_batch, dtype=torch.float32).to(self.device)
        done_tensor = torch.tensor(done_batch, dtype=torch.float32).to(self.device)

        # Centralized critic input: flatten all obs and actions
        obs_flat = obs_tensor.view(batch_size, -1)
        act_flat = act_tensor.view(batch_size, -1)

        # Next actions by target actors
        next_actions = []
        for i, target_actor in enumerate(self.target_actors):
            next_act = target_actor(next_obs_tensor[:, i, :])
            next_actions.append(next_act)
        next_act_tensor = torch.stack(next_actions, dim=1)  # (B, N, action_dim)
        next_act_flat = next_act_tensor.view(batch_size, -1)

        # Compute target Q value
        with torch.no_grad():
            target_q = self.target_critic(next_obs_tensor.view(batch_size, -1), next_act_flat)
            target_value = rew_tensor.sum(dim=1, keepdim=True) + self.gamma * target_q * (1 - done_tensor.sum(dim=1, keepdim=True))

        # Critic Update
        current_q = self.critic(obs_flat, act_flat)
        critic_loss = F.mse_loss(current_q, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update (decentralized execution, centralized gradients via critic)
        for i, actor in enumerate(self.actors):
            current_actions = []
            for j, a in enumerate(self.actors):
                if i == j:
                    a_input = obs_tensor[:, j, :]
                    current_actions.append(a(a_input))
                else:
                    current_actions.append(act_tensor[:, j, :].detach())
            all_actions = torch.stack(current_actions, dim=1)
            all_actions_flat = all_actions.view(batch_size, -1)

            actor_loss = -self.critic(obs_flat, all_actions_flat).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # Soft update target networks
        for actor, target_actor in zip(self.actors, self.target_actors):
            soft_update(target_actor, actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    def store(self, obs, actions, rewards, next_obs, dones):
        self.buffer.add(obs, actions, rewards, next_obs, dones)

    def reset_noise(self):
        for n in self.noise:
            n.reset()
