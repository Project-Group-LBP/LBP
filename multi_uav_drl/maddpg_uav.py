import torch
import numpy as np
import torch.nn.functional as F
from agents import ActorNetwork, CriticNetwork, soft_update, OUNoise
from buffer import ReplayBuffer


class MADDPG:
    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=160, device="cpu"):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        total_obs_dim = num_agents * obs_dim
        total_action_dim = num_agents * action_dim

        self.actors = [ActorNetwork(obs_dim, action_dim, hidden_dim).to(device) for _ in range(num_agents)]
        self.target_actors = [ActorNetwork(obs_dim, action_dim, hidden_dim).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=0.001) for actor in self.actors]
        self.critics = [CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(device) for _ in range(num_agents)]
        self.target_critics = [CriticNetwork(total_obs_dim, total_action_dim, hidden_dim).to(device) for _ in range(num_agents)]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=0.002) for critic in self.critics]

        # Copy weights to target networks
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
        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())

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

        for i in range(self.num_agents):

            # Sample from buffer
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = self.buffer.sample(batch_size)

            # Convert to torch tensors
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)
            act_tensor = torch.tensor(act_batch, dtype=torch.float32).to(self.device)
            next_obs_tensor = torch.tensor(next_obs_batch, dtype=torch.float32).to(self.device)
            rew_tensor = torch.tensor(rew_batch, dtype=torch.float32).to(self.device)
            done_tensor = torch.tensor(done_batch, dtype=torch.float32).to(self.device)

            obs_flat = obs_tensor.view(batch_size, -1)
            act_flat = act_tensor.view(batch_size, -1)
            next_obs_flat = next_obs_tensor.view(batch_size, -1)

            # Get next actions from target actors
            next_actions = []
            for x, target_actor in enumerate(self.target_actors):
                next_agent_action = target_actor(next_obs_tensor[:, x, :])
                next_actions.append(next_agent_action)
            next_actions_tensor = torch.stack(next_actions, dim=1)
            next_actions_flat = next_actions_tensor.view(batch_size, -1)

            # Calculate target Q-value
            with torch.no_grad():
                target_q = self.target_critics[i](next_obs_flat, next_actions_flat)
                target_value = rew_tensor[:, i].unsqueeze(1) + self.gamma * target_q * (1 - done_tensor[:, i].unsqueeze(1))

            # Current Q-value
            current_q = self.critics[i](obs_flat, act_flat)

            # Critic loss
            critic_loss = F.mse_loss(current_q, target_value)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # Create action inputs where only the current agent's action comes from the actor
            current_actions = []
            for j in range(self.num_agents):
                if j == i:
                    current_agent_action = self.actors[i](obs_tensor[:, i, :])
                    current_actions.append(current_agent_action)
                else:
                    current_actions.append(act_tensor[:, j, :].detach())

            current_actions_tensor = torch.stack(current_actions, dim=1)
            current_actions_flat = current_actions_tensor.view(batch_size, -1)

            # Policy gradient
            actor_loss = -self.critics[i](obs_flat, current_actions_flat).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            # Soft update target networks
            soft_update(self.target_critics[i], self.critics[i], self.tau)
            soft_update(self.target_actors[i], self.actors[i], self.tau)

    def store(self, obs, actions, rewards, next_obs, dones):
        self.buffer.add(obs, actions, rewards, next_obs, dones)

    def reset_noise(self):
        for n in self.noise:
            n.reset()
