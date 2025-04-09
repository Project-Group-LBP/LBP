import torch
import torch.nn.functional as F
import numpy as np
import os
from .agents import ActorNetwork, QuantileCriticNetwork, soft_update, GaussianNoise
from .buffer import ReplayBuffer

ALPHA = 0.5  # CVaR quantile level

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    def __init__(self, num_agents, obs_shape, action_dim, hidden_dim=160, gamma=0.83, tau=0.01, actor_lr=1e-3, critic_lr=1e-3, device=device):
        self.num_agents = num_agents
        self.device = device

        # CNN-based actors and critics
        self.obs_dim = obs_shape
        self.grid_width, self.grid_height, self.input_channels = self.obs_dim

        # Initialize actor networks
        self.actors = [ActorNetwork(action_dim=action_dim, hidden_dim=hidden_dim).to(device) for _ in range(num_agents)]
        self.target_actors = [ActorNetwork(action_dim=action_dim, hidden_dim=hidden_dim).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]

        # Calculate feature dimension for critic
        self.total_feature_dim = num_agents * self.actors[0].cnn.feature_dim
        self.total_action_dim = num_agents * action_dim

        # Initialize critic networks
        self.critics = [QuantileCriticNetwork(self.total_feature_dim, self.total_action_dim, hidden_dim).to(device) for _ in range(num_agents)]
        self.target_critics = [QuantileCriticNetwork(self.total_feature_dim, self.total_action_dim, hidden_dim).to(device) for _ in range(num_agents)]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]

        # Initialize target networks
        self._init_target_networks()

        # Replay buffer
        self.buffer = ReplayBuffer(max_size=1000000, num_agents=num_agents, obs_dim=self.obs_dim, action_dim=action_dim)

        # Exploration noise
        self.noise = [GaussianNoise(action_dim) for _ in range(num_agents)]

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau

    def _init_target_networks(self):
        for target_actor, actor in zip(self.target_actors, self.actors):
            target_actor.load_state_dict(actor.state_dict())
        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())

    def select_action(self, obs, noise=True):
        actions = []

        for i, actor in enumerate(self.actors):
            obs_tensor = torch.tensor(obs[i], dtype=torch.float32).to(self.device)

            # CNN expects (batch, channels, height, width)
            if len(obs_tensor.shape) == 3:  # (height, width, channels)
                obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, channels, height, width)

            with torch.no_grad():
                action = actor(obs_tensor)

            # Handle single action without batch dimension
            if len(action.shape) > 1:
                action = action[0]  # Take first element if batch dimension exists

            if noise:
                action += self.noise[i].sample()

            actions.append(torch.clamp(action, -1, 1))

        actions = torch.stack(actions)
        return actions.detach().cpu().numpy()

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return  # Not enough samples

        for i in range(self.num_agents):
            # Sample from buffer
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = self.buffer.sample(batch_size)

            # Convert to torch tensors and ensure contiguous memory layout
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(self.device).contiguous()
            act_tensor = torch.tensor(act_batch, dtype=torch.float32).to(self.device).contiguous()
            next_obs_tensor = torch.tensor(next_obs_batch, dtype=torch.float32).to(self.device).contiguous()
            rew_tensor = torch.tensor(rew_batch, dtype=torch.float32).to(self.device).contiguous()
            done_tensor = torch.tensor(done_batch, dtype=torch.float32).to(self.device).contiguous()

            # Process observations through CNN for each agent
            all_obs_features = []
            all_next_obs_features = []

            for j in range(self.num_agents):
                # Process current observations
                # Reshape for CNN: (batch, agent, height, width, channels) -> (batch, channels, height, width)
                agent_obs = obs_tensor[:, j].permute(0, 3, 1, 2).contiguous()
                agent_features = self.actors[j].cnn(agent_obs)
                all_obs_features.append(agent_features)

                # Process next observations
                agent_next_obs = next_obs_tensor[:, j].permute(0, 3, 1, 2).contiguous()
                agent_next_features = self.target_actors[j].cnn(agent_next_obs)
                all_next_obs_features.append(agent_next_features)

            # Concatenate features from all agents
            obs_flat = torch.cat(all_obs_features, dim=1)
            next_obs_flat = torch.cat(all_next_obs_features, dim=1)

            # Flatten actions
            act_flat = act_tensor.reshape(batch_size, -1)

            # Get next actions from target actors
            next_actions = []
            for j, target_actor in enumerate(self.target_actors):
                # Process grid observation
                next_obs_j = next_obs_tensor[:, j].permute(0, 3, 1, 2).contiguous()
                next_agent_action = target_actor(next_obs_j)
                next_actions.append(next_agent_action)

            next_actions_tensor = torch.stack(next_actions, dim=1)
            next_actions_flat = next_actions_tensor.reshape(batch_size, -1)

            # Compute predicted quantile distribution
            critic_output = self.critics[i](next_obs_flat, next_actions_flat)  # [batch_size, num_quantiles]
            critic_sorted, _ = torch.sort(critic_output, dim=1)
            cvar_pred = torch.mean(critic_sorted[:, : int(ALPHA * critic_output.size(1))], dim=1, keepdim=True)  # [batch_size, 1]

            # Calculate target Q-value
            with torch.no_grad():
                target_output = self.target_critics[i](next_obs_flat, next_actions_flat)
                target_sorted, _ = torch.sort(target_output, dim=1)
                cvar_target = torch.mean(target_sorted[:, : int(ALPHA * critic_output.size(1))], dim=1, keepdim=True)

            # Current Q-value
            critic_loss = F.mse_loss(cvar_pred, cvar_target)

            # Critic loss
            critic_loss = F.mse_loss(cvar_pred, cvar_target)
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # Create action inputs where only the current agent's action comes from the actor
            current_actions = []
            for j in range(self.num_agents):
                if j == i:
                    # Process grid observation
                    obs_i = obs_tensor[:, i].permute(0, 3, 1, 2).contiguous()
                    current_agent_action = self.actors[i](obs_i)
                    current_actions.append(current_agent_action)
                else:
                    current_actions.append(act_tensor[:, j].detach())

            current_actions_tensor = torch.stack(current_actions, dim=1)
            current_actions_flat = current_actions_tensor.reshape(batch_size, -1)

            # Policy gradient
            actor_loss = -self.critics[i](obs_flat.detach(), current_actions_flat).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()  # Retain graph for critic update
            self.actor_optimizers[i].step()

            # Soft update target networks
            soft_update(self.target_critics[i], self.critics[i], self.tau)
            soft_update(self.target_actors[i], self.actors[i], self.tau)

        # Decay noise scale after each update
        self.decay_noise()

    def store(self, obs, actions, rewards, next_obs, dones):
        self.buffer.add(obs, actions, rewards, next_obs, dones)

    def decay_noise(self):
        for n in self.noise:
            n.decay()

    def reset_noise(self):
        for n in self.noise:
            n.reset()

    def save(self, path):
        for i in range(self.num_agents):
            # Save actor, target actor, and actor optimizer
            torch.save(
                {
                    "actor": self.actors[i].state_dict(),
                    "target_actor": self.target_actors[i].state_dict(),
                    "actor_optimizer": self.actor_optimizers[i].state_dict(),
                    "critic": self.critics[i].state_dict(),
                    "target_critic": self.target_critics[i].state_dict(),
                    "critic_optimizer": self.critic_optimizers[i].state_dict(),
                },
                f"{path}/agent_{i}.pth",
            )

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model directory not found: {path}")

        for i in range(self.num_agents):
            agent_path = f"{path}/agent_{i}.pth"
            if not os.path.exists(agent_path):
                raise FileNotFoundError(f"❌ Model file not found: {agent_path}")

            checkpoint = torch.load(agent_path, map_location=self.device)

            # Load actor networks and optimizer
            self.actors[i].load_state_dict(checkpoint["actor"])
            self.target_actors[i].load_state_dict(checkpoint["target_actor"])
            self.actor_optimizers[i].load_state_dict(checkpoint["actor_optimizer"])

            # Load critic networks and optimizer
            self.critics[i].load_state_dict(checkpoint["critic"])
            self.target_critics[i].load_state_dict(checkpoint["target_critic"])
            self.critic_optimizers[i].load_state_dict(checkpoint["critic_optimizer"])

        print(f"📥 Models loaded successfully from {path}\n")
