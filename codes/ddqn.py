import numpy as np
import random
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

# Environment parameters
grid_size = 10
num_episodes = 10
max_steps = 50
learning_rate = 0.001
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.1

# Agent and obstacle parameters
num_agents = 3
num_obstacles = 15

class Environment:
    def __init__(self, grid_size, num_agents, num_obstacles):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_obstacles = num_obstacles
        self.visited = np.zeros((grid_size, grid_size))
        self.visited_counts = np.zeros((grid_size, grid_size))
        self.reset()

    def reset(self):
        # Reset positions and visited array
        self.agent_positions = {i: (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)) for i in range(self.num_agents)}
        self.obstacle_positions = [(random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)) for _ in range(self.num_obstacles)]
        self.visited = np.zeros((self.grid_size, self.grid_size))
        self.visited_counts = np.zeros((self.grid_size, self.grid_size))
        self.state = self._get_state()
        return self.state

    def step(self, actions):
        rewards = []
        for agent_id, action in enumerate(actions):
            new_pos = self._move(self.agent_positions[agent_id], action)
            if new_pos not in self.obstacle_positions:
                self.agent_positions[agent_id] = new_pos
                reward = self._calculate_reward(new_pos, agent_id)
                if self.visited[new_pos] == 0:
                    self.visited[new_pos] = 1  # Mark cell as visited
                    reward += 10  # Higher reward for visiting new cells
                else:
                    reward += 1  # Small reward for moving

                self.visited_counts[new_pos] += 1
            else:
                reward = -1  # Penalty for hitting an obstacle (minimal)
            rewards.append(reward)
       
        self.state = self._get_state()
        done = np.all(self.visited)  # Check if all cells are visited
        return self.state, rewards, done

    def _calculate_reward(self, new_pos, agent_id):
        # Reduce penalties to avoid negative rewards
        distance_penalty = 0
        for other_id, pos in self.agent_positions.items():
            if other_id != agent_id:
                dist = np.sqrt((new_pos[0] - pos[0]) ** 2 + (new_pos[1] - pos[1]) ** 2)
                distance_penalty += -0.05 * (1 / (dist + 1))  # Reduced distance penalty factor
       
        # Reward based on sigmoid of visit count
        visit_count = self.visited_counts[new_pos]
        sigmoid_penalty = -0.05 * (1 / (1 + np.exp(-visit_count + 2)))  # Reduced revisit penalty factor

        # Base positive reward for any movement to encourage exploration
        base_reward = 1.0
       
        # Combined reward
        combined_reward = base_reward + distance_penalty + sigmoid_penalty
        return combined_reward

    def _move(self, position, action):
        x, y = position
        if action == 0:   # Up
            return (max(x - 1, 0), y)
        elif action == 1:  # Down
            return (min(x + 1, self.grid_size - 1), y)
        elif action == 2:  # Left
            return (x, max(y - 1, 0))
        elif action == 3:  # Right
            return (x, min(y + 1, self.grid_size - 1))
        return position

    def _get_state(self):
        state = np.zeros((self.grid_size, self.grid_size))
        for pos in self.obstacle_positions:
            state[pos] = -1  # Obstacles
        for agent_id, pos in self.agent_positions.items():
            state[pos] = agent_id + 1  # Agents
        return state

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = epsilon
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(self.state_size, self.state_size)))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)
                target[0][action] = reward + gamma * np.amax(t[0])
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

# Initialize environment and agent
env = Environment(grid_size, num_agents, num_obstacles)
state_size = grid_size
action_size = 4  # Up, Down, Left, Right
agents = [DDQNAgent(state_size, action_size) for _ in range(num_agents)]

# Initialize data for plotting
reward_per_episode = []
coverage_per_episode = []
max_steps_history = []

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    total_rewards = [0] * num_agents
    cumulative_rewards_per_step = []
    coverage_per_step = []
   
    for step in range(max_steps):
        print(f"Episode {episode + 1}, Step {step + 1}")
        actions = [agent.act(np.reshape(state, [1, state_size, state_size])) for agent in agents]
        next_state, rewards, done = env.step(actions)
        cumulative_reward = sum(rewards) if step == 0 else cumulative_rewards_per_step[-1] + sum(rewards)
        cumulative_rewards_per_step.append(cumulative_reward)  # Collect cumulative rewards for plotting
        coverage = np.sum(env.visited) / (grid_size * grid_size) * 100
        coverage_per_step.append(coverage)
        print(f"Coverage after Step {step + 1}: {coverage:.2f}%")
       
        for i, agent in enumerate(agents):
            agent.remember(np.reshape(state, [1, state_size, state_size]), actions[i], rewards[i], np.reshape(next_state, [1, state_size, state_size]), done)
            total_rewards[i] += rewards[i]
        state = next_state
        if done:
            break
        for agent in agents:
            agent.replay(32)
    for agent in agents:
        agent.update_target_model()
   
    reward_per_episode.append(cumulative_rewards_per_step)
    coverage_per_episode.append(coverage_per_step)
    print(f"Episode {episode + 1}: Total Rewards: {total_rewards}")
    print(f"Steps taken in Episode {episode + 1}: {len(cumulative_rewards_per_step)}")

plt.figure(figsize=(10, 6))
for episode in range(num_episodes):
    plt.plot(reward_per_episode[episode], label=f'Episode {episode + 1}')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward vs. Steps for Each Episode')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for episode in range(num_episodes):
    plt.plot(coverage_per_episode[episode], label=f'Episode {episode + 1}')
plt.xlabel('Steps')
plt.ylabel('Coverage (%)')
plt.title('Coverage vs. Steps for Each Episode')
plt.legend()
plt.show()
