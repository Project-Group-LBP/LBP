import numpy as np
import random
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

# Parameters
GRID_SIZE = 10
NUM_EPISODES = 10
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
        return (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

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
            (x, min(y + 1, self.grid_size - 1))
        ][action]

    def _get_state(self):
        state = np.zeros((self.grid_size, self.grid_size))
        for pos in self.obstacle_positions:
            state[pos] = -1
        for agent_id, pos in self.agent_positions.items():
            state[pos] = agent_id + 1
        return state

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = EPSILON
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.state_size, self.state_size)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                # t = self.target_model.predict(next_state)
                # target[0][action] = reward + GAMMA * np.amax(t[0])
                next_action = np.argmax(self.model.predict(next_state, verbose=0))
                target[0][action] = reward + GAMMA * self.target_model.predict(next_state, verbose=0)[0][next_action]
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

def initialize_environment():
    return Environment(GRID_SIZE, NUM_AGENTS, NUM_OBSTACLES)

def initialize_agents():
    return [DDQNAgent(GRID_SIZE, 4) for _ in range(NUM_AGENTS)]

def execute_episode(env, agents):
    state = env.reset()
    total_rewards = [0] * NUM_AGENTS
    episode_rewards, episode_coverage = [], []

    for step in range(MAX_STEPS):
        actions = [agent.act(np.reshape(state, [1, GRID_SIZE, GRID_SIZE])) for agent in agents]
        next_state, rewards, done = env.step(actions)
        for i, agent in enumerate(agents):
            agent.remember(np.reshape(state, [1, GRID_SIZE, GRID_SIZE]), actions[i], rewards[i], np.reshape(next_state, [1, GRID_SIZE, GRID_SIZE]), done)
            total_rewards[i] += rewards[i]
        state = next_state
        episode_rewards.append(sum(rewards))
        episode_coverage.append(np.sum(env.visited) / (GRID_SIZE * GRID_SIZE) * 100)
        if done:
            break
        for agent in agents:
            agent.replay()

    for agent in agents:
        agent.update_target_model()

    return total_rewards, episode_rewards, episode_coverage

def train_agents():
    env = initialize_environment()
    agents = initialize_agents()
    rewards_history, coverage_history = [], []

    for episode in range(NUM_EPISODES):
        total_rewards, episode_rewards, episode_coverage = execute_episode(env, agents)
        rewards_history.append(episode_rewards)
        coverage_history.append(episode_coverage)
        print(f"Episode {episode + 1}: Total Rewards: {total_rewards}")

    return rewards_history, coverage_history

def plot_results(rewards_history, coverage_history):
    plt.figure(figsize=(10, 6))
    for episode_rewards in rewards_history:
        plt.plot(episode_rewards)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward vs. Steps')
    plt.show()

    plt.figure(figsize=(10, 6))
    for episode_coverage in coverage_history:
        plt.plot(episode_coverage)
    plt.xlabel('Steps')
    plt.ylabel('Coverage (%)')
    plt.title('Coverage vs. Steps')
    plt.show()

if __name__ == "__main__":
    rewards_history, coverage_history = train_agents()
    plot_results(rewards_history, coverage_history)
