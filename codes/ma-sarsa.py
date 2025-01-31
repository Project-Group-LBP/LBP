# Multi-agent independent SARSA learning
# where multiple agents collaborate for maximum coverage of the grid while avoiding obstacles
# (Reward of an agent is affected by positions of other agents)

import numpy as np
import random
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
    

class MultiAgentSARSA:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        
    def get_q(self, state, agent_id):
        state_tuple = tuple(state.flatten())
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros((self.env.num_agents, 4))
        return self.q_table[state_tuple][agent_id]
    
    def choose_action(self, state, agent_id):
        if np.random.rand() < self.epsilon:
            return random.choice(range(4))
        return np.argmax(self.get_q(state, agent_id))
    
    def train(self, episodes=1000):
        rewards_per_episode = []
        rolling_avg_rewards = []
        
        for episode in range(episodes):
            state = self.env.reset()
            actions = [self.choose_action(state, agent_id) for agent_id in range(self.env.num_agents)]
            total_reward = 0
            
            for _ in range(MAX_STEPS):
                next_state, rewards, done = self.env.step(actions)
                next_actions = [self.choose_action(next_state, agent_id) for agent_id in range(self.env.num_agents)]
                
                for agent_id in range(self.env.num_agents):
                    q_values = self.get_q(state, agent_id)
                    next_q_values = self.get_q(next_state, agent_id)
                    q_values[actions[agent_id]] += self.alpha * (
                        rewards[agent_id] + self.gamma * next_q_values[next_actions[agent_id]] - q_values[actions[agent_id]]
                    )
                    
                state, actions = next_state, next_actions
                total_reward += sum(rewards)
                
                if done:
                    break
            
            rewards_per_episode.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # Calculate and store rolling average reward
            if len(rewards_per_episode) >= 50:
                rolling_avg_rewards.append(
                    np.mean(rewards_per_episode[-50:])
                )  # Average of the last 100 rewards
            else:
                rolling_avg_rewards.append(np.mean(rewards_per_episode))
            
        return rewards_per_episode, rolling_avg_rewards
    
    def test(self, episodes=100):
        total_rewards = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            for _ in range(MAX_STEPS):
                actions = [np.argmax(self.get_q(state, agent_id)) for agent_id in range(self.env.num_agents)]
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
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [2, 1]})
    
    axs[0].plot(np.arange(1, len(train_rewards) + 1), train_rewards, label='Rewards per Episode', alpha=0.5)
    axs[0].plot(np.arange(1, len(train_rewards) + 1), rolling_avg_rewards, label='Rolling Average Rewards (per 50 episodes)', color="red")
    axs[0].axhline(y=avg_train_reward, color="green", linestyle="--", label=f"Overall Avg: {avg_train_reward:.2f}")
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Total Reward')
    axs[0].set_title('Training Progress')
    axs[0].legend()
    axs[0].grid()
    
    axs[1].bar(['Training', 'Testing'], [avg_train_reward, avg_test_reward], color=['blue', 'orange'])
    axs[1].set_ylabel('Average Reward')
    axs[1].set_title('Avg Training vs Avg Testing Reward')

    # Annotate the bars with exact values
    for i, v in enumerate([avg_train_reward, avg_test_reward]):
        axs[1].text(i, v + 1, f"{v:.2f}", ha="center", fontsize=10, color="black")

    axs[1].grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    env = Environment(GRID_SIZE, NUM_AGENTS, NUM_OBSTACLES)
    agents = MultiAgentSARSA(env, LEARNING_RATE, GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN)
    
    print("Training in progress...")
    train_rewards, rolling_avg_rewards = agents.train(1000)

    print("Testing using target Q-table...")
    test_rewards = agents.test(100)

    avg_train_reward = np.mean(train_rewards)
    avg_test_reward = np.mean(test_rewards)

    print(f"Average Training Reward: {avg_train_reward}")
    print(f"Average Testing Reward: {avg_test_reward}")
    
    plot_results(train_rewards, rolling_avg_rewards, test_rewards)
