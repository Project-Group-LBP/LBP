import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt


class SARSAAgent:
    def __init__(self, env, alpha, gamma, epsilon):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lower_bounds = self.env.observation_space.low
        self.upper_bounds = self.env.observation_space.high
        num_bins = 10
        self.state_bins = []
        for i in range(len(self.upper_bounds)):
            self.state_bins.append(np.linspace(self.lower_bounds[i], self.upper_bounds[i], num_bins))

        # Initialize the Q-table with zeros
        self.Q = np.zeros([len(bins) + 1 for bins in self.state_bins] + [self.env.action_space.n])

    def discretize_state(self, state):
        state = np.array(state)
        state = np.clip(state, self.lower_bounds, self.upper_bounds)
        state_indices = []
        for i, bins in enumerate(self.state_bins):
            index = np.digitize(state[i], bins) - 1
            index = np.clip(index, 0, len(bins) - 1)
            state_indices.append(index)
        return tuple(state_indices)

    def choose_action(self, state):  # Behaviour policy: Epsilon-greedy
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    # SARSA algorithm
    def train_agent(self, num_train_episodes, max_steps):
        episode_rewards = []
        rolling_avg_rewards = []

        for episode in range(num_train_episodes):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            action = self.choose_action(state)
            total_reward = 0

            for _ in range(max_steps):
                # Take action and observe reward and next state
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_state)

                # Choose next action using epsilon-greedy policy
                next_action = self.choose_action(next_state)

                # Update Q-table using the SARSA update rule
                self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])

                state, action = next_state, next_action  # Transition to next state and action
                total_reward += reward

                if done or truncated:  # If the episode ends
                    break

            # Store reward for the episode
            episode_rewards.append(total_reward)

            # Calculate and store rolling average reward
            if len(episode_rewards) >= 100:
                rolling_avg_rewards.append(np.mean(episode_rewards[-100:]))  # Average of the last 100 rewards
            else:
                rolling_avg_rewards.append(np.mean(episode_rewards))

            # Print the progress every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = rolling_avg_rewards[-1]
                print(f"Episode {episode + 1}/{num_train_episodes}, Total Reward: {total_reward}, " f"Average Reward (last 100): {avg_reward:.2f}")

        overall_training_avg = np.mean(episode_rewards)
        print(f"Average Training Reward: {overall_training_avg:.2f}")

        return episode_rewards, rolling_avg_rewards, overall_training_avg

    def test_agent(self, num_test_episodes, max_steps):
        episode_rewards = []
        for episode in range(num_test_episodes):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            total_reward = 0

            for _ in range(max_steps):
                action = np.argmax(self.Q[state])  # Test with greedy policy
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_state)

                state = next_state
                total_reward += reward

                if done or truncated:
                    break

            episode_rewards.append(total_reward)
            print(f"Test Episode {episode + 1} Reward: {total_reward}")

        overall_test_avg = np.mean(episode_rewards)
        print(f"Average Test Reward: {overall_test_avg:.2f}")

        return overall_test_avg

    def plot_graph(self, episode_rewards, rolling_avg_rewards, training_avg, testing_avg):
        episodes = np.arange(1, len(episode_rewards) + 1)
        overall_avg_reward = np.mean(episode_rewards)  # Calculate overall average reward for training

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [2, 1]})

        # First subplot: Reward vs. Number of Episodes (Training)
        plt1 = axes[0]
        plt1.plot(episodes, episode_rewards, label="Reward per Episode", alpha=0.5)
        plt1.plot(episodes, rolling_avg_rewards, label="Rolling Avg (Last 100 Episodes)", color="red")
        plt1.axhline(y=overall_avg_reward, color="green", linestyle="--", label=f"Overall Avg: {overall_avg_reward:.2f}")
        plt1.set_title("Reward Gained vs Number of Training Episodes")
        plt1.set_xlabel("Number of Training Episodes")
        plt1.set_ylabel("Reward")
        plt1.legend()
        plt1.grid()

        # Second subplot: Comparison of Training and Testing Averages
        plt2 = axes[1]
        categories = ["Training", "Testing"]
        values = [training_avg, testing_avg]

        plt2.bar(categories, values, color=["blue", "orange"], alpha=0.7, width=0.5)
        plt2.set_title("Comparison of Average Rewards: Training vs Testing")
        plt2.set_ylabel("Average Reward")
        plt2.set_ylim(0, max(values) + 10)  # Adjust y-axis for better visibility

        # Annotate the bars with exact values
        for i, v in enumerate(values):
            plt2.text(i, v + 1, f"{v:.2f}", ha="center", fontsize=10, color="black")

        plt2.grid(axis="y", linestyle="--", alpha=0.7)

        # Show the combined figure
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Using CartPole for environment
    env = gym.make("CartPole-v1")

    # SARSA parameters
    num_train_episodes = 1000
    max_steps = 200
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    num_test_episodes = 10

    agent = SARSAAgent(env, alpha, gamma, epsilon)
    episode_rewards, rolling_avg_rewards, training_avg = agent.train_agent(num_train_episodes, max_steps)
    testing_avg = agent.test_agent(num_test_episodes, max_steps)
    agent.plot_graph(episode_rewards, rolling_avg_rewards, training_avg, testing_avg)
    env.close()
