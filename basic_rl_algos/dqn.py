import gymnasium as gym
import numpy as np
import random
from collections import deque
from tensorflow import keras
import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self, env, lr, gamma, eps_init, eps_min, eps_decay):
        self.env = env
        self.learning_rate = lr
        self.gamma = gamma
        self.epsilon = eps_init
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        # Experience replay buffer
        self.memory = deque(maxlen=2000)

        # Main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.update_target_frequency = 5  # Update target network every n episodes

    def _build_model(self):
        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=(self.state_size,)),
                keras.layers.Dense(24, activation="relu"),
                keras.layers.Dense(24, activation="relu"),
                keras.layers.Dense(self.action_size, activation="linear"),
            ]
        )
        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Training : Behaviour policy : epsilon-greedy
    # Testing : Target policy : greedy
    def act(self, state, training):
        # Exploration
        if training and np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        # Exploitation
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    # Vectorized operations
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0][0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3][0] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Q(s', a) using target network
        target_next = self.target_model.predict(next_states, verbose=0)
        target_q = rewards + self.gamma * np.amax(target_next, axis=1) * (1 - dones)

        # Q(s, a) using main network
        target_f = self.model.predict(states, verbose=0)
        target_f[np.arange(batch_size), actions] = target_q

        self.model.fit(states, target_f, epochs=1, verbose=0)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_dqn(self, episodes, max_steps, batch_size):
        scores = []
        rolling_avg_scores = []

        for episode in range(episodes):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            score = 0

            for _ in range(max_steps):
                action = self.act(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                score += reward
                self.replay(batch_size)

                if done:
                    break

            scores.append(score)
            # Print the progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_score = np.mean(scores[-10:])
                rolling_avg_scores.append(avg_score)
                print(
                    f"Episode: {episode + 1}/{episodes}, Score: {score}",
                    f"  Average Score (last 10): {avg_score:.2f}"
                    f"  Epsilon: {self.epsilon}",
                )

            # Update target network periodically
            if episode % self.update_target_frequency == 0:
                self.update_target_network()

        overall_training_avg = np.mean(scores)
        print(f"\nAverage Training Score: {overall_training_avg:.2f}\n")
        return scores, rolling_avg_scores, overall_training_avg

    def test_dqn(self, episodes, max_steps):
        scores = []
        for episode in range(episodes):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            score = 0

            for _ in range(max_steps):
                action = self.act(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                state = next_state
                score += reward
                if terminated or truncated:
                    break

            scores.append(score)
            print(f"Test Episode: {episode+1}, Score: {score}")

        overall_test_avg = np.mean(scores)
        print(f"Average Test Score: {overall_test_avg:.2f}")
        return overall_test_avg

    def plot_graph(self, scores, rolling_avg_scores, training_avg, test_avg):
        episodes = np.arange(1, len(scores) + 1)
        rolling_avg_episodes = np.arange(10, len(scores) + 1, 10)

        overall_avg_score = np.mean(
            scores
        )  # Calculate overall average score for training

        # Create a figure with subplots
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [2, 1]}
        )

        # First subplot: Scores vs. Number of Episodes (Training)
        plt1 = axes[0]
        plt1.plot(episodes, scores, label="Score per Episode", alpha=0.5)
        plt1.plot(
            rolling_avg_episodes,
            rolling_avg_scores,
            label="Rolling Avg (Last 10 Episodes)",
            color="red",
        )
        plt1.axhline(
            y=overall_avg_score,
            color="green",
            linestyle="--",
            label=f"Overall Avg: {overall_avg_score:.2f}",
        )
        plt1.set_title("Score Gained vs Number of Training Episodes")
        plt1.set_xlabel("Number of Training Episodes")
        plt1.set_ylabel("Score")
        plt1.legend()
        plt1.grid()

        # Second subplot: Comparison of Training and Testing Averages
        plt2 = axes[1]
        categories = ["Training", "Testing"]
        values = [training_avg, test_avg]

        plt2.bar(categories, values, color=["blue", "orange"], alpha=0.7, width=0.5)
        plt2.set_title("Comparison of Average Scores: Training vs Testing")
        plt2.set_ylabel("Average Score")
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

    # Parameters
    num_train_episodes = 100
    max_steps = 200
    learning_rate = 0.001
    gamma = 0.95
    epsilon_initial = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 32
    num_test_episodes = 10

    agent = DQNAgent(
        env, learning_rate, gamma, epsilon_initial, epsilon_min, epsilon_decay
    )
    scores, rolling_avg_scores, training_avg = agent.train_dqn(
        num_train_episodes, max_steps, batch_size
    )
    test_avg = agent.test_dqn(num_test_episodes, max_steps)
    agent.plot_graph(scores, rolling_avg_scores, training_avg, test_avg)
    env.close()
