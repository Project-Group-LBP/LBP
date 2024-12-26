import gymnasium as gym
import numpy as np
import random
from collections import deque
from tensorflow import keras


class DDQNAgent:
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
        if training and np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0][0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3][0] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Get the next action to take based on the current model
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)

        # Q(s', a) using target network (for the action selected by the main network)
        target_next = self.target_model.predict(next_states, verbose=0)
        target_q = rewards + self.gamma * target_next[ np.arange(batch_size), next_actions ] * (1 - dones)

        # Q(s, a) using main network
        target_f = self.model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            target_f[i][action] = target_q[i]

        self.model.fit(states, target_f, epochs=1, verbose=0)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_ddqn(self, episodes, max_steps, batch_size):
        scores = []
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
                print(
                    f"Episode: {episode + 1}/{episodes}, Score: {score}",
                    f"  Average Score (last 10): {avg_score:.2f}"
                    f"  Epsilon: {self.epsilon}",
                )

            # Update target network periodically
            if episode % self.update_target_frequency == 0:
                self.update_target_network()

        print(f"\nAverage Overall Score: {np.mean(scores):.2f}\n")

    def test_ddqn(self, episodes, max_steps):
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

        print(f"Average Test Score: {np.mean(scores):.2f}")


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

    agent = DDQNAgent(
        env, learning_rate, gamma, epsilon_initial, epsilon_min, epsilon_decay
    )
    agent.train_ddqn(num_train_episodes, max_steps, batch_size)
    agent.test_ddqn(num_test_episodes, max_steps)
    env.close()
