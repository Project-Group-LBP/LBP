# DDPG implementation in keras and tensorflow

import tensorflow as tf
from tensorflow import keras
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Pendulum-v1")
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]


class OUActionNoise:
    def __init__(self, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.mu = np.ones(num_actions) * mu
        self.theta = theta
        self.sigma = np.ones(num_actions) * sigma
        self.dt = dt
        self.reset()

    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.state = x + dx
        return self.state

    def reset(self):
        self.state = np.copy(self.mu)


def get_actor_model():
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    # Initialization for last layer to be between -0.003 and 0.003
    # to tackle vanishing gradients problem associated with tanh activation fn
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(num_states,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init),
            keras.layers.Lambda(lambda x: x * upper_bound),
        ]
    )
    return model


def get_critic_model():
    state_input = keras.layers.Input(shape=(num_states,))
    state_output = keras.layers.Dense(32, activation="relu")(state_input)
    action_input = keras.layers.Input(shape=(num_actions,))
    action_output = keras.layers.Dense(32, activation="relu")(action_input)

    concat = keras.layers.Concatenate()([state_output, action_output])
    combined_layer_1 = keras.layers.Dense(256, activation="relu")(concat)
    combined_layer_2 = keras.layers.Dense(256, activation="relu")(combined_layer_1)
    output = keras.layers.Dense(1)(combined_layer_2)
    return keras.Model([state_input, action_input], output)


# Polyak averaging
def update_target(target, learned, tau):
    target_weights = target.get_weights()
    learned_weights = learned.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = learned_weights[i] * tau + target_weights[i] * (1 - tau)
    target.set_weights(target_weights)


class Buffer:
    def __init__(self, buffer_capacity, batch_size):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((buffer_capacity, num_states))
        self.action_buffer = np.zeros((buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((buffer_capacity, 1))
        self.next_state_buffer = np.zeros((buffer_capacity, num_states))
        self.done_buffer = np.zeros((buffer_capacity, 1), dtype=np.bool_)

    def record(self, state, action, reward, next_state, done):
        # Set index to zero if buffer_capacity is exceeded replacing old records
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.done_buffer[index] = done
        self.buffer_counter += 1

    def sample(self):
        # Get valid range for sampling
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        states = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_states = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)
        dones = tf.convert_to_tensor(self.done_buffer[batch_indices], dtype=tf.float32)
        return states, actions, rewards, next_states, dones


class DDPGAgent:
    def __init__(self):
        self.buffer = Buffer(10000, 32)
        self.min_buffer_size = 500
        self.gamma = 0.99
        self.tau = 0.005
        self.actor_model = get_actor_model()
        self.critic_model = get_critic_model()
        self.target_actor = get_actor_model()
        self.target_critic = get_critic_model()
        # Copying the weights initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=0.002)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.ou_noise = OUActionNoise()

    def policy(self, states):
        sampled_actions = tf.squeeze(self.actor_model(states))
        # Adding noise to action for exploration and ensuring it is within bounds
        actions = np.clip(sampled_actions.numpy() + self.ou_noise(), lower_bound, upper_bound)
        return [np.squeeze(actions)]

    # Heart of the algorithm - updating actor and critic networks : check pseudocode
    @tf.function
    def learn(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.sample()

        # Critic update
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            target_critic_values = self.target_critic([next_state_batch, target_actions], training=True)
            y = reward_batch + self.gamma * (1 - done_batch) * target_critic_values
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used minus for gradient ascent
            actor_loss = -tf.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

    def train(self, num_episodes, max_steps=200):
        print("Training agent:")
        ep_reward_list = []
        avg_reward_list = []
        for ep in range(num_episodes):
            state, _ = env.reset()
            self.ou_noise.reset()
            episodic_reward = 0

            for _ in range(max_steps):
                tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                action = self.policy(tf_state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.buffer.record(state, action, reward, next_state, done)
                state = next_state
                episodic_reward += reward

                if self.buffer.buffer_counter > self.min_buffer_size:
                    self.learn()
                    update_target(self.target_actor, self.actor_model, self.tau)
                    update_target(self.target_critic, self.critic_model, self.tau)

                if done:
                    break

            ep_reward_list.append(episodic_reward)
            avg_reward = np.mean(ep_reward_list[-10:])
            avg_reward_list.append(avg_reward)
            if (ep + 1) % 10 == 0:
                print(f"Episode {ep+1} : Avg Reward = {avg_reward}")
        return avg_reward_list

    def test(self, num_episodes, max_steps=200):
        print("\nTesting agent:")
        for ep in range(num_episodes):
            state, _ = env.reset()
            episodic_reward = 0

            for _ in range(max_steps):
                tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                action = tf.squeeze(self.actor_model(tf_state))
                action = [np.clip(action.numpy(), lower_bound, upper_bound)]
                next_state, reward, terminated, truncated, _ = env.step(action)
                state = next_state
                episodic_reward += reward
                if terminated or truncated:
                    break

            print(f"Episode {ep+1} : Reward = {episodic_reward}")


agent = DDPGAgent()
avg_reward_list = agent.train(num_episodes=100)
agent.test(num_episodes=10)
env.close()
plt.plot(avg_reward_list)
plt.title("Training Progress")
plt.xlabel("Episode")
plt.ylabel("Avg. Reward")
plt.show()
