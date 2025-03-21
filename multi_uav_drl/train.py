# TODO: plot graphs of the results stored

import numpy as np
from uav_env import MultiUAVEnv
from maddpg_uav import MADDPG
import time


def train():
    # Environment and MADDPG Initialization
    env = MultiUAVEnv()
    num_agents = env.num_uavs
    obs_dim = 2  # UAV observes its 2D position
    action_dim = 2  # (angle, distance) or 2D continuous control
    maddpg = MADDPG(num_agents=num_agents, obs_dim=obs_dim, action_dim=action_dim, device="cpu")

    num_episodes = 1000
    max_steps = env.max_steps
    batch_size = 64

    # Initialize for analysis/plotting
    score_log_per_episode = {"coverage": [], "fairness": [], "energy_efficiency": [], "comm_penalty_per_uav": []}
    episode_rewards = []

    print("\n🚀 MADDPG UAV Training Started...\n")
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()  # shape: (num_agents, obs_dim)
        maddpg.reset_noise()
        # To stored step details for each episode
        step = 0
        episode_reward = 0
        score_log = {"coverage": [], "fairness": [], "energy_efficiency": [], "comm_penalty_per_uav": []}

        for step in range(max_steps):
            # Each agent selects action based on local observation
            actions = maddpg.select_action(obs)  # shape: (num_agents, action_dim)

            # Step the environment
            next_obs, done, reward_per_uav, (cov, fair, energy_eff, comm_per_uav) = env.step(actions)

            # Store log scores per step
            score_log["coverage"].append(cov)
            score_log["fairness"].append(fair)
            score_log["energy_efficiency"].append(energy_eff)
            score_log["comm_penalty_per_uav"].append(comm_per_uav)

            # Prepare reward and done per agent (even if shared reward)
            rewards = np.copy(reward_per_uav)
            dones = np.full((num_agents,), done)

            # Store experience in buffer
            maddpg.store(obs, actions, rewards, next_obs, dones)

            # Train MADDPG - Perform centralized critic + decentralized actor update
            maddpg.update(batch_size)

            obs = next_obs
            episode_reward += np.sum(rewards)
            if done:
                break

        # Store rewards and scores per episode
        episode_rewards.append(episode_reward / step)
        score_log_per_episode["coverage"].append(np.mean(score_log["coverage"]))
        score_log_per_episode["fairness"].append(np.mean(score_log["fairness"]))
        score_log_per_episode["energy_efficiency"].append(np.mean(score_log["energy_efficiency"]))
        score_log_per_episode["comm_penalty_per_uav"].append(np.mean(np.stack(score_log["comm_penalty_per_uav"], axis=0), axis=0))

        # Logging
        log_freq = env.log_freq
        if episode % log_freq == 0:
            elapsed_time = time.time() - start_time
            comm_penalty_avg_per_uav = np.mean(np.stack(score_log_per_episode["comm_penalty_per_uav"][-log_freq:], axis=0), axis=0)
            comm_penalty_avg_per_uav = np.round(comm_penalty_avg_per_uav, decimals=3)
            print(f"🔄 Episode {episode} | " f"Total Reward: {np.mean(episode_rewards[-log_freq:]):.3f} | " f"Coverage Avg: {np.mean(score_log_per_episode['coverage'][-log_freq:]):.3f} | " f"Fairness Avg: {np.mean(score_log_per_episode['fairness'][-log_freq:]):.3f} | " f"Energy Efficiency Avg: {np.mean(score_log_per_episode['energy_efficiency'][-log_freq:]):.3f} | " f"Comm Penalty Avg: {comm_penalty_avg_per_uav} | " f"Elapsed Time: {elapsed_time:.2f}s")

    print("\n✅ Training Completed!\n")


if __name__ == "__main__":
    train()
