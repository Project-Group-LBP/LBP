import torch
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
    maddpg = MADDPG(num_agents=num_agents, obs_dim=obs_dim, action_dim=action_dim, device='cpu')

    num_episodes = 1000
    max_steps = env.max_steps
    batch_size = 64

    # Initialize for analysis/plotting
    score_log = {'coverage': [], 'fairness': [], 'comm_penalty': [], 'obstacle_penalty': [], 'movement': []}
    episode_rewards = []

    print("\n🚀 MADDPG UAV Training Started...\n")
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()  # shape: (num_agents, obs_dim)
        maddpg.reset_noise()
        episode_reward = 0

        for step in range(max_steps):
            # Each agent selects action based on local observation
            actions = maddpg.select_action(obs)  # shape: (num_agents, action_dim)

            # Step the environment
            next_obs, done = env.step(actions)

            # ✅ Compute reward AFTER environment transition
            reward, (cov, fair, comm, obs_pen, move_eff) = env._calculate_reward(prev_positions=obs)

            # ✅ Log scores for plotting/analysis later
            score_log['coverage'].append(cov)
            score_log['fairness'].append(fair)
            score_log['comm_penalty'].append(comm)
            score_log['obstacle_penalty'].append(obs_pen)
            score_log['movement'].append(move_eff)

            # ✅ Prepare reward and done per agent (even if shared reward)
            rewards = np.full((num_agents,), reward)  # Team reward, structure future-proof
            dones = np.full((num_agents,), done)

            # Store experience in buffer
            maddpg.store(obs, actions, rewards, next_obs, dones)

            # Train MADDPG - Perform centralized critic + decentralized actor update
            maddpg.update(batch_size)

            obs = next_obs
            episode_reward += reward
            if done:
                break
        
        episode_rewards.append(episode_reward)

        # Logging
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"🔄 Episode {episode} | "
                  f"Total Reward: {episode_reward:.3f} | "
                  f"Coverage Avg: {np.mean(score_log['coverage'][-max_steps:]):.3f} | "
                  f"Fairness Avg: {np.mean(score_log['fairness'][-max_steps:]):.3f} | "
                  f"Comm Penalty Avg: {np.mean(score_log['comm_penalty'][-max_steps:]):.3f} | "
                  f"Obstacle Penalty Avg: {np.mean(score_log['obstacle_penalty'][-max_steps:]):.3f} | "
                  f"Movement Eff: {np.mean(score_log['movement'][-max_steps:]):.3f} | "
                  f"Elapsed Time: {elapsed_time:.2f}s")

    print("\n✅ Training Completed!\n")

if __name__ == "__main__":
    train()
