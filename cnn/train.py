# TODO: plot graphs of the results stored
# TODO: save model periodically
import numpy as np
import time
from env import Env as MultiUAVEnv
import input
from maddpg_uav import MADDPG


def train(use_image_init=False, image_path=None):

    if use_image_init and image_path:
        input.input_image(image_path)
    env = MultiUAVEnv(image_init=use_image_init, log_dir="./state_images")
    num_agents = env.num_uavs
    obs_dim = (env.width, env.height, env.channels)
    action_dim = 2

    maddpg = MADDPG(num_agents=num_agents, obs_shape=obs_dim, action_dim=action_dim, device="cpu")

    num_episodes = 100  # 1000
    max_steps = 100  # 1000
    batch_size = 64
    log_freq = 1  # 100

    # Initialize for analysis/plotting
    score_log_per_episode = {"coverage": [], "fairness": [], "energy_efficiency": [], "penalty_per_uav": []}
    episode_rewards = []

    print("\n🚀 MADDPG UAV Training Started...\n")
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()  # shape: (num_agents, obs_dim)
        if episode == 1:
            env.save_image()
        maddpg.reset_noise()

        episode_reward = 0
        score_log = {"coverage": [], "fairness": [], "energy_efficiency": [], "penalty_per_uav": []}

        for _ in range(max_steps):
            # Each agent selects action based on local observation
            actions = maddpg.select_action(obs)  # shape: (num_agents, action_dim)
            # Step the environment
            next_obs, done, rewards, (cov, fair, energy_eff, penalty) = env.step(actions)

            # Store log scores per step
            score_log["coverage"].append(cov)
            score_log["fairness"].append(fair)
            score_log["energy_efficiency"].append(energy_eff)
            score_log["penalty_per_uav"].append(penalty)

            dones = np.full((num_agents,), done)

            # Store experience in buffer
            maddpg.store(obs, actions, rewards, next_obs, dones)

            # Update MADDPG agents
            maddpg.update(batch_size)

            obs = next_obs
            episode_reward += np.sum(rewards)
            if done:
                break

        # Store rewards and scores per episode
        episode_rewards.append(episode_reward)
        score_log_per_episode["coverage"].append(np.mean(score_log["coverage"]))
        score_log_per_episode["fairness"].append(np.mean(score_log["fairness"]))
        score_log_per_episode["energy_efficiency"].append(np.mean(score_log["energy_efficiency"]))
        score_log_per_episode["penalty_per_uav"].append(np.mean(np.stack(score_log["penalty_per_uav"], axis=0), axis=0))

        # Logging
        if episode % log_freq == 0:
            elapsed_time = time.time() - start_time
            env.save_image(f"state_epi_{episode}")
            penalty_avg = np.mean(np.stack(score_log_per_episode["penalty_per_uav"][-log_freq:], axis=0), axis=0)
            penalty_avg = np.round(penalty_avg, decimals=3)
            print(f"🔄 Episode {episode} | " f"Total Reward: {np.mean(episode_rewards[-log_freq:]):.3f} | " f"Coverage Avg: {np.mean(score_log_per_episode['coverage'][-log_freq:]):.3f} | " f"Fairness Avg: {np.mean(score_log_per_episode['fairness'][-log_freq:]):.3f} | " f"Energy Efficiency Avg: {np.mean(score_log_per_episode['energy_efficiency'][-log_freq:]):.3f} | " f"Penalty Avg: {penalty_avg} | " f"Elapsed Time: {elapsed_time:.2f}s")

    print("\n✅ Training Completed!\n")


if __name__ == "__main__":
    train(use_image_init=True, image_path="initial_state.png")
