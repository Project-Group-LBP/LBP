# TODO: plot graphs of the results stored
import numpy as np
import argparse
import os
import time
from env import Env as MultiUAVEnv
import input
from maddpg_uav import MADDPG

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # will this be needed?


def save_models(maddpg, episode, save_dir="saved_models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"maddpg_episode_{episode}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    maddpg.save(save_path)
    print(f"📁 Models saved at episode {episode}")


def train(use_image_init=False, image_path=None):

    if use_image_init and image_path:
        input.input_image(image_path)
    env = MultiUAVEnv(image_init=use_image_init, log_dir="./train_state_images")
    num_agents = env.num_uavs
    obs_dim = (env.width, env.height, env.channels)
    action_dim = 2

    maddpg = MADDPG(num_agents=num_agents, obs_shape=obs_dim, action_dim=action_dim, device="cpu")

    num_episodes = 100  # 500
    max_steps = 500  # 500
    batch_size = 32
    log_freq = 1 # 10
    learn_freq = 5  # learn every 5 steps
    save_freq = 10  # save models every 10 episodes

    # Initialize for analysis/plotting
    score_log_per_episode = {"coverage": [], "fairness": [], "energy_efficiency": [], "penalty_per_uav": []}
    episode_rewards = []

    print("\n🚀 MADDPG UAV Training Started...\n")
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()  # shape: (num_agents, obs_dim)
        maddpg.reset_noise()

        if episode == 1:
            env.save_state_image()

        episode_reward = 0
        score_log = {"coverage": [], "fairness": [], "energy_efficiency": [], "penalty_per_uav": []}

        for i in range(max_steps):
            actions = maddpg.select_action(obs, noise=True)  # shape: (num_agents, action_dim)
            next_obs, done, rewards, (cov, fair, energy_eff, penalty) = env.step(actions)

            dones = np.full((num_agents,), done)

            # Store experience in buffer
            maddpg.store(obs, actions, rewards, next_obs, dones)

            # Update MADDPG agents
            if (i + 1) % learn_freq == 0:
                maddpg.update(batch_size)

            obs = next_obs

            # Store log scores per step
            score_log["coverage"].append(cov)
            score_log["fairness"].append(fair)
            score_log["energy_efficiency"].append(energy_eff)
            score_log["penalty_per_uav"].append(penalty)
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
            env.save_state_image(f"state_epi_{episode}")
            env.save_heat_map_image(f"heat_map_epi_{episode}")
            penalty_avg = np.mean(np.stack(score_log_per_episode["penalty_per_uav"][-log_freq:], axis=0), axis=0)
            penalty_avg = np.round(penalty_avg, decimals=3)
            print(f"🔄 Episode {episode} | " f"Total Reward: {np.mean(episode_rewards[-log_freq:]):.3f} | " f"Coverage Avg: {np.mean(score_log_per_episode['coverage'][-log_freq:]):.3f} | " f"Fairness Avg: {np.mean(score_log_per_episode['fairness'][-log_freq:]):.3f} | " f"Energy Efficiency Avg: {np.mean(score_log_per_episode['energy_efficiency'][-log_freq:]):.3f} | " f"Penalty Avg: {penalty_avg} | " f"Elapsed Time: {elapsed_time:.2f}s")

        # Save models periodically
        if episode % save_freq == 0:
            save_models(maddpg, episode)

    # Save final models
    save_models(maddpg, "final")
    print("\n✅ Training Completed!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MADDPG UAV")
    parser.add_argument("--use_img", action="store_true", help="Use image initialization")
    parser.add_argument("--img_path", type=str, help="Path to the initial state image (required if --use_img is specified)")
    args = parser.parse_args()

    if args.use_img and not args.img_path:
        parser.error("--img_path is required when using --use_img")

    train(use_image_init=args.use_img, image_path=args.img_path)
