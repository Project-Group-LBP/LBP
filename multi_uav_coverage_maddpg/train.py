import numpy as np
import argparse
import os
import time
from env import Env as MultiUAVEnv
from maddpg.maddpg_uav import MADDPG
from utils.input import input_image
from utils.logger import Logger
from utils.plot_logs import generate_plots
from datetime import datetime


def save_models(maddpg, episode, save_dir="saved_models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"maddpg_episode_{episode}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    maddpg.save(save_path)
    print(f"📁 Models saved for episode {episode}\n")


def train(num_episodes, use_image_init=False, image_path=None, resume_model=None):
    if use_image_init:
        if not image_path:
            raise ValueError("Image path is required when using image initialization.")
        input_image(image_path)

    env = MultiUAVEnv(image_init=use_image_init, log_dir="./train_images")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n🚀 Training started at {timestamp} for {num_episodes} episodes\n")

    logger = Logger(
        log_dir="./train_logs",
        log_file_name=f"logs_{timestamp}.txt",
        log_data_file_name=f"log_data_{timestamp}.json",
    )
    num_agents = env.num_uavs
    obs_dim = (env.height, env.width, env.channels)
    action_dim = 2

    maddpg = MADDPG(num_agents=num_agents, obs_shape=obs_dim, action_dim=action_dim)
    if resume_model:
        maddpg.load(resume_model)
        print(f"📂 Resumed training from: {resume_model}\n")

    MAX_STEPS = 300
    BATCH_SIZE = 32
    LOG_FREQ = 1  # 10
    IMG_FREQ = 10  # save image every 10 episodes
    LEARN_FREQ = 5  # learn every 5 steps
    SAVE_FREQ = 25  # save models every 25 episodes

    # Initialize for analysis/plotting
    score_log_per_episode = {"coverage": [], "fairness": [], "energy_efficiency": [], "penalty_per_uav": []}
    episode_rewards = []

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()  # shape: (num_agents, obs_dim)
        maddpg.reset_noise()

        if episode == 1:
            env.save_state_image()

        episode_reward = 0
        score_log = {"coverage": 0, "fairness": 0, "energy_efficiency": 0, "penalty_per_uav": []}

        for i in range(MAX_STEPS):
            actions = maddpg.select_action(obs, noise=True)  # shape: (num_agents, action_dim)
            next_obs, done, rewards, (cov, fair, energy_eff, penalty) = env.step(actions)

            dones = np.full((num_agents,), done)

            # Store experience in buffer
            maddpg.store(obs, actions, rewards, next_obs, dones)

            # Update MADDPG agents
            if (i + 1) % LEARN_FREQ == 0:
                maddpg.update(BATCH_SIZE)

            obs = next_obs

            # Store log scores per step
            score_log["coverage"] = cov
            score_log["fairness"] = fair
            score_log["energy_efficiency"] = energy_eff
            score_log["penalty_per_uav"] = penalty
            episode_reward += np.sum(rewards)

            if done:
                break

        # Store rewards and scores per episode
        episode_rewards.append(episode_reward)
        score_log_per_episode["coverage"].append(score_log["coverage"])
        score_log_per_episode["fairness"].append(score_log["fairness"])
        score_log_per_episode["energy_efficiency"].append(score_log["energy_efficiency"])
        score_log_per_episode["penalty_per_uav"].append(score_log["penalty_per_uav"])

        # Log episode metrics for plotting
        if episode % LOG_FREQ == 0:
            elapsed_time = time.time() - start_time
            logger.log_episode_metrics(episode, episode_rewards, score_log_per_episode, LOG_FREQ, elapsed_time)

        # Save images for analysis
        if episode % IMG_FREQ == 0:
            env.save_state_image(f"state_epi_{episode}")
            env.save_heat_map_image(f"heat_map_epi_{episode}")

        # Save models periodically
        if episode % SAVE_FREQ == 0:
            save_models(maddpg, episode)

    # Save final models
    save_models(maddpg, "final")
    print("✅ Training Completed!\n")

    # Call the plotting function at the end of training
    print("📊 Generating plots...\n")
    generate_plots(log_file=f"./train_logs/log_data_{timestamp}.json", output_dir="./plots/", output_file="training_plots.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MADDPG UAV")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of episodes to train (default : 500)")
    parser.add_argument("--use_img", action="store_true", help="Use image initialization")
    parser.add_argument("--img_path", type=str, help="Path to the initial state image (required if --use_img is specified)")
    parser.add_argument("--resume", type=str, help="Path to saved model directory to resume training from")
    args = parser.parse_args()

    if args.use_img and not args.img_path:
        parser.error("--img_path is required when using --use_img")

    train(num_episodes=args.num_episodes, use_image_init=args.use_img, image_path=args.img_path, resume_model=args.resume)
