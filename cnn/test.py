import numpy as np
import argparse
import os
import time
from env import Env as MultiUAVEnv
from maddpg_uav import MADDPG
from input import input_image
from logger import Logger
from plot_logs import generate_plots
from datetime import datetime

def test(load_dir, use_image_init=False, image_path=None):
    if use_image_init:
        if not image_path:
            raise ValueError("Image path is required when using image initialization.")
        input_image(image_path)

    env = MultiUAVEnv(image_init=use_image_init, log_dir="./test_images")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"🚀 Training started at {timestamp}")

    log_data_file_name = f"log_data_{timestamp}.json"
    logger = Logger(
        log_dir="./test_logs",
        log_file_name=f"logs_{timestamp}.txt",
        log_data_file_name=log_data_file_name,
    )
    num_agents = env.num_uavs
    obs_dim = (env.height, env.width, env.channels)
    action_dim = 2

    maddpg = MADDPG(num_agents=num_agents, obs_shape=obs_dim, action_dim=action_dim)
    if not load_dir or not os.path.exists(load_dir):
        raise ValueError(f"Model path does not exist: {load_dir}")
    maddpg.load(load_dir)
    print(f"📂 Loading model from: {load_dir}")

    NUM_EPISODES = 50
    MAX_STEPS = 300
    LOG_FREQ = 1

    # Initialize for analysis/plotting
    score_log_per_episode = {"coverage": [], "fairness": [], "energy_efficiency": [], "penalty_per_uav": []}
    episode_rewards = []

    print("\n🚀 MADDPG UAV Testing Started...\n")
    start_time = time.time()

    for episode in range(1, NUM_EPISODES + 1):
        obs = env.reset()  # shape: (num_agents, obs_dim)
        maddpg.reset_noise()

        if episode == 1:
            env.save_state_image()

        episode_reward = 0
        score_log = {"coverage": 0, "fairness": 0, "energy_efficiency": 0, "penalty_per_uav": 0}

        for _ in range(MAX_STEPS):
            actions = maddpg.select_action(obs, noise=False)  # shape: (num_agents, action_dim)
            next_obs, done, rewards, (cov, fair, energy_eff, penalty) = env.step(actions)
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

        # Logging
        if episode % LOG_FREQ == 0:
            elapsed_time = time.time() - start_time
            env.save_state_image(f"test_tate_epi_{episode}")
            logger.log_episode_metrics(episode, episode_rewards, score_log_per_episode, LOG_FREQ, elapsed_time)

    print("✅ Testing Completed!\n")

    # Call the plotting function at the end of testing
    print("📊 Generating plots...")
    generate_plots(log_file=f'./test_logs/{log_data_file_name}', output_dir='./plots/', output_file='testing_plots.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MADDPG UAV")
    parser.add_argument("--use_img", action="store_true", help="Use image initialization")
    parser.add_argument("--img_path", type=str, help="Path to the initial state image (required if --use_img is specified)")
    parser.add_argument("--model_path", type=str, default="saved_models/maddpg_episode_final", help="Path to the saved model directory (default: saved_models/maddpg_episode_final)")
    args = parser.parse_args()

    if args.use_img and not args.img_path:
        parser.error("--img_path is required when using --use_img")

    test(load_dir=args.model_path, use_image_init=args.use_img, image_path=args.img_path)
