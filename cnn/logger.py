import json
import os
import numpy as np

class Logger:
    def __init__(self, log_dir=".", log_file_name="logs.txt", log_data_file_name="log_data.json"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Plaintext log file
        self.log_file_path = os.path.join(log_dir, log_file_name)
        # JSON file for plotting
        self.json_file_path = os.path.join(log_dir, log_data_file_name)

    def save(self, json_data):
        with open(self.json_file_path, "w") as f:
            json.dump(json_data, f, indent=4)

    def log_episode_metrics(self, episode, episode_rewards, score_log_per_episode, LOG_FREQ, elapsed_time, log_dir="logs"):
        '''Log the metrics of the episode to a file and saves them in ./logs/training_logs.txt'''
        
        reward_avg = np.mean(episode_rewards[-LOG_FREQ:])
        coverage_avg = np.mean(score_log_per_episode["coverage"][-LOG_FREQ:])
        fairness_avg = np.mean(score_log_per_episode["fairness"][-LOG_FREQ:])
        energy_avg = np.mean(score_log_per_episode["energy_efficiency"][-LOG_FREQ:])
        penalty_avg = np.mean(np.stack(score_log_per_episode["penalty_per_uav"][-LOG_FREQ:], axis=0), axis=0)
        penalty_avg = (np.round(penalty_avg, decimals=3)).tolist()

        log_msg = (
            f"🔄 Episode {episode} | "
            f"Total Reward: {reward_avg:.3f} | "
            f"Coverage Avg: {coverage_avg:.3f} | "
            f"Fairness Avg: {fairness_avg:.3f} | "
            f"Energy Efficiency Avg: {energy_avg:.3f} | "
            f"Penalty Avg: {penalty_avg} | "
            f"Elapsed Time: {elapsed_time:.2f}s\n"
        )

        # # Print to terminal (OPTIONAL)
        # print(log_msg.strip())

        # Save to file
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(log_msg)

        # Append structured data to json
        data_entry = {
            "episode": episode,
            "reward": reward_avg,
            "coverage": coverage_avg,
            "fairness": fairness_avg,
            "energy_efficiency": energy_avg,
            "penalty": penalty_avg,
            "time": elapsed_time
        }

        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, "r") as jf:
                try:
                    json_data = json.load(jf)
                except json.JSONDecodeError:
                    json_data = []
        else:
            json_data = []

        json_data.append(data_entry)

        self.save(json_data)
