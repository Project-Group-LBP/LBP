import os
import json
import matplotlib.pyplot as plt


def plot_subplot(ax, x, y, xlabel, ylabel, title):
    """Helper function to plot a single subplot"""
    ax.plot(x, y, label=ylabel, linestyle="-", marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def generate_plots(log_file="./train_logs/log_data.json", output_dir="./plots/", output_file="plots.png"):
    """Generate plots from the logs stored in 'log_file'"""

    # Load data from JSON file
    with open(log_file, "r") as file:
        log_data = json.load(file)

    # Extract parameters
    parameters = {"episode": [entry["episode"] for entry in log_data], "reward": [entry["reward"] for entry in log_data], "coverage": [entry["coverage"] for entry in log_data], "fairness": [entry["fairness"] for entry in log_data], "energy_efficiency": [entry["energy_efficiency"] for entry in log_data]}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle("Performance Metrics per Episode", fontsize=16)

    # Plot each metric
    plot_subplot(axs[0, 0], parameters["episode"], parameters["reward"], "Episode", "Reward", "Reward vs Episode")
    plot_subplot(axs[0, 1], parameters["episode"], parameters["coverage"], "Episode", "Coverage", "Coverage vs Episode")
    plot_subplot(axs[1, 0], parameters["episode"], parameters["fairness"], "Episode", "Fairness", "Fairness vs Episode")
    plot_subplot(axs[1, 1], parameters["episode"], parameters["energy_efficiency"], "Episode", "Energy Efficiency", "Energy Efficiency vs Episode")

    # Adjust layout and save the figure
    plt.tight_layout()
    output_file_name = os.path.join(output_dir, output_file)
    plt.savefig(output_file_name)
    plt.close()

    print(f"📈 Plots saved to {output_file_name}")
