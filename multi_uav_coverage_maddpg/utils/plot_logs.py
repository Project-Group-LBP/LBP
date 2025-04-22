import os
import json
import matplotlib.pyplot as plt


def plot_metric(x, y, xlabel, ylabel, title, output_path):
    """Helper function to plot a single metric as a scatter plot and save it separately"""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 {title} plot saved to {output_path}")


def generate_plots(log_file, output_dir, output_file_prefix):
    """Generate plots from the logs stored in 'log_file'"""

    # Load data from JSON file
    with open(log_file, "r") as file:
        log_data = json.load(file)

    # Extract parameters
    parameters = {"episode": [entry["episode"] for entry in log_data], "reward": [entry["reward"] for entry in log_data], "coverage": [entry["coverage"] for entry in log_data], "fairness": [entry["fairness"] for entry in log_data], "energy_efficiency": [entry["energy_efficiency"] for entry in log_data]}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot each metric as a separate file
    metrics = ["reward", "coverage", "fairness", "energy_efficiency"]

    for metric in metrics:
        output_path = os.path.join(output_dir, f"{output_file_prefix}_{metric}.png")
        plot_metric(parameters["episode"], parameters[metric], "Episode", metric.replace("_", " ").title(), f"{metric.replace('_', ' ').title()} vs Episode", output_path)

    plot_metric(parameters["coverage"], parameters["fairness"], "Coverage", "Fairness", "Fairness vs Coverage", os.path.join(output_dir, f"{output_file_prefix}_fairness_vs_coverage.png"))
    plot_metric(parameters["energy_efficiency"], parameters["fairness"], "Energy Efficiency", "Fairness", "Fairness vs Energy Efficiency", os.path.join(output_dir, f"{output_file_prefix}_fairness_vs_energy_efficiency.png"))

    print(f"\n✅ All scatter plots saved to {output_dir}\n")
