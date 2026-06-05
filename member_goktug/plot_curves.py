"""
Plot learning curves for BrainBlock DQN experiments (Member B2).

Reads episode_data.npz files from results directories and generates
publication-quality learning curve plots.

Usage:
  python -m member_goktug.plot_curves --results-dir results --pattern "dqn_*"
  python -m member_goktug.plot_curves --results-dir results/dqn_shaped_mlp_seed42
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Rolling mean smoothing."""
    if len(data) < window:
        return np.cumsum(data) / np.arange(1, len(data) + 1)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_single_run(data_path: Path, output_dir: Path):
    """Plot learning curves for a single run."""
    data = np.load(data_path)
    rewards = data["rewards"]
    lengths = data["lengths"]
    successes = data["successes"]
    coverages = data["coverages"]
    invalids = data.get("invalids", np.zeros_like(rewards))

    episodes = np.arange(1, len(rewards) + 1)
    window = min(100, len(rewards) // 5) if len(rewards) > 5 else 1

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"DQN Learning Curves — {data_path.parent.name}", fontsize=14)

    # Reward
    ax = axes[0, 0]
    ax.plot(episodes[:len(smooth(rewards, window))], smooth(rewards, window),
            color="#2196F3", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Total Reward vs. Episode")
    ax.grid(True, alpha=0.3)

    # Coverage
    ax = axes[0, 1]
    ax.plot(episodes[:len(smooth(coverages, window))], smooth(coverages, window),
            color="#4CAF50", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coverage")
    ax.set_title("Board Coverage vs. Episode")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Success Rate
    ax = axes[0, 2]
    ax.plot(episodes[:len(smooth(successes, window))], smooth(successes, window),
            color="#FF9800", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate vs. Episode")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Episode Length
    ax = axes[1, 0]
    ax.plot(episodes[:len(smooth(lengths, window))], smooth(lengths, window),
            color="#9C27B0", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Length vs. Episode")
    ax.grid(True, alpha=0.3)

    # Invalid Rate
    ax = axes[1, 1]
    ax.plot(episodes[:len(smooth(invalids, window))], smooth(invalids, window),
            color="#F44336", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Invalid Action Rate")
    ax.set_title("Invalid Action Rate vs. Episode")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Cumulative successes
    ax = axes[1, 2]
    cum_success = np.cumsum(successes)
    ax.plot(episodes, cum_success, color="#009688", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Successes")
    ax.set_title("Cumulative Successes vs. Episode")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "learning_curves.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_comparison(results_dirs: list[Path], output_dir: Path):
    """Plot multi-run comparison with mean ± std shading."""
    if len(results_dirs) < 2:
        return

    # Group by config name (without seed)
    groups = {}
    for d in results_dirs:
        data_path = d / "episode_data.npz"
        if not data_path.exists():
            continue
        # Parse name: dqn_{reward}_{encoder}_seed{seed}
        name = d.name
        parts = name.rsplit("_seed", 1)
        config_name = parts[0] if len(parts) == 2 else name
        if config_name not in groups:
            groups[config_name] = []
        groups[config_name].append(data_path)

    if not groups:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("DQN Experiment Comparison (Member B2)", fontsize=14)

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

    for idx, (config_name, paths) in enumerate(groups.items()):
        color = colors[idx % len(colors)]

        # Load all runs
        all_rewards = []
        all_coverages = []
        all_successes = []

        for p in paths:
            data = np.load(p)
            all_rewards.append(data["rewards"])
            all_coverages.append(data["coverages"])
            all_successes.append(data["successes"])

        # Truncate to shortest run length
        min_len = min(len(r) for r in all_rewards)
        window = min(100, min_len // 5) if min_len > 5 else 1

        smoothed_rewards = np.array([smooth(r[:min_len], window) for r in all_rewards])
        smoothed_coverages = np.array([smooth(c[:min_len], window) for c in all_coverages])
        smoothed_successes = np.array([smooth(s[:min_len], window) for s in all_successes])

        episodes = np.arange(1, smoothed_rewards.shape[1] + 1)

        # Reward
        mean_r = smoothed_rewards.mean(axis=0)
        std_r = smoothed_rewards.std(axis=0)
        axes[0].plot(episodes, mean_r, color=color, label=config_name, linewidth=1.5)
        axes[0].fill_between(episodes, mean_r - std_r, mean_r + std_r,
                             color=color, alpha=0.2)

        # Coverage
        mean_c = smoothed_coverages.mean(axis=0)
        std_c = smoothed_coverages.std(axis=0)
        axes[1].plot(episodes, mean_c, color=color, label=config_name, linewidth=1.5)
        axes[1].fill_between(episodes, mean_c - std_c, mean_c + std_c,
                             color=color, alpha=0.2)

        # Success Rate
        mean_s = smoothed_successes.mean(axis=0)
        std_s = smoothed_successes.std(axis=0)
        axes[2].plot(episodes, mean_s, color=color, label=config_name, linewidth=1.5)
        axes[2].fill_between(episodes, mean_s - std_s, mean_s + std_s,
                             color=color, alpha=0.2)

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Mean Reward ± Std")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Coverage")
    axes[1].set_title("Mean Coverage ± Std")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Success Rate")
    axes[2].set_title("Mean Success Rate ± Std")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "comparison_curves.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved: {save_path}")


def parse_plot_args():
    parser = argparse.ArgumentParser(description="Plot DQN learning curves (B2)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Root results directory or single run directory")
    parser.add_argument("--pattern", type=str, default="dqn_*",
                        help="Glob pattern to match result directories")
    return parser.parse_args()


def main():
    args = parse_plot_args()
    root = Path(args.results_dir)

    # Check if this is a single run or a root directory
    if (root / "episode_data.npz").exists():
        # Single run
        print(f"Plotting single run: {root}")
        plot_single_run(root / "episode_data.npz", root)
        return

    # Multi-run: find all matching directories
    results_dirs = sorted([
        Path(d) for d in glob.glob(str(root / args.pattern))
        if Path(d).is_dir() and (Path(d) / "episode_data.npz").exists()
    ])

    if not results_dirs:
        print(f"No results found matching {root / args.pattern}")
        return

    print(f"Found {len(results_dirs)} result directories:")
    for d in results_dirs:
        print(f"  {d.name}")

    # Plot individual runs
    for d in results_dirs:
        print(f"\nPlotting: {d.name}")
        plot_single_run(d / "episode_data.npz", d)

    # Plot comparison
    print("\nGenerating comparison plot...")
    plot_comparison(results_dirs, root)


if __name__ == "__main__":
    main()
