"""
Plot learning curves from training metrics.

Usage:
  python -m member_umut.plot_curves --results-dir results/mlp_shaped_mask_seed42
  python -m member_umut.plot_curves --compare results/mlp_shaped_mask_seed42 results/mlp_sparse_mask_seed42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def smooth(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Running mean smoothing."""
    if len(data) < window:
        return np.cumsum(data) / np.arange(1, len(data) + 1)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_single_run(results_dir: str, save: bool = True):
    """Plot learning curves for a single training run."""
    path = Path(results_dir)
    data = np.load(path / "episode_data.npz")

    rewards = data["rewards"]
    lengths = data["lengths"]
    successes = data["successes"]
    coverages = data["coverages"]
    invalids = data["invalids"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Learning Curves — {path.name}", fontsize=14)
    window = 100

    # (1) Reward vs episode
    ax = axes[0, 0]
    ax.plot(smooth(rewards, window), linewidth=0.8, color="#4FC3F7")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic Return")
    ax.set_title("Total Reward vs Episode")
    ax.grid(True, alpha=0.3)

    # (2) Coverage vs episode
    ax = axes[0, 1]
    ax.plot(smooth(coverages, window), linewidth=0.8, color="#81C784")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coverage (fraction)")
    ax.set_title("Covered Area vs Episode")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # (3) Episode length vs episode
    ax = axes[1, 0]
    ax.plot(smooth(lengths, window), linewidth=0.8, color="#FFD54F")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Length over Time")
    ax.grid(True, alpha=0.3)

    # (4) Invalid-action rate vs episode
    ax = axes[1, 1]
    ax.plot(smooth(invalids, window), linewidth=0.8, color="#FF8A65")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Invalid Action Rate")
    ax.set_title("Invalid-Action Rate over Time")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        save_path = path / "learning_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_comparison(dirs: list[str], save_dir: str = "results/figures"):
    """Compare learning curves across multiple runs."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Learning Curve Comparison", fontsize=14)
    window = 200

    colors = ["#4FC3F7", "#FF8A65", "#81C784", "#CE93D8", "#FFD54F"]

    for i, d in enumerate(dirs):
        path = Path(d)
        data = np.load(path / "episode_data.npz")
        label = path.name
        c = colors[i % len(colors)]

        axes[0, 0].plot(smooth(data["rewards"], window), label=label,
                        linewidth=0.8, color=c)
        axes[0, 1].plot(smooth(data["coverages"], window), label=label,
                        linewidth=0.8, color=c)
        axes[1, 0].plot(smooth(data["lengths"], window), label=label,
                        linewidth=0.8, color=c)
        axes[1, 1].plot(smooth(data["invalids"], window), label=label,
                        linewidth=0.8, color=c)

    for ax, title, ylabel in zip(
        axes.flat,
        ["Reward vs Episode", "Coverage vs Episode",
         "Episode Length", "Invalid-Action Rate"],
        ["Return", "Coverage", "Length", "Invalid Rate"],
    ):
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_path / "comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_multi_seed(base_pattern: str, seeds: list[int], save_dir: str = "results/figures"):
    """
    Plot mean ± std across multiple seeds.

    base_pattern: e.g. "results/mlp_shaped_mask_seed{seed}"
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_rewards = []
    all_coverages = []
    all_lengths = []
    all_successes = []

    min_len = float("inf")
    for seed in seeds:
        path = Path(base_pattern.format(seed=seed))
        data = np.load(path / "episode_data.npz")
        min_len = min(min_len, len(data["rewards"]))

    for seed in seeds:
        path = Path(base_pattern.format(seed=seed))
        data = np.load(path / "episode_data.npz")
        all_rewards.append(smooth(data["rewards"][:min_len], 200))
        all_coverages.append(smooth(data["coverages"][:min_len], 200))
        all_lengths.append(smooth(data["lengths"][:min_len], 200))
        all_successes.append(smooth(data["successes"][:min_len], 200))

    min_smooth = min(len(a) for a in all_rewards)
    all_rewards = np.array([a[:min_smooth] for a in all_rewards])
    all_coverages = np.array([a[:min_smooth] for a in all_coverages])
    all_lengths = np.array([a[:min_smooth] for a in all_lengths])
    all_successes = np.array([a[:min_smooth] for a in all_successes])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    label = Path(base_pattern.format(seed="*")).name

    for ax, data_arr, title, ylabel, color in zip(
        axes.flat,
        [all_rewards, all_coverages, all_lengths, all_successes],
        ["Reward", "Coverage", "Episode Length", "Success Rate"],
        ["Return", "Coverage", "Length", "Rate"],
        ["#4FC3F7", "#81C784", "#FFD54F", "#CE93D8"],
    ):
        mean = data_arr.mean(axis=0)
        std = data_arr.std(axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, color=color, linewidth=1)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        ax.set_xlabel("Episode (smoothed)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} ({label})")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Multi-Seed Results (seeds: {seeds})", fontsize=14)
    plt.tight_layout()
    out = save_path / f"multi_seed_{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, help="Single run dir")
    parser.add_argument("--compare", nargs="+", help="Multiple dirs to compare")
    parser.add_argument("--multi-seed", type=str,
                        help="Pattern like 'results/mlp_shaped_mask_seed{seed}'")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()

    if args.results_dir:
        plot_single_run(args.results_dir)
    if args.compare:
        plot_comparison(args.compare)
    if args.multi_seed:
        plot_multi_seed(args.multi_seed, args.seeds)
