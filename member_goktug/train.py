"""
Training script for BrainBlock DQN (Member B2).

Timestep-based training loop aligned with Member A's PPO pipeline.
Supports dual reward modes, selectable encoder, unique tiling tracking,
best-model saving, and CSV metrics compatible with Member A.

Usage:
  python -m member_goktug.train --reward shaped --encoder mlp --seed 42 --total-timesteps 2000000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from member_goktug.agent import DQNAgent
from member_goktug.environment import BrainBlockEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on BrainBlock (Member B2)")
    # ── Environment ───────────────────────────────────────────────────
    parser.add_argument("--reward", type=str, default="shaped",
                        choices=["sparse", "shaped"], help="Reward mode")
    parser.add_argument("--encoder", type=str, default="mlp",
                        choices=["mlp", "cnn_mlp"], help="Encoder architecture")
    # ── Shared §4 hyperparameters ─────────────────────────────────────
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dim")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Max gradient norm")
    # ── DQN-specific ──────────────────────────────────────────────────
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft target-network update coefficient")
    parser.add_argument("--buffer-size", type=int, default=500_000,
                        help="Replay buffer capacity")
    parser.add_argument("--learning-starts", type=int, default=10_000,
                        help="Timesteps of random play before learning starts")
    parser.add_argument("--train-freq", type=int, default=4,
                        help="Train every N environment steps")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Starting epsilon for ε-greedy")
    parser.add_argument("--epsilon-end", type=float, default=0.05,
                        help="Final epsilon")
    parser.add_argument("--epsilon-decay-steps", type=int, default=500_000,
                        help="Timesteps over which epsilon decays linearly")
    # ── Run settings ──────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000,
                        help="Total training timesteps")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print stats every N log windows")
    parser.add_argument("--log-freq", type=int, default=2048,
                        help="Log every N timesteps (aligned with Member A rollout)")
    parser.add_argument("--save-interval", type=int, default=100_000,
                        help="Save checkpoint every N timesteps")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (auto-generated if not set)")
    return parser.parse_args()


def make_output_dir(args) -> Path:
    """Create output directory with descriptive name."""
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        name = f"dqn_{args.reward}_{args.encoder}_seed{args.seed}"
        out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_epsilon(step: int, args) -> float:
    """Linear epsilon decay from epsilon_start → epsilon_end over epsilon_decay_steps."""
    if step < args.learning_starts:
        return args.epsilon_start
    decay_step = step - args.learning_starts
    frac = min(1.0, decay_step / args.epsilon_decay_steps)
    return args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)


def train(args):
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = make_output_dir(args)
    print(f"Output dir: {out_dir}")

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Environment
    env = BrainBlockEnv(reward_mode=args.reward)

    # Agent
    agent = DQNAgent(
        encoder_type=args.encoder,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        max_grad_norm=args.max_grad_norm,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        device=str(device),
    )
    param_count = sum(p.numel() for p in agent.q_net.parameters())
    print(f"Q-Network parameters: {param_count:,}")

    # Logging — CSV columns aligned with Member A
    csv_path = out_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "update", "timestep", "episode", "mean_reward", "mean_length",
        "success_rate", "mean_coverage", "invalid_rate",
        "loss", "epsilon", "unique_tilings", "wall_time",
    ])

    # Training state
    global_step = 0
    episode_count = 0
    update_count = 0
    start_time = time.time()

    # Diversity tracking
    seen_tilings: set = set()
    discovered_tilings: list = []

    best_success_rate = 0.0

    # Episode trackers (for logging window)
    ep_rewards = []
    ep_lengths = []
    ep_successes = []
    ep_coverages = []
    ep_invalids = []
    recent_losses = []

    # Next log checkpoint
    next_log_step = args.log_freq

    # Reset env
    obs, info = env.reset(seed=args.seed)
    action_mask = info["action_mask"]
    ep_reward = 0.0
    ep_length = 0

    print(f"Training for {args.total_timesteps:,} timesteps...")
    print(f"Reward mode: {args.reward} | Encoder: {args.encoder}")
    print(f"Buffer size: {args.buffer_size:,} | Learning starts: {args.learning_starts:,}")
    print(f"Epsilon: {args.epsilon_start} → {args.epsilon_end} over {args.epsilon_decay_steps:,} steps")
    print("=" * 70)

    while global_step < args.total_timesteps:
        # Get epsilon
        eps = get_epsilon(global_step, args)

        # Select action
        action = agent.act(obs, action_mask, epsilon=eps)

        # Step environment
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        next_action_mask = next_info["action_mask"]

        # Store transition
        agent.memory.add(obs, action, reward, next_obs, done,
                         action_mask, next_action_mask)

        ep_reward += reward
        ep_length += 1
        global_step += 1

        # Train agent
        if global_step >= args.learning_starts and global_step % args.train_freq == 0:
            stats = agent.update()
            update_count += 1
            if stats["loss"] > 0:
                recent_losses.append(stats["loss"])

        if done:
            reason = next_info.get("termination_reason", "unknown")

            # Track unique tilings for successful episodes
            if reason == "success":
                tiling_key = frozenset(
                    (pt, frozenset(cells)) for pt, cells in env._placed
                )
                if tiling_key not in seen_tilings:
                    seen_tilings.add(tiling_key)
                    discovered_tilings.append({
                        "tiling_id": len(discovered_tilings) + 1,
                        "episode": episode_count,
                        "global_step": global_step,
                        "placed": [
                            {"piece": pt, "cells": sorted(list(cells))}
                            for pt, cells in env._placed
                        ],
                    })

            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            ep_successes.append(1.0 if reason == "success" else 0.0)
            ep_coverages.append(next_info.get("coverage", 0.0))
            ep_invalids.append(1.0 if reason == "illegal_action" else 0.0)
            episode_count += 1

            # Reset
            obs, info = env.reset()
            action_mask = info["action_mask"]
            ep_reward = 0.0
            ep_length = 0
        else:
            obs = next_obs
            action_mask = next_action_mask

        # Logging (timestep-aligned)
        if global_step >= next_log_step and len(ep_rewards) > 0:
            next_log_step += args.log_freq

            window = min(100, len(ep_rewards))
            mean_reward = np.mean(ep_rewards[-window:])
            mean_length = np.mean(ep_lengths[-window:])
            success_rate = np.mean(ep_successes[-window:])
            mean_coverage = np.mean(ep_coverages[-window:])
            invalid_rate = np.mean(ep_invalids[-window:])
            mean_loss = np.mean(recent_losses[-200:]) if recent_losses else 0.0
            wall_time = time.time() - start_time

            csv_writer.writerow([
                update_count, global_step, episode_count,
                f"{mean_reward:.4f}", f"{mean_length:.2f}",
                f"{success_rate:.4f}", f"{mean_coverage:.4f}",
                f"{invalid_rate:.4f}",
                f"{mean_loss:.6f}", f"{eps:.4f}",
                len(seen_tilings),
                f"{wall_time:.1f}",
            ])
            csv_file.flush()

            # Save best model
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                agent.save(str(out_dir / "best_model.pt"))

            # Console output
            if (global_step // args.log_freq) % args.log_interval == 0:
                print(
                    f"Step {global_step:>8,} | Ep {episode_count:>6,} | "
                    f"R={mean_reward:+.3f} | Len={mean_length:.1f} | "
                    f"Succ={success_rate:.3f} [best={best_success_rate:.3f}] | "
                    f"Cov={mean_coverage:.3f} | Inv={invalid_rate:.3f} | "
                    f"Loss={mean_loss:.4f} | ε={eps:.3f} | "
                    f"UniqueT={len(seen_tilings):3d} | "
                    f"t={wall_time:.0f}s"
                )

        # Checkpointing
        if global_step % args.save_interval == 0:
            ckpt_path = out_dir / f"checkpoint_{global_step}.pt"
            agent.save(str(ckpt_path))

    # Final save
    agent.save(str(out_dir / "final_model.pt"))
    csv_file.close()

    # Save discovered tilings
    with open(out_dir / "discovered_tilings.json", "w") as f:
        json.dump(discovered_tilings, f, indent=2)

    # Save episode-level data for plotting
    np.savez(
        out_dir / "episode_data.npz",
        rewards=np.array(ep_rewards),
        lengths=np.array(ep_lengths),
        successes=np.array(ep_successes),
        coverages=np.array(ep_coverages),
        invalids=np.array(ep_invalids),
    )

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Training complete: {episode_count:,} episodes, {global_step:,} steps in {elapsed:.1f}s")
    print(f"Final success rate: {np.mean(ep_successes[-100:]):.4f}")
    print(f"Unique tilings discovered: {len(seen_tilings)}")
    print(f"Model saved to: {out_dir / 'final_model.pt'}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
