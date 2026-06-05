"""
Training script for BrainBlock PPO.

Usage:
  python -m member_umut.train --reward shaped --encoder mlp --seed 42 --episodes 500000

Logs metrics to CSV and optionally to console.
Checkpoints are saved periodically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from member_umut.agent import PPOAgent, PPOConfig
from member_umut.environment import BrainBlockEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on BrainBlock")
    parser.add_argument("--reward", type=str, default="shaped",
                        choices=["sparse", "shaped"], help="Reward mode")
    parser.add_argument("--encoder", type=str, default="mlp",
                        choices=["mlp", "cnn_mlp"], help="Encoder architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000,
                        help="Total training timesteps")
    parser.add_argument("--rollout-steps", type=int, default=2048,
                        help="Steps per rollout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy bonus coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Max gradient norm")
    parser.add_argument("--mini-batch-size", type=int, default=64,
                        help="Mini-batch size")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="PPO update epochs per rollout")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden layer dimension")
    parser.add_argument("--no-masking", action="store_true",
                        help="Disable action masking (for failure-mode analysis)")
    parser.add_argument("--diversity-bonus", type=float, default=0.0,
                        help="Extra reward added when agent finds a tiling it has never seen before (0=off)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print stats every N updates")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save checkpoint every N updates")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (auto-generated if not set)")
    return parser.parse_args()


def make_output_dir(args) -> Path:
    """Create output directory with descriptive name."""
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        mask_tag = "nomask" if args.no_masking else "mask"
        div_tag = f"_div{args.diversity_bonus:.2f}".rstrip("0").rstrip(".") if args.diversity_bonus > 0 else ""
        name = f"{args.encoder}_{args.reward}_{mask_tag}_seed{args.seed}{div_tag}"
        out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Config
    config = PPOConfig(
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        rollout_steps=args.rollout_steps,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        encoder_type=args.encoder,
        hidden_dim=args.hidden_dim,
    )

    out_dir = make_output_dir(args)
    print(f"Output dir: {out_dir}")

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Environment
    env = BrainBlockEnv(reward_mode=args.reward)

    # Agent
    agent = PPOAgent(config, device)
    param_count = sum(p.numel() for p in agent.network.parameters())
    print(f"Network parameters: {param_count:,}")

    # Logging
    csv_path = out_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "update", "timestep", "episode", "mean_reward", "mean_length",
        "success_rate", "mean_coverage", "invalid_rate",
        "policy_loss", "value_loss", "entropy", "approx_kl",
        "unique_tilings", "wall_time",
    ])

    # Training state
    global_step = 0
    episode_count = 0
    update_count = 0
    start_time = time.time()

    # Diversity tracking: set of frozenset tilings seen during training
    seen_tilings: set = set()
    discovered_tilings: list = []  # saved to disk at end of training

    best_success_rate = 0.0

    # Episode trackers (for logging window)
    ep_rewards = []
    ep_lengths = []
    ep_successes = []
    ep_coverages = []
    ep_invalids = []  # track invalid action episodes

    # Reset env
    obs, info = env.reset(seed=args.seed)
    action_mask = info["action_mask"]
    ep_reward = 0.0
    ep_length = 0

    num_updates = args.total_timesteps // args.rollout_steps
    print(f"Total updates: {num_updates}, rollout steps: {args.rollout_steps}")
    print(f"Training for {args.total_timesteps:,} timesteps...")
    print("=" * 70)

    for update in range(1, num_updates + 1):
        # Collect rollout
        for step in range(args.rollout_steps):
            # Optionally disable masking
            if args.no_masking:
                mask_for_agent = np.ones(320, dtype=np.int8)
            else:
                mask_for_agent = action_mask

            action, log_prob, value = agent.select_action(obs, mask_for_agent)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            # Track unique tilings; optionally augment reward for novel ones
            if terminated and next_info.get("termination_reason") == "success":
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
                    if args.diversity_bonus > 0:
                        reward += args.diversity_bonus

            agent.buffer.add(
                obs["grid"], obs["vec"], mask_for_agent.astype(np.float32),
                action, log_prob, reward, done, value,
            )

            ep_reward += reward
            ep_length += 1
            global_step += 1

            if done:
                reason = next_info.get("termination_reason", "unknown")
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
                action_mask = next_info["action_mask"]

        # Bootstrap value for GAE
        if args.no_masking:
            bootstrap_mask = np.ones(320, dtype=np.int8)
        else:
            bootstrap_mask = action_mask
        last_value = agent.get_value(obs, bootstrap_mask) if not done else 0.0

        # PPO update
        update_stats = agent.update(last_value)
        update_count += 1

        # Logging
        if len(ep_rewards) > 0 and update % args.log_interval == 0:
            window = min(100, len(ep_rewards))
            recent_rewards = ep_rewards[-window:]
            recent_lengths = ep_lengths[-window:]
            recent_successes = ep_successes[-window:]
            recent_coverages = ep_coverages[-window:]
            recent_invalids = ep_invalids[-window:]

            mean_reward = np.mean(recent_rewards)
            mean_length = np.mean(recent_lengths)
            success_rate = np.mean(recent_successes)
            mean_coverage = np.mean(recent_coverages)
            invalid_rate = np.mean(recent_invalids)
            wall_time = time.time() - start_time

            csv_writer.writerow([
                update, global_step, episode_count,
                f"{mean_reward:.4f}", f"{mean_length:.2f}",
                f"{success_rate:.4f}", f"{mean_coverage:.4f}",
                f"{invalid_rate:.4f}",
                f"{update_stats['policy_loss']:.6f}",
                f"{update_stats['value_loss']:.6f}",
                f"{update_stats['entropy']:.4f}",
                f"{update_stats['approx_kl']:.6f}",
                len(seen_tilings),
                f"{wall_time:.1f}",
            ])
            csv_file.flush()

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                agent.save(str(out_dir / "best_model.pt"))

            print(
                f"Update {update:5d} | Step {global_step:>8,} | Ep {episode_count:>6,} | "
                f"R={mean_reward:+.3f} | Len={mean_length:.1f} | "
                f"Succ={success_rate:.3f} [best={best_success_rate:.3f}] | "
                f"Cov={mean_coverage:.3f} | Inv={invalid_rate:.3f} | "
                f"PL={update_stats['policy_loss']:.4f} | "
                f"VL={update_stats['value_loss']:.4f} | "
                f"Ent={update_stats['entropy']:.3f} | "
                f"UniqueT={len(seen_tilings):3d} | "
                f"t={wall_time:.0f}s"
            )

        # Checkpointing
        if update % args.save_interval == 0:
            ckpt_path = out_dir / f"checkpoint_{update}.pt"
            agent.save(str(ckpt_path))

    # Final save
    agent.save(str(out_dir / "final_model.pt"))
    csv_file.close()

    # Save discovered tilings to JSON
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
