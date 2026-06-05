"""
Evaluation script for BrainBlock PPO.

Usage:
  python -m member_umut.evaluate --model results/mlp_shaped_mask_seed42/final_model.pt \
                               --encoder mlp --reward shaped --episodes 1000 --seed 42

Features:
  - Deterministic rollouts (argmax policy)
  - Collects success rate, mean return, mean episode length, invalid-action rate
  - Finds and visualizes distinct solutions
  - Generates qualitative step-by-step rollout trace
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from member_umut.agent import PPOAgent, PPOConfig
from member_umut.environment import BrainBlockEnv
from common.visualize import render_board, render_episode_replay


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO on BrainBlock")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--encoder", type=str, default="mlp",
                        choices=["mlp", "cnn_mlp"])
    parser.add_argument("--reward", type=str, default="shaped",
                        choices=["sparse", "shaped"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--render-solutions", type=int, default=5,
                        help="Number of distinct solutions to render")
    parser.add_argument("--render-trace", action="store_true",
                        help="Render step-by-step trace for one episode")
    parser.add_argument("--no-masking", action="store_true",
                        help="Disable action masking (for failure-mode evaluation)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample from policy distribution instead of argmax (more diverse solutions)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature for sampling (>1 = more uniform, implies --stochastic)")
    return parser.parse_args()


def board_to_tiling_key(placed: list[tuple[str, set]]) -> frozenset:
    """
    Convert placed pieces to a canonical tiling representation.
    A tiling is a frozenset of (piece_type, frozenset of (x, y) cells).
    Placement order does NOT matter.
    """
    return frozenset(
        (piece_type, frozenset(cells))
        for piece_type, cells in placed
    )


@torch.no_grad()
def evaluate(args):
    device = torch.device("cpu")

    # Build agent
    config = PPOConfig(
        encoder_type=args.encoder,
        hidden_dim=args.hidden_dim,
    )
    agent = PPOAgent(config, device)
    agent.load(args.model)
    agent.network.eval()

    # Environment
    env = BrainBlockEnv(reward_mode=args.reward)

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.model).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    all_rewards = []
    all_lengths = []
    all_successes = []
    all_coverages = []
    all_invalids = []

    # Distinct solutions
    unique_solutions: dict[frozenset, list[tuple[str, set]]] = {}
    trace_episode = None  # store one episode trace

    print(f"Evaluating {args.episodes} episodes...")
    print(f"Model: {args.model}")
    print(f"Encoder: {args.encoder}, Reward: {args.reward}")
    print("=" * 60)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        action_mask = info["action_mask"]

        ep_reward = 0.0
        ep_length = 0
        done = False

        # For trace
        board_snapshots = []
        piece_history = []

        while not done:
            grid = torch.tensor(obs["grid"], device=device).unsqueeze(0)
            vec = torch.tensor(obs["vec"], device=device).unsqueeze(0)
            eff_mask = np.ones(320, dtype=np.float32) if args.no_masking else action_mask.astype(np.float32)
            mask = torch.tensor(eff_mask, device=device).unsqueeze(0)

            dist, _ = agent.network(grid, vec, mask)
            if args.temperature != 1.0:
                from torch.distributions import Categorical
                action = Categorical(logits=dist.logits / args.temperature).sample().item()
            elif args.stochastic:
                action = dist.sample().item()
            else:
                action = dist.probs.argmax(dim=-1).item()

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_length += 1

            # Snapshot
            board_snapshots.append(env.board.copy())
            # Get current piece (before it was removed from queue — use placed history)
            if env._placed:
                piece_history.append(env._placed[-1][0])

            if done:
                reason = next_info.get("termination_reason", "unknown")
                all_rewards.append(ep_reward)
                all_lengths.append(ep_length)
                all_successes.append(1.0 if reason == "success" else 0.0)
                all_coverages.append(next_info.get("coverage", 0.0))
                all_invalids.append(1.0 if reason == "illegal_action" else 0.0)

                # Track distinct solutions
                if reason == "success":
                    key = board_to_tiling_key(env._placed)
                    if key not in unique_solutions:
                        unique_solutions[key] = [
                            (pt, set(cells)) for pt, cells in env._placed
                        ]
                        print(f"  Episode {ep}: Found solution #{len(unique_solutions)}")

                # Save first successful trace
                if reason == "success" and trace_episode is None:
                    trace_episode = (board_snapshots, piece_history)
            else:
                obs = next_obs
                action_mask = next_info["action_mask"]

    # ── Summary stats ──────────────────────────────────────────────
    print("=" * 60)
    success_rate = np.mean(all_successes)
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_length = np.mean(all_lengths)
    invalid_rate = np.mean(all_invalids)
    mean_coverage = np.mean(all_coverages)

    results = {
        "episodes": args.episodes,
        "success_rate": float(success_rate),
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "mean_length": float(mean_length),
        "invalid_rate": float(invalid_rate),
        "mean_coverage": float(mean_coverage),
        "unique_solutions_found": len(unique_solutions),
    }

    print(f"Success rate:     {success_rate:.4f}")
    print(f"Mean reward:      {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"Mean ep length:   {mean_length:.2f}")
    print(f"Invalid-act rate: {invalid_rate:.4f}")
    print(f"Mean coverage:    {mean_coverage:.4f}")
    print(f"Unique solutions: {len(unique_solutions)}")

    # Save results
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save unique solutions as JSON for later visualization
    solutions_json = [
        {"solution_id": i + 1,
         "placed": [{"piece": pt, "cells": sorted([list(c) for c in cells])}
                    for pt, cells in placed]}
        for i, placed in enumerate(unique_solutions.values())
    ]
    with open(out_dir / "eval_solutions.json", "w") as f:
        json.dump(solutions_json, f, indent=2)

    # ── Render distinct solutions ──────────────────────────────────
    n_render = min(args.render_solutions, len(unique_solutions))
    solution_list = list(unique_solutions.values())
    for i in range(n_render):
        placed = solution_list[i]
        save_path = str(out_dir / f"solution_{i+1}.png")
        render_board(placed, title=f"Solution #{i+1}", show=False, save_path=save_path)
        print(f"  Saved: {save_path}")

    # ── Render step-by-step trace ──────────────────────────────────
    if args.render_trace and trace_episode is not None:
        snaps, pieces = trace_episode
        save_path = str(out_dir / "episode_trace.png")
        render_episode_replay(snaps, pieces, save_path=save_path, show=False)
        print(f"  Saved trace: {save_path}")

    print(f"\nResults saved to: {out_dir}")
    return results


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
