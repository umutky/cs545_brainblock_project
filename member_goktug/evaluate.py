"""
Evaluation script for BrainBlock DQN (Member B2).

Usage:
  python -m member_goktug.evaluate --model results/dqn_shaped_mlp_seed42/best_model.pt \
                                --encoder mlp --reward shaped --episodes 1000 --seed 42

Features:
  - Deterministic (ε=0) and stochastic evaluation modes
  - Success rate, mean return ± std, coverage, episode length
  - Unique tiling discovery and counting
  - Solution visualization saved as PNG
  - Summary saved as JSON
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from member_goktug.agent import DQNAgent
from member_goktug.environment import BrainBlockEnv
from common.visualize import render_board, render_episode_replay


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DQN on BrainBlock (B2)")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--encoder", type=str, default="mlp",
                        choices=["mlp", "cnn_mlp"], help="Encoder architecture")
    parser.add_argument("--reward", type=str, default="shaped",
                        choices=["sparse", "shaped"], help="Reward mode")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (ε=0.05) instead of greedy")
    parser.add_argument("--render-solutions", type=int, default=5,
                        help="Max number of solution boards to render as PNG")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (auto if not set)")
    return parser.parse_args()


def board_to_tiling_key(placed: list[tuple[str, set]]):
    """
    Convert placed pieces to a canonical tiling representation.
    A tiling is a frozenset of (piece_type, frozenset of (x, y) cells).
    Placement order does NOT matter.
    """
    return frozenset(
        (piece_type, frozenset(cells)) for piece_type, cells in placed
    )


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        model_dir = Path(args.model).parent
        mode = "stoch" if args.stochastic else "det"
        model_tag = "best" if "best" in args.model else "final"
        out_dir = model_dir / f"eval_{model_tag}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Environment
    env = BrainBlockEnv(reward_mode=args.reward)

    # Agent
    agent = DQNAgent(encoder_type=args.encoder, device=str(device))
    agent.load(args.model)
    agent.q_net.eval()

    eps = 0.05 if args.stochastic else 0.0

    # Metrics
    rewards = []
    lengths = []
    successes = []
    coverages = []
    invalids = []
    seen_tilings = set()
    solution_count = 0

    print(f"Evaluating {args.model}")
    print(f"  Mode: {'stochastic (ε=0.05)' if args.stochastic else 'deterministic (ε=0)'}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Reward: {args.reward} | Encoder: {args.encoder}")
    print("-" * 50)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        action_mask = info["action_mask"]

        ep_reward = 0.0
        ep_length = 0
        done = False

        while not done:
            action = agent.act(obs, action_mask, epsilon=eps)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            action_mask = info["action_mask"]
            ep_reward += reward
            ep_length += 1

        reason = info.get("termination_reason", "unknown")
        rewards.append(ep_reward)
        lengths.append(ep_length)
        successes.append(1.0 if reason == "success" else 0.0)
        coverages.append(info.get("coverage", 0.0))
        invalids.append(1.0 if reason == "illegal_action" else 0.0)

        # Track unique tilings
        if reason == "success":
            tiling_key = board_to_tiling_key(env._placed)
            if tiling_key not in seen_tilings:
                seen_tilings.add(tiling_key)

                # Render solution visualization
                if solution_count < args.render_solutions:
                    render_board(
                        env._placed,
                        title=f"Solution #{solution_count + 1} (ep {ep})",
                        show=False,
                        save_path=str(out_dir / f"solution_{solution_count + 1}.png")
                    )
                solution_count += 1

    # Summary
    results = {
        "model": args.model,
        "reward_mode": args.reward,
        "encoder": args.encoder,
        "mode": "stochastic" if args.stochastic else "deterministic",
        "epsilon": eps,
        "episodes": args.episodes,
        "seed": args.seed,
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "mean_coverage": float(np.mean(coverages)),
        "invalid_rate": float(np.mean(invalids)),
        "unique_tilings": len(seen_tilings),
    }

    # Save results
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Results ({args.episodes} episodes):")
    print(f"  Success rate:   {results['success_rate']:.4f}")
    print(f"  Mean reward:    {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"  Mean length:    {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"  Mean coverage:  {results['mean_coverage']:.4f}")
    print(f"  Invalid rate:   {results['invalid_rate']:.4f}")
    print(f"  Unique tilings: {results['unique_tilings']}")
    print(f"  Saved to: {out_dir}")

    return results


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
