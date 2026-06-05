"""
Run all DQN experiments for Member B2.

| # | Encoder | Reward  | Seeds |
|---|---------|---------|-------|
| 1 | MLP     | sparse  | 5     |
| 2 | MLP     | shaped  | 5     |
| 3 | CNN+MLP | shaped  | 5     |

Usage:
  # Run all 15 experiments
  python -m member_goktug.run_experiments --total-timesteps 2000000

  # Quick sanity check
  python -m member_goktug.run_experiments --total-timesteps 50000 --configs 2

  # Single config + seed
  python -m member_goktug.run_experiments --total-timesteps 2000000 --configs 2 --seeds 42
"""

from __future__ import annotations

import argparse
import subprocess
import sys


SEEDS = [42, 123, 456, 789, 1024]

CONFIGS = [
    {"id": 1, "encoder": "mlp",     "reward": "sparse", "desc": "MLP sparse (R1)"},
    {"id": 2, "encoder": "mlp",     "reward": "shaped", "desc": "MLP shaped (R2)"},
    {"id": 3, "encoder": "cnn_mlp", "reward": "shaped", "desc": "CNN+MLP shaped (R2)"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run BrainBlock DQN experiments")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--configs", nargs="+", type=int, default=None,
                        help="Config IDs to run (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds to use (default: all 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    return parser.parse_args()


def main():
    args = parse_args()

    configs_to_run = CONFIGS
    if args.configs:
        configs_to_run = [c for c in CONFIGS if c["id"] in args.configs]

    seeds = args.seeds or SEEDS

    total_runs = len(configs_to_run) * len(seeds)
    print(f"=== BrainBlock DQN Experiment Runner (Member B2) ===")
    print(f"Configs: {[c['id'] for c in configs_to_run]}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {total_runs}")
    print(f"Timesteps per run: {args.total_timesteps:,}")
    print("=" * 55)

    run_idx = 0
    for config in configs_to_run:
        for seed in seeds:
            run_idx += 1
            reward_short = "r1" if config["reward"] == "sparse" else "r2"
            run_name = f"dqn_{reward_short}_{config['encoder']}_seed{seed}"

            cmd = [
                sys.executable, "-m", "member_goktug.train",
                "--reward", config["reward"],
                "--encoder", config["encoder"],
                "--seed", str(seed),
                "--total-timesteps", str(args.total_timesteps),
                "--output-dir", f"results/{run_name}"
            ]

            print(f"\n[{run_idx}/{total_runs}] {config['desc']} | seed={seed}")
            print(f"  → {run_name}")

            if args.dry_run:
                print(f"  CMD: {' '.join(cmd)}")
                continue

            print(f"  CMD: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=".")
            if result.returncode != 0:
                print(f"  ❌ FAILED (exit code {result.returncode})")
            else:
                print(f"  ✅ Done")

    print("\n" + "=" * 55)
    print("All experiments completed.")


if __name__ == "__main__":
    main()
