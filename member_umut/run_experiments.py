"""
Run all experiments from the experiment matrix (§3.3 of ppo_action_plan).

| # | Encoder | Reward  | Masking | Seeds |
|---|---------|---------|---------|-------|
| 1 | MLP     | sparse  | Yes     | 5     |
| 2 | MLP     | shaped  | Yes     | 5     |
| 3 | CNN+MLP | sparse  | Yes     | 5     |
| 4 | CNN+MLP | shaped  | Yes     | 5     |
| 5 | MLP     | shaped  | No      | 5     |

Usage:
  # Run all 25 experiments
  python -m member_umut.run_experiments --total-timesteps 5000000

  # Quick sanity check (short run)
  python -m member_umut.run_experiments --total-timesteps 50000 --configs 2

  # Run a single config
  python -m member_umut.run_experiments --total-timesteps 5000000 --configs 4 --seeds 42
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product


SEEDS = [42, 123, 456, 789, 1024]

CONFIGS = [
    {"id": 1, "encoder": "mlp",     "reward": "sparse", "masking": True,  "desc": "MLP sparse masked"},
    {"id": 2, "encoder": "mlp",     "reward": "shaped", "masking": True,  "desc": "MLP shaped masked"},
    {"id": 3, "encoder": "cnn_mlp", "reward": "sparse", "masking": True,  "desc": "CNN+MLP sparse masked"},
    {"id": 4, "encoder": "cnn_mlp", "reward": "shaped", "masking": True,  "desc": "CNN+MLP shaped masked"},
    {"id": 5, "encoder": "mlp",     "reward": "shaped", "masking": False, "desc": "MLP shaped NO MASK (failure mode)"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run BrainBlock PPO experiments")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
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
    print(f"=== BrainBlock PPO Experiment Runner ===")
    print(f"Configs: {[c['id'] for c in configs_to_run]}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {total_runs}")
    print(f"Timesteps per run: {args.total_timesteps:,}")
    print("=" * 50)

    run_idx = 0
    for config in configs_to_run:
        for seed in seeds:
            run_idx += 1
            mask_tag = "nomask" if not config["masking"] else "mask"
            run_name = f"{config['encoder']}_{config['reward']}_{mask_tag}_seed{seed}"

            cmd = [
                sys.executable, "-m", "member_umut.train",
                "--reward", config["reward"],
                "--encoder", config["encoder"],
                "--seed", str(seed),
                "--total-timesteps", str(args.total_timesteps),
            ]
            if not config["masking"]:
                cmd.append("--no-masking")

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

    print("\n" + "=" * 50)
    print("All experiments completed.")


if __name__ == "__main__":
    main()
