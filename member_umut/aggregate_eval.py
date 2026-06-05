"""
Aggregate evaluation results across multiple seeds into a summary table.

Usage:
  # Aggregate one config (e.g. MLP shaped masked, seeds 42 123 456 789 1024):
  python -m member_umut.aggregate_eval \\
      --dirs results/mlp_shaped_mask_seed42/eval \\
             results/mlp_shaped_mask_seed123/eval \\
             results/mlp_shaped_mask_seed456/eval \\
             results/mlp_shaped_mask_seed789/eval \\
             results/mlp_shaped_mask_seed1024/eval \\
      --label "MLP shaped masked"

  # Compare all 5 configs at once (after running all experiments + evals):
  python -m member_umut.aggregate_eval --all-configs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = [42, 123, 456, 789, 1024]

CONFIGS = [
    {"id": 1, "encoder": "mlp",     "reward": "sparse", "mask": "mask",   "desc": "MLP sparse masked"},
    {"id": 2, "encoder": "mlp",     "reward": "shaped", "mask": "mask",   "desc": "MLP shaped masked"},
    {"id": 3, "encoder": "cnn_mlp", "reward": "sparse", "mask": "mask",   "desc": "CNN+MLP sparse masked"},
    {"id": 4, "encoder": "cnn_mlp", "reward": "shaped", "mask": "mask",   "desc": "CNN+MLP shaped masked"},
    {"id": 5, "encoder": "mlp",     "reward": "shaped", "mask": "nomask", "desc": "MLP shaped NO MASK"},
]

METRICS = [
    ("Success rate",       "success_rate"),
    ("Mean return",        "mean_reward"),
    ("Mean ep length",     "mean_length"),
    ("Invalid-act rate",   "invalid_rate"),
    ("Mean coverage",      "mean_coverage"),
    ("Unique solutions",   "unique_solutions_found"),
]


def load_results(dirs: list[str]) -> list[dict]:
    results = []
    for d in dirs:
        path = Path(d) / "eval_results.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            results.append(json.load(f))
    return results


def aggregate(results: list[dict], label: str = "") -> dict:
    summary = {"label": label, "n_seeds": len(results)}
    for _, key in METRICS:
        vals = [r[key] for r in results if key in r]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
    return summary


def print_table(summary: dict):
    label = summary.get("label", "")
    n = summary.get("n_seeds", "?")
    print(f"\n{'='*60}")
    print(f"Config: {label}  (n={n} seeds)")
    print(f"{'='*60}")
    for display_name, key in METRICS:
        m = summary.get(f"{key}_mean")
        s = summary.get(f"{key}_std")
        if m is not None:
            print(f"  {display_name:<22} {m:.4f} ± {s:.4f}")
    print()


def run_all_configs(results_root: str = "results", seeds: list[int] = None):
    if seeds is None:
        seeds = SEEDS
    all_summaries = []
    for cfg in CONFIGS:
        dirs = [
            f"{results_root}/{cfg['encoder']}_{cfg['reward']}_{cfg['mask']}_seed{s}/eval"
            for s in seeds
        ]
        results = load_results(dirs)
        if not results:
            print(f"  Skipping config {cfg['id']} — no eval results found")
            continue
        summary = aggregate(results, label=cfg["desc"])
        print_table(summary)
        all_summaries.append(summary)
    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Aggregate eval results across seeds")
    parser.add_argument("--dirs", nargs="+", default=None,
                        help="Eval output directories (one per seed)")
    parser.add_argument("--label", type=str, default="",
                        help="Label for this config")
    parser.add_argument("--save", type=str, default=None,
                        help="Save aggregated JSON to this path")
    parser.add_argument("--all-configs", action="store_true",
                        help="Aggregate all 5 configs automatically")
    parser.add_argument("--results-root", type=str, default="results",
                        help="Root directory for results (used with --all-configs)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    if args.all_configs:
        summaries = run_all_configs(args.results_root, args.seeds)
        if args.save:
            with open(args.save, "w") as f:
                json.dump(summaries, f, indent=2)
            print(f"Saved all summaries to: {args.save}")
        return

    if not args.dirs:
        parser.error("Provide --dirs or --all-configs")

    results = load_results(args.dirs)
    if not results:
        print("No valid results found.")
        return

    summary = aggregate(results, label=args.label)
    print_table(summary)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to: {args.save}")


if __name__ == "__main__":
    main()
