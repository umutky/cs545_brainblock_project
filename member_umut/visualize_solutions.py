"""
Visualize all tilings discovered during training.

Usage:
  python -m member_umut.visualize_solutions \
      --tilings results/mlp_shaped_mask_seed42/discovered_tilings.json \
      --output  results/mlp_shaped_mask_seed42/solutions/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from common.pieces import PIECE_COLORS, PIECE_TYPES


def render_tiling(placed, ax, title=""):
    color_grid = [[PIECE_COLORS[None]] * 8 for _ in range(5)]
    for entry in placed:
        piece = entry["piece"]
        for cx, cy in entry["cells"]:
            color_grid[cy][cx] = PIECE_COLORS[piece]

    for row in range(5):
        for col in range(8):
            rect = mpatches.FancyBboxPatch(
                (col, 4 - row), 1, 1,
                boxstyle="round,pad=0.05",
                facecolor=color_grid[row][col],
                edgecolor="white", linewidth=1.5,
            )
            ax.add_patch(rect)

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tilings", type=str, required=True,
                        help="Path to discovered_tilings.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for PNG files")
    parser.add_argument("--max", type=int, default=None,
                        help="Max number of tilings to render (default: all)")
    parser.add_argument("--grid", action="store_true",
                        help="Save all tilings in one grid image")
    args = parser.parse_args()

    with open(args.tilings) as f:
        tilings = json.load(f)

    if args.max:
        tilings = tilings[:args.max]

    n = len(tilings)
    print(f"Rendering {n} tilings from {args.tilings}")

    out_dir = Path(args.output) if args.output else Path(args.tilings).parent / "solutions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.grid:
        # All tilings in one big grid image
        cols = min(10, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 1.5))
        axes = np.array(axes).flatten()

        for i, tiling in enumerate(tilings):
            title = f"#{tiling['tiling_id']} (ep {tiling['episode']})"
            render_tiling(tiling["placed"], axes[i], title=title)

        for j in range(n, len(axes)):
            axes[j].axis("off")

        plt.suptitle(f"All {n} Tilings Discovered During Training", fontsize=12)
        plt.tight_layout()
        grid_path = out_dir / "all_tilings_grid.png"
        plt.savefig(grid_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Grid saved: {grid_path}")
    else:
        # Individual PNGs
        for tiling in tilings:
            fig, ax = plt.subplots(figsize=(6, 4))
            title = f"Solution #{tiling['tiling_id']} — found at episode {tiling['episode']:,}"
            render_tiling(tiling["placed"], ax, title=title)

            legend = [mpatches.Patch(color=PIECE_COLORS[p], label=p) for p in PIECE_TYPES]
            ax.legend(handles=legend, loc="upper right", fontsize=8,
                      bbox_to_anchor=(1.12, 1))
            plt.tight_layout()

            path = out_dir / f"solution_{tiling['tiling_id']:03d}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"Saved {n} individual PNGs to {out_dir}/")

    print(f"Done. Total tilings: {n}")


if __name__ == "__main__":
    main()
