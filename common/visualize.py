"""
Board visualization utilities for BrainBlock.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from common.pieces import PIECE_TYPES, PIECE_COLORS


def render_board(
    placed: list[tuple[str, set]],
    title: str = "BrainBlock",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Render the board state from a list of placed pieces.

    Args:
        placed: list of (piece_type, set of (x, y) cells)
        title: plot title
        ax: optional matplotlib axes to draw on
        show: if True, display the plot
        save_path: if provided, save figure to this path

    Returns:
        RGB image array if ax was None, else None.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    color_grid = [[PIECE_COLORS[None]] * 8 for _ in range(5)]
    for piece_type, cells in placed:
        for cx, cy in cells:
            color_grid[cy][cx] = PIECE_COLORS[piece_type]

    for row in range(5):
        for col in range(8):
            rect = mpatches.FancyBboxPatch(
                (col, 4 - row), 1, 1,
                boxstyle="round,pad=0.05",
                facecolor=color_grid[row][col],
                edgecolor="white", linewidth=2,
            )
            ax.add_patch(rect)

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=PIECE_COLORS[p], label=p) for p in PIECE_TYPES
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9,
              bbox_to_anchor=(1.12, 1))
    ax.set_title(title, fontsize=11)

    plt.tight_layout()

    if own_fig:
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        plt.close(fig)
        return img

    return None


def render_episode_replay(
    board_snapshots: list[np.ndarray],
    piece_history: list[str],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Replay a full episode from saved board states.

    Args:
        board_snapshots: list of board arrays (5×8) after each placement.
        piece_history: piece type placed at each step.
        save_path: if provided, save figure to this path.
    """
    n = len(board_snapshots)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = np.array(axes).flatten()

    placed_so_far: list[tuple[str, set]] = []
    prev_board = np.zeros((5, 8), dtype=np.int8)

    for i, (board_snap, piece_type) in enumerate(zip(board_snapshots, piece_history)):
        diff = np.argwhere(board_snap - prev_board)
        new_cells = {(c, r) for r, c in diff}
        placed_so_far.append((piece_type, new_cells))
        prev_board = board_snap.copy()

        color_grid = [[PIECE_COLORS[None]] * 8 for _ in range(5)]
        for pt, cells in placed_so_far:
            for cx, cy in cells:
                color_grid[cy][cx] = PIECE_COLORS[pt]

        ax = axes[i]
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
        ax.set_title(f"Step {i+1}: {piece_type}", fontsize=8)

    for j in range(len(board_snapshots), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Episode Replay", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
