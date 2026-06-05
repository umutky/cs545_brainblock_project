"""
BrainBlock — Terminal Demo (Rich)

PPO agent solves the 8×5 tetromino packing puzzle in the terminal.
Auto-plays continuously; Ctrl+C to quit.

Usage:
  python terminal_demo.py --model results/ppo_r2_mlp_ent005_div1_seed42/best_model.pt
  python terminal_demo.py --model results/ppo_r2_mlp_ent005_div1_seed42/best_model.pt --speed 0.2
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.align import Align

sys.path.insert(0, ".")

from member_umut.environment import BrainBlockEnv
from member_umut.agent import PPOAgent, PPOConfig
from common.pieces import PIECE_TYPES, decode_action

# ── Color map ──────────────────────────────────────────────────────────

PIECE_STYLE = {
    "I": ("on #4FC3F7", "black"),  # cyan
    "O": ("on #FFD54F", "black"),  # yellow
    "L": ("on #FF8A65", "black"),  # orange
    "Z": ("on #81C784", "black"),  # green
    "T": ("on #CE93D8", "black"),  # purple
    None: ("on #2d3149", "white"), # empty — slightly lighter for contrast
}

PIECE_FG = {
    "I": "#4FC3F7",
    "O": "#FFD54F",
    "L": "#FF8A65",
    "Z": "#81C784",
    "T": "#CE93D8",
}

CELL = "  "         # 2 spaces = one cell content
HIGHLIGHT_CELL = "▓▓"
GRID_LINE = "on #0d0f1c"  # very dark — acts as cell separator
GRID_SEP_COL = " "        # 1-char column separator
GRID_SEP_ROW = " " * (8 * len(CELL) + 7 * len(GRID_SEP_COL))  # full-width row separator


def build_board_text(placed, new_cells=None):
    """Render the 8×5 grid with visible grid lines between cells."""
    if new_cells is None:
        new_cells = set()

    cell_map: dict[tuple[int,int], str] = {}
    for piece_type, cells in placed:
        for cx, cy in cells:
            cell_map[(cx, cy)] = piece_type

    lines = Text()
    for row in range(4, -1, -1):          # top row = y=4
        # ── data row ──────────────────────────────────────────────────
        for col in range(8):
            if col > 0:
                lines.append(GRID_SEP_COL, style=GRID_LINE)
            pt = cell_map.get((col, row))
            is_new = (col, row) in new_cells
            bg, _ = PIECE_STYLE.get(pt, PIECE_STYLE[None])
            char = HIGHLIGHT_CELL if is_new else CELL
            lines.append(char, style=bg)
        lines.append("\n")
        # ── row separator (between rows, not after last) ───────────────
        if row > 0:
            lines.append(GRID_SEP_ROW, style=GRID_LINE)
            lines.append("\n")
    return lines


def build_queue_panel(env, step, total_reward, status, episode, speed):
    """Build the right-side info panel."""
    t = Text()

    # Episode & step
    t.append(f"Episode  ", style="dim")
    t.append(f"#{episode}\n", style="bold white")
    t.append(f"Step     ", style="dim")
    t.append(f"{step}", style="bold white")
    t.append(f" / 10\n\n", style="dim")

    # Coverage bar
    coverage = float(env.board.sum()) / 40.0 if env.board is not None else 0.0
    filled = int(coverage * 16)
    bar = "█" * filled + "░" * (16 - filled)
    bar_color = "green" if coverage >= 1.0 else "cyan"
    t.append(f"Coverage ", style="dim")
    t.append(f"{coverage:.0%}\n", style="bold white")
    t.append(bar + "\n\n", style=bar_color)

    # Reward
    r_color = "green" if total_reward > 0 else "red" if total_reward < 0 else "white"
    t.append("Reward   ", style="dim")
    t.append(f"{total_reward:+.3f}\n\n", style=f"bold {r_color}")

    # Current piece
    if env._queue:
        cur = env._current
        t.append("Current  ", style="dim")
        t.append(f"[{cur}]\n\n", style=f"bold {PIECE_FG.get(cur, 'white')}")

        # Remaining queue — show ordered sequence as colored dots
        tail = env._queue[1:]
        if tail:
            t.append("Up next  ", style="dim")
            for p in tail[:9]:
                t.append(f"[{p}]", style=f"bold {PIECE_FG[p]}")
                t.append(" ", style="")
            t.append("\n")
    t.append("\n")

    # Status
    if "SUCCESS" in status:
        t.append(f"✔  {status}\n", style="bold green")
    elif "DEAD" in status or "ILLEGAL" in status:
        t.append(f"✘  {status}\n", style="bold red")
    elif status:
        t.append(f"   {status}\n", style="white")
    else:
        t.append("   playing…\n", style="dim")

    t.append(f"\nspeed {speed}s  |  Ctrl+C quit", style="dim")
    return t


def render_frame(env, step, total_reward, status, episode, speed, new_cells):
    """Return a single Rich renderable for the current frame."""
    board_text = build_board_text(env._placed, new_cells)
    board_panel = Panel(
        board_text,
        title="[bold cyan]BrainBlock[/bold cyan]",
        subtitle=f"[dim]8 × 5 tetromino packing[/dim]",
        border_style="cyan",
        padding=(0, 1),
    )

    info_text = build_queue_panel(env, step, total_reward, status, episode, speed)
    info_panel = Panel(
        info_text,
        title="[bold]Info[/bold]",
        border_style="dim",
        width=32,
        padding=(0, 1),
    )

    from rich.table import Table as RTable
    layout = RTable.grid(padding=(0, 1))
    layout.add_column()
    layout.add_column()
    layout.add_row(board_panel, info_panel)
    return layout


def load_agent(model_path, encoder="mlp", hidden_dim=256):
    config = PPOConfig(encoder_type=encoder, hidden_dim=hidden_dim)
    agent = PPOAgent(config, device=torch.device("cpu"))
    agent.load(model_path)
    agent.network.eval()
    return agent


def choose_action(agent, obs, action_mask):
    with torch.no_grad():
        grid = torch.tensor(obs["grid"]).unsqueeze(0)
        vec  = torch.tensor(obs["vec"]).unsqueeze(0)
        mask = torch.tensor(action_mask.astype(np.float32)).unsqueeze(0)
        dist, _ = agent.network(grid, vec, mask)
        action = dist.probs.argmax(dim=-1).item()
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--encoder", default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--speed", type=float, default=0.2,
                        help="Seconds between steps")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=0,
                        help="Stop after N episodes (0 = run forever)")
    args = parser.parse_args()

    agent = load_agent(args.model, args.encoder, args.hidden_dim)
    env   = BrainBlockEnv(reward_mode="shaped")
    console = Console()

    episode   = 0
    rng = np.random.default_rng(args.seed if args.seed is not None else None)

    with Live(console=console, refresh_per_second=20, screen=False) as live:
        while True:
            seed = int(rng.integers(0, 100_000))
            obs, info = env.reset(seed=seed)
            action_mask = info["action_mask"]

            step = 0
            total_reward = 0.0
            status = ""
            done = False
            new_cells: set = set()

            # Show initial board
            live.update(render_frame(env, step, total_reward, status,
                                     episode + 1, args.speed, new_cells))
            time.sleep(args.speed)

            while not done:
                action = choose_action(agent, obs, action_mask)

                placed_before = len(env._placed)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1

                new_cells = set()
                if len(env._placed) > placed_before:
                    _, cells = env._placed[-1]
                    new_cells = cells

                reason = next_info.get("termination_reason", "")
                if reason == "success":
                    status = "SUCCESS — Board solved!"
                elif reason == "dead_end":
                    cov = next_info.get("coverage", 0)
                    status = f"DEAD END  {cov:.0%}"
                elif reason == "illegal_action":
                    status = "ILLEGAL ACTION"

                obs = next_obs
                action_mask = next_info.get("action_mask", action_mask)

                live.update(render_frame(env, step, total_reward, status,
                                         episode + 1, args.speed, new_cells))
                time.sleep(args.speed)

            # Pause on final frame
            time.sleep(args.speed * 4)
            episode += 1
            if args.episodes > 0 and episode >= args.episodes:
                break


if __name__ == "__main__":
    main()
