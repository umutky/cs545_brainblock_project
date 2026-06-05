"""
Pygame Live Demo — Watch the PPO agent solve BrainBlock step by step.

Usage:
  # With a trained model:
  python -m member_umut.demo --model results/mlp_shaped_mask_seed42/final_model.pt --encoder mlp

  # Without a model (random agent — useful for testing the visualization):
  python -m member_umut.demo --random

  # With backtracking solver (perfect play):
  python -m member_umut.demo --solver

Controls:
  SPACE / ENTER  — Next step (when paused)
  A              — Toggle auto-play
  R              — Reset (new episode)
  +/-            — Speed up / slow down auto-play
  Q / ESC        — Quit
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import numpy as np
import pygame
import torch

from member_umut.environment import BrainBlockEnv
from member_umut.agent import PPOAgent, PPOConfig
from common.pieces import (
    PIECE_TYPES, PIECE_IDX, PIECE_COLORS, ORIENT_TABLE,
    decode_action, encode_action, VALID_ORIENTS,
)

# ── Colors ─────────────────────────────────────────────────────────────

# Pygame-friendly RGB tuples
def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

COLORS = {k: hex_to_rgb(v) for k, v in PIECE_COLORS.items()}

# UI colors
BG_COLOR = (18, 18, 24)
GRID_BG = (30, 32, 44)
GRID_LINE = (50, 55, 75)
TEXT_COLOR = (220, 225, 240)
TEXT_DIM = (120, 130, 155)
ACCENT = (100, 180, 255)
SUCCESS_COLOR = (100, 220, 130)
FAIL_COLOR = (255, 100, 100)
PANEL_BG = (25, 27, 38)
SHADOW_COLOR = (0, 0, 0, 80)

# Piece colors (brighter, more saturated for Pygame)
PIECE_RGB = {
    "I": (79, 195, 247),
    "O": (255, 213, 79),
    "L": (255, 138, 101),
    "Z": (129, 199, 132),
    "T": (206, 147, 216),
    None: (45, 50, 65),
}

PIECE_RGB_HIGHLIGHT = {
    "I": (130, 215, 255),
    "O": (255, 230, 130),
    "L": (255, 175, 145),
    "Z": (170, 220, 170),
    "T": (225, 185, 235),
}


# ── Constants ──────────────────────────────────────────────────────────

CELL_SIZE = 72
BOARD_W, BOARD_H = 8, 5
BOARD_PX_W = BOARD_W * CELL_SIZE
BOARD_PX_H = BOARD_H * CELL_SIZE

MARGIN = 40
PANEL_W = 280
WIN_W = MARGIN + BOARD_PX_W + MARGIN + PANEL_W + MARGIN
WIN_H = MARGIN + BOARD_PX_H + MARGIN + 120

BOARD_X = MARGIN
BOARD_Y = MARGIN + 50

FPS = 60


# ── Solver (for --solver mode) ────────────────────────────────────────

def find_solution_for_queue(queue: list[str]):
    """Backtracking solver."""
    from itertools import product as iprod
    board = np.zeros((5, 8), dtype=np.int8)
    moves = []

    def dfs(step):
        if step == len(queue):
            return board.sum() == 40
        piece = queue[step]
        for orient in VALID_ORIENTS[piece]:
            cells = ORIENT_TABLE[piece][orient]
            for x, y in iprod(range(8), range(5)):
                if all(
                    0 <= x + dx < 8 and 0 <= y + dy < 5 and board[y + dy, x + dx] == 0
                    for dx, dy in cells
                ):
                    for dx, dy in cells:
                        board[y + dy, x + dx] = 1
                    moves.append((orient, x, y))
                    if dfs(step + 1):
                        return True
                    moves.pop()
                    for dx, dy in cells:
                        board[y + dy, x + dx] = 0
        return False

    return moves if dfs(0) else None


# ── Drawing helpers ────────────────────────────────────────────────────

def draw_rounded_rect(surface, color, rect, radius=8):
    """Draw a rounded rectangle."""
    pygame.draw.rect(surface, color, rect, border_radius=radius)


def draw_cell(surface, x, y, color, highlight=False, newly_placed=False):
    """Draw a single board cell with styling."""
    px = BOARD_X + x * CELL_SIZE
    py = BOARD_Y + (BOARD_H - 1 - y) * CELL_SIZE  # flip y for display

    # Cell background
    cell_rect = pygame.Rect(px + 2, py + 2, CELL_SIZE - 4, CELL_SIZE - 4)
    draw_rounded_rect(surface, color, cell_rect, radius=6)

    if newly_placed:
        # Glow effect for newly placed piece
        glow_rect = pygame.Rect(px, py, CELL_SIZE, CELL_SIZE)
        glow_surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*color, 60), (0, 0, CELL_SIZE, CELL_SIZE),
                         border_radius=8)
        surface.blit(glow_surf, (px, py))

    if highlight:
        # Border highlight
        pygame.draw.rect(surface, (255, 255, 255), cell_rect, width=2,
                         border_radius=6)

    # Inner shine (top-left highlight)
    shine_rect = pygame.Rect(px + 4, py + 4, CELL_SIZE // 2 - 4, CELL_SIZE // 4)
    shine_surf = pygame.Surface((shine_rect.w, shine_rect.h), pygame.SRCALPHA)
    pygame.draw.rect(shine_surf, (255, 255, 255, 30), (0, 0, shine_rect.w, shine_rect.h),
                     border_radius=3)
    surface.blit(shine_surf, shine_rect.topleft)


def draw_board(surface, placed, new_cells=None):
    """Draw the full board."""
    if new_cells is None:
        new_cells = set()

    # Board background with shadow
    shadow_rect = pygame.Rect(BOARD_X + 4, BOARD_Y + 4, BOARD_PX_W, BOARD_PX_H)
    shadow_surf = pygame.Surface((BOARD_PX_W, BOARD_PX_H), pygame.SRCALPHA)
    shadow_surf.fill((0, 0, 0, 40))
    surface.blit(shadow_surf, shadow_rect.topleft)

    board_rect = pygame.Rect(BOARD_X, BOARD_Y, BOARD_PX_W, BOARD_PX_H)
    draw_rounded_rect(surface, GRID_BG, board_rect, radius=10)

    # Build color map
    cell_map = {}
    for piece_type, cells in placed:
        for cx, cy in cells:
            cell_map[(cx, cy)] = piece_type

    # Draw cells
    for col in range(BOARD_W):
        for row in range(BOARD_H):
            piece_type = cell_map.get((col, row))
            color = PIECE_RGB.get(piece_type, PIECE_RGB[None])
            is_new = (col, row) in new_cells
            draw_cell(surface, col, row, color, highlight=is_new, newly_placed=is_new)

    # Grid lines
    for i in range(BOARD_W + 1):
        x = BOARD_X + i * CELL_SIZE
        pygame.draw.line(surface, GRID_LINE, (x, BOARD_Y), (x, BOARD_Y + BOARD_PX_H), 1)
    for j in range(BOARD_H + 1):
        y = BOARD_Y + j * CELL_SIZE
        pygame.draw.line(surface, GRID_LINE, (BOARD_X, y), (BOARD_X + BOARD_PX_W, y), 1)

    # Coordinate labels
    font_small = pygame.font.SysFont("Menlo", 12)
    for col in range(BOARD_W):
        lbl = font_small.render(str(col), True, TEXT_DIM)
        surface.blit(lbl, (BOARD_X + col * CELL_SIZE + CELL_SIZE // 2 - lbl.get_width() // 2,
                           BOARD_Y + BOARD_PX_H + 4))
    for row in range(BOARD_H):
        lbl = font_small.render(str(row), True, TEXT_DIM)
        surface.blit(lbl, (BOARD_X - 16,
                           BOARD_Y + (BOARD_H - 1 - row) * CELL_SIZE + CELL_SIZE // 2 - 6))


def draw_panel(surface, env, step, total_reward, status, mode_label, auto_play, speed):
    """Draw the info panel on the right."""
    px = BOARD_X + BOARD_PX_W + MARGIN
    py = BOARD_Y

    # Panel background
    panel_rect = pygame.Rect(px, py, PANEL_W, BOARD_PX_H)
    draw_rounded_rect(surface, PANEL_BG, panel_rect, radius=10)

    font = pygame.font.SysFont("Menlo", 15)
    font_bold = pygame.font.SysFont("Menlo", 15, bold=True)
    font_large = pygame.font.SysFont("Menlo", 18, bold=True)
    font_small = pygame.font.SysFont("Menlo", 12)

    y = py + 15
    line_h = 24

    # Mode
    mode_surf = font_bold.render(f"Mode: {mode_label}", True, ACCENT)
    surface.blit(mode_surf, (px + 15, y))
    y += line_h + 5

    # Divider
    pygame.draw.line(surface, GRID_LINE, (px + 15, y), (px + PANEL_W - 15, y), 1)
    y += 10

    # Step
    step_surf = font.render(f"Step: {step}/10", True, TEXT_COLOR)
    surface.blit(step_surf, (px + 15, y))
    y += line_h

    # Coverage
    coverage = env.board.sum() / 40.0 if env.board is not None else 0
    cov_surf = font.render(f"Coverage: {coverage:.0%}", True, TEXT_COLOR)
    surface.blit(cov_surf, (px + 15, y))
    y += line_h

    # Coverage bar
    bar_x = px + 15
    bar_w = PANEL_W - 30
    bar_h = 14
    pygame.draw.rect(surface, GRID_BG, (bar_x, y, bar_w, bar_h), border_radius=4)
    fill_w = int(bar_w * coverage)
    if fill_w > 0:
        bar_color = SUCCESS_COLOR if coverage >= 1.0 else ACCENT
        pygame.draw.rect(surface, bar_color, (bar_x, y, fill_w, bar_h), border_radius=4)
    y += bar_h + 10

    # Reward
    rew_surf = font.render(f"Reward: {total_reward:+.3f}", True, TEXT_COLOR)
    surface.blit(rew_surf, (px + 15, y))
    y += line_h

    # Current piece
    if env._queue and len(env._queue) > 0:
        piece = env._current
        piece_color = PIECE_RGB.get(piece, TEXT_COLOR)
        cur_surf = font.render(f"Current: ", True, TEXT_COLOR)
        piece_surf = font_bold.render(piece, True, piece_color)
        surface.blit(cur_surf, (px + 15, y))
        surface.blit(piece_surf, (px + 15 + cur_surf.get_width(), y))
    y += line_h

    # Remaining pieces
    remain_surf = font.render("Remaining:", True, TEXT_DIM)
    surface.blit(remain_surf, (px + 15, y))
    y += line_h

    if env._queue:
        tail = env._queue[1:] if len(env._queue) > 1 else []
        counts = {}
        for p in tail:
            counts[p] = counts.get(p, 0) + 1
        for p in PIECE_TYPES:
            c = counts.get(p, 0)
            color = PIECE_RGB.get(p, TEXT_DIM) if c > 0 else TEXT_DIM
            txt = font_small.render(f"  {p}: {c}", True, color)
            surface.blit(txt, (px + 15, y))
            y += 18
    y += 10

    # Divider
    pygame.draw.line(surface, GRID_LINE, (px + 15, y), (px + PANEL_W - 15, y), 1)
    y += 10

    # Status
    if status:
        status_color = SUCCESS_COLOR if "SUCCESS" in status else (
            FAIL_COLOR if "DEAD" in status or "ILLEGAL" in status else TEXT_COLOR
        )
        status_surf = font_bold.render(status, True, status_color)
        surface.blit(status_surf, (px + 15, y))
    y += line_h + 5

    # Controls
    auto_str = "ON" if auto_play else "OFF"
    ctrl_lines = [
        f"Auto: {auto_str}  Speed: {speed:.1f}s",
        "SPACE  next step",
        "A      toggle auto",
        "+/-    speed",
        "R      reset",
        "Q      quit",
    ]
    for line in ctrl_lines:
        ctrl_surf = font_small.render(line, True, TEXT_DIM)
        surface.blit(ctrl_surf, (px + 15, y))
        y += 16


def draw_title(surface, title):
    """Draw the title bar."""
    font_title = pygame.font.SysFont("Menlo", 22, bold=True)
    title_surf = font_title.render(title, True, TEXT_COLOR)
    surface.blit(title_surf, (MARGIN, 15))


def draw_footer(surface, text):
    """Draw footer text."""
    font_footer = pygame.font.SysFont("Menlo", 12)
    footer_surf = font_footer.render(text, True, TEXT_DIM)
    y = WIN_H - 30
    surface.blit(footer_surf, (MARGIN, y))


# ── Main Demo Loop ────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="BrainBlock Pygame Demo")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "sac"],
                        help="Algorithm: ppo (default), dqn, or sac")
    parser.add_argument("--encoder", type=str, default="mlp",
                        choices=["mlp", "cnn_mlp"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--random", action="store_true",
                        help="Use random (legal) actions instead of model")
    parser.add_argument("--solver", action="store_true",
                        help="Use backtracking solver (perfect play)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fixed seed (random each episode if not set)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Auto-play delay in seconds")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample from policy instead of argmax (more diverse episodes)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Determine mode ────────────────────────────────────────────
    agent = None
    solver_moves = None
    is_sac = False

    if args.solver:
        mode_label = "SOLVER"
    elif args.random:
        mode_label = "RANDOM"
    elif args.model:
        device = torch.device("cpu")
        if args.algo == "dqn":
            from member_goktug.agent import DQNAgent
            agent = DQNAgent(encoder_type=args.encoder, hidden_dim=args.hidden_dim, device="cpu")
            agent.load(args.model)
            agent.q_net.eval()
            # DQN is always deterministic (argmax); --stochastic adds ε=0.05
            mode_label = "DQN (ε=0.05)" if args.stochastic else "DQN (det, ε=0)"
        elif args.algo == "sac":
            from member_umut.sac_agent import SACAgent, SACConfig
            is_sac = True
            config = SACConfig(encoder_type=args.encoder, hidden_dim=args.hidden_dim)
            agent = SACAgent(config, device)
            agent.load(args.model)
            agent.actor.eval()
            mode_label = f"SAC ({'stoch' if args.stochastic else 'det'})"
        else:
            config = PPOConfig(encoder_type=args.encoder, hidden_dim=args.hidden_dim)
            agent = PPOAgent(config, device)
            agent.load(args.model)
            agent.network.eval()
            mode_label = f"PPO ({'stoch' if args.stochastic else 'det'})"
        print(f"Loaded {args.algo.upper()} model: {args.model}")
    else:
        print("Error: specify --model, --random, or --solver")
        sys.exit(1)

    # ── Pygame init ───────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("BrainBlock — Agent Demo")
    clock = pygame.time.Clock()

    # ── Environment ───────────────────────────────────────────────
    env = BrainBlockEnv(reward_mode="shaped")
    episode_seed = args.seed if args.seed is not None else np.random.randint(0, 100000)
    obs, info = env.reset(seed=episode_seed)
    action_mask = info["action_mask"]

    step_count = 0
    total_reward = 0.0
    status = ""
    done = False
    new_cells = set()  # cells placed in the last step (for highlight)

    auto_play = False
    speed = args.speed
    last_auto_time = time.time()

    # If solver mode, pre-compute solution
    if args.solver:
        solver_moves = find_solution_for_queue(env._queue.copy())
        if solver_moves is None:
            status = "SOLVER: NO SOLUTION FOUND"
        else:
            status = f"SOLVER: found {len(solver_moves)}-step solution"

    episode_count = 0

    # ── Main loop ─────────────────────────────────────────────────
    running = True
    while running:
        # ── Events ────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_a:
                    auto_play = not auto_play
                    last_auto_time = time.time()

                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if not done:
                        # Take one step
                        new_cells = _do_step(env, agent, action_mask, obs,
                                             solver_moves, step_count, args)
                        obs_result = _post_step(env)
                        obs, action_mask, step_count, total_reward, status, done, new_cells = (
                            _update_state(env, obs, action_mask, step_count,
                                          total_reward, new_cells, obs_result)
                        )
                    else:
                        # Reset
                        episode_seed = args.seed if args.seed is not None else np.random.randint(0, 100000)
                        obs, info = env.reset(seed=episode_seed)
                        action_mask = info["action_mask"]
                        step_count = 0
                        total_reward = 0.0
                        status = ""
                        done = False
                        new_cells = set()
                        episode_count += 1
                        if args.solver:
                            solver_moves = find_solution_for_queue(env._queue.copy())
                            if solver_moves is None:
                                status = "SOLVER: NO SOLUTION"

                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = max(0.1, speed - 0.2)

                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = min(5.0, speed + 0.2)

                elif event.key == pygame.K_r:
                    episode_seed = args.seed if args.seed is not None else np.random.randint(0, 100000)
                    obs, info = env.reset(seed=episode_seed)
                    action_mask = info["action_mask"]
                    step_count = 0
                    total_reward = 0.0
                    status = ""
                    done = False
                    new_cells = set()
                    episode_count += 1
                    if args.solver:
                        solver_moves = find_solution_for_queue(env._queue.copy())

        # ── Auto-play ─────────────────────────────────────────────
        if auto_play and not done and time.time() - last_auto_time >= speed:
            new_cells = _do_step(env, agent, action_mask, obs,
                                 solver_moves, step_count, args, is_sac)
            obs_result = _post_step(env)
            obs, action_mask, step_count, total_reward, status, done, new_cells = (
                _update_state(env, obs, action_mask, step_count,
                              total_reward, new_cells, obs_result)
            )
            last_auto_time = time.time()

            if done and auto_play:
                # Auto-reset after a brief pause
                pygame.time.wait(int(speed * 1500))
                episode_seed = args.seed if args.seed is not None else np.random.randint(0, 100000)
                obs, info = env.reset(seed=episode_seed)
                action_mask = info["action_mask"]
                step_count = 0
                total_reward = 0.0
                status = ""
                done = False
                new_cells = set()
                episode_count += 1
                if args.solver:
                    solver_moves = find_solution_for_queue(env._queue.copy())
                last_auto_time = time.time()

        # ── Draw ──────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        draw_title(screen, f"BrainBlock — Episode #{episode_count + 1}")
        draw_board(screen, env._placed, new_cells)
        draw_panel(screen, env, step_count, total_reward, status, mode_label,
                   auto_play, speed)
        draw_footer(screen, f"Seed: {episode_seed} | Queue: {' '.join(env._queue) if env._queue else 'DONE'}")

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


# ── Step helpers ───────────────────────────────────────────────────────

def _do_step(env, agent, action_mask, obs, solver_moves, step_count, args, is_sac=False):
    """Choose and execute one action. Returns new_cells."""
    if args.solver and solver_moves and step_count < len(solver_moves):
        orient, x, y = solver_moves[step_count]
        action = encode_action(orient, x, y)
    elif agent is not None:
        with torch.no_grad():
            grid = torch.tensor(obs["grid"]).unsqueeze(0)
            vec  = torch.tensor(obs["vec"]).unsqueeze(0)
            mask = torch.tensor(action_mask.astype(np.float32)).unsqueeze(0)
            if args.algo == "dqn":
                epsilon = 0.05 if args.stochastic else 0.0
                action = agent.act(
                    {"grid": obs["grid"], "vec": obs["vec"]},
                    action_mask,
                    epsilon=epsilon,
                )
            elif is_sac:
                probs, _ = agent.actor(grid, vec, mask)
                if args.stochastic:
                    action = torch.multinomial(probs, 1).squeeze().item()
                else:
                    action = probs.argmax(dim=-1).item()
            else:
                dist, _ = agent.network(grid, vec, mask)
                if args.stochastic:
                    action = dist.sample().item()
                else:
                    action = dist.probs.argmax(dim=-1).item()
    else:
        # Random legal action
        legal = np.where(action_mask)[0]
        if len(legal) == 0:
            return set()
        action = int(np.random.choice(legal))

    # Remember placed count before step
    placed_before = len(env._placed)

    next_obs, reward, terminated, truncated, next_info = env.step(action)

    # Determine new cells
    new_cells = set()
    if len(env._placed) > placed_before:
        _, cells = env._placed[-1]
        new_cells = cells

    # Store results for caller
    env._last_step_result = (next_obs, reward, terminated, truncated, next_info, new_cells)
    return new_cells


def _post_step(env):
    """Retrieve the stored step result."""
    return env._last_step_result


def _update_state(env, obs, action_mask, step_count, total_reward, new_cells, result):
    """Update state variables after a step."""
    next_obs, reward, terminated, truncated, next_info, new_cells = result
    done = terminated or truncated
    total_reward += reward
    step_count += 1

    reason = next_info.get("termination_reason", "")
    if reason == "success":
        status = "✅ SUCCESS — Board fully solved!"
    elif reason == "dead_end":
        coverage = next_info.get("coverage", 0)
        status = f"❌ DEAD END — Coverage: {coverage:.0%}"
    elif reason == "illegal_action":
        status = "⚠️ ILLEGAL ACTION"
    else:
        status = ""

    if done:
        return next_obs, action_mask, step_count, total_reward, status, done, new_cells
    else:
        return next_obs, next_info["action_mask"], step_count, total_reward, status, done, new_cells


if __name__ == "__main__":
    main()
