"""
BrainBlock — Human Interactive Play

Controls:
  Mouse hover          — piece placement preview
  Scroll / ← →        — cycle orientations
  Left click           — place piece (only valid positions)
  R                    — new game
  Q / ESC              — quit

Run:
  python -m member_umut.play
  python -m member_umut.play --seed 7
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np
import pygame

from member_umut.environment import BrainBlockEnv
from common.pieces import (
    PIECE_TYPES, PIECE_COLORS, ORIENT_TABLE, VALID_ORIENTS,
    encode_action, is_legal,
)

# ── Colors ───────────────────────────────────────────────────────────────

def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

BG         = (18,  18,  24)
GRID_BG    = (30,  32,  44)
GRID_LINE  = (50,  55,  75)
TEXT       = (220, 225, 240)
DIM        = (120, 130, 155)
ACCENT     = (100, 180, 255)
OK_COL     = (100, 220, 130)
BAD_COL    = (255, 100, 100)
WARN_COL   = (255, 200,  80)
PANEL_BG   = (25,  27,  38)

PIECE_RGB = {
    "I": (79,  195, 247),
    "O": (255, 213,  79),
    "L": (255, 138, 101),
    "Z": (129, 199, 132),
    "T": (206, 147, 216),
    None: (45,  50,  65),
}

# ── Layout ───────────────────────────────────────────────────────────────

CELL  = 80
BW, BH = 8, 5
BPW   = BW * CELL
BPH   = BH * CELL

MAR   = 40
PANW  = 300
WW    = MAR + BPW + MAR + PANW + MAR
WH    = MAR + 50 + BPH + MAR + 50

BX    = MAR
BY    = MAR + 50

FPS   = 60


# ── Helpers ──────────────────────────────────────────────────────────────

def mouse_to_grid(mx: int, my: int) -> tuple[int, int]:
    gx = (mx - BX) // CELL
    gy = BH - 1 - (my - BY) // CELL
    return gx, gy


def on_board(mx: int, my: int) -> bool:
    return BX <= mx < BX + BPW and BY <= my < BY + BPH


def _cell_rect(gx: int, gy: int):
    return pygame.Rect(
        BX + gx * CELL + 2,
        BY + (BH - 1 - gy) * CELL + 2,
        CELL - 4,
        CELL - 4,
    )


# ── Draw board ───────────────────────────────────────────────────────────

def draw_board(surf, env):
    # Shadow
    sh = pygame.Surface((BPW, BPH), pygame.SRCALPHA)
    sh.fill((0, 0, 0, 28))
    surf.blit(sh, (BX + 4, BY + 4))

    pygame.draw.rect(surf, GRID_BG, (BX, BY, BPW, BPH), border_radius=10)

    cell_map = {}
    for pt, cells in env._placed:
        for cx, cy in cells:
            cell_map[(cx, cy)] = pt

    for gx in range(BW):
        for gy in range(BH):
            pt = cell_map.get((gx, gy))
            color = PIECE_RGB.get(pt, PIECE_RGB[None])
            r = _cell_rect(gx, gy)
            pygame.draw.rect(surf, color, r, border_radius=6)
            # Shine
            shine = pygame.Surface((r.w // 2, r.h // 4), pygame.SRCALPHA)
            shine.fill((255, 255, 255, 20))
            surf.blit(shine, (r.x + 4, r.y + 4))

    # Grid lines
    for i in range(BW + 1):
        x = BX + i * CELL
        pygame.draw.line(surf, GRID_LINE, (x, BY), (x, BY + BPH), 1)
    for j in range(BH + 1):
        y = BY + j * CELL
        pygame.draw.line(surf, GRID_LINE, (BX, y), (BX + BPW, y), 1)

    # Coordinate hints
    f = pygame.font.SysFont("Menlo", 11)
    for gx in range(BW):
        lbl = f.render(str(gx), True, DIM)
        surf.blit(lbl, (BX + gx * CELL + CELL // 2 - 4, BY + BPH + 4))


def draw_preview(surf, env, gx: int, gy: int, piece: str, orient: int):
    offsets = ORIENT_TABLE[piece][orient]
    if offsets is None:
        return

    valid = is_legal(env.board, piece, orient, gx, gy)
    border = OK_COL if valid else BAD_COL
    base_alpha = 150 if valid else 80
    color = PIECE_RGB[piece]

    for dx, dy in offsets:
        cx, cy = gx + dx, gy + dy
        if not (0 <= cx < BW and 0 <= cy < BH):
            continue
        r = _cell_rect(cx, cy)
        s = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        occupied = env.board[cy, cx] != 0
        fill_col = BAD_COL if occupied else color
        a = base_alpha if not occupied else 180
        s.fill((*fill_col, a))
        pygame.draw.rect(s, (*border, 220), (0, 0, r.w, r.h), width=3, border_radius=6)
        surf.blit(s, r.topleft)


# ── Mini piece shape ─────────────────────────────────────────────────────

def draw_piece_shape(surf, cx: int, cy: int, piece: str, orient: int, sz: int = 20):
    offsets = ORIENT_TABLE[piece][orient]
    if not offsets:
        return
    xs = [dx for dx, dy in offsets]
    ys = [dy for dx, dy in offsets]
    w = (max(xs) - min(xs) + 1) * sz
    h = (max(ys) - min(ys) + 1) * sz
    ox = cx - w // 2 - min(xs) * sz
    oy = cy + h // 2 + min(ys) * sz
    color = PIECE_RGB[piece]
    for dx, dy in offsets:
        px = ox + dx * sz + 2
        py = oy - dy * sz + 2
        pygame.draw.rect(surf, color, (px, py, sz - 4, sz - 4), border_radius=3)
        pygame.draw.rect(surf, (255, 255, 255), (px, py, sz - 4, sz - 4), width=1, border_radius=3)


# ── Panel ────────────────────────────────────────────────────────────────

def draw_panel(surf, env, piece, orient_idx, valid_orients, action_mask, status, done):
    px = BX + BPW + MAR
    py = BY

    pygame.draw.rect(surf, PANEL_BG, (px, py, PANW, BPH), border_radius=10)

    FL  = pygame.font.SysFont("Menlo", 20, bold=True)
    FM  = pygame.font.SysFont("Menlo", 15, bold=True)
    F   = pygame.font.SysFont("Menlo", 14)
    FS  = pygame.font.SysFont("Menlo", 12)

    pad = 16
    yo  = py + 16
    lh  = 22

    def blit(text, color, font=F):
        nonlocal yo
        surf.blit(font.render(text, True, color), (px + pad, yo))
        yo += lh

    def divider():
        nonlocal yo
        yo += 4
        pygame.draw.line(surf, GRID_LINE, (px + pad, yo), (px + PANW - pad, yo), 1)
        yo += 8

    # ── Current piece ────────────────────────────────────────────
    blit("Place piece:", DIM, FS)
    surf.blit(FL.render(f"  [ {piece} ]", True, PIECE_RGB[piece]), (px + pad, yo))
    yo += 28

    # ── Orientation ──────────────────────────────────────────────
    orient = valid_orients[orient_idx]
    blit(f"Orient: {orient_idx + 1} / {len(valid_orients)}", TEXT)

    # Mini shape preview
    draw_piece_shape(surf, px + PANW // 2, yo + 40, piece, orient, sz=22)
    yo += 70

    divider()

    # ── Stats ────────────────────────────────────────────────────
    n_valid = int(action_mask.sum())
    va_col = OK_COL if n_valid > 10 else WARN_COL if n_valid > 0 else BAD_COL
    blit(f"Valid placements: {n_valid}", va_col)

    cov = env.board.sum() / 40.0
    blit(f"Coverage: {cov:.0%}", TEXT)

    # Coverage bar
    bar_w = PANW - pad * 2
    pygame.draw.rect(surf, GRID_BG, (px + pad, yo, bar_w, 10), border_radius=4)
    fill = int(bar_w * cov)
    if fill > 0:
        pygame.draw.rect(surf, OK_COL if cov >= 1.0 else ACCENT,
                         (px + pad, yo, fill, 10), border_radius=4)
    yo += 18

    blit(f"Placed: {len(env._placed)} / 10", TEXT)

    # Remaining queue
    if env._queue and len(env._queue) > 1:
        yo += 4
        blit("Remaining:", DIM, FS)
        counts = Counter(env._queue[1:])
        for pt in PIECE_TYPES:
            c = counts.get(pt, 0)
            col = PIECE_RGB[pt] if c > 0 else DIM
            surf.blit(FS.render(f"  {pt}: {'■ ' * c}", True, col), (px + pad, yo))
            yo += 16
    yo += 6

    divider()

    # ── Status ───────────────────────────────────────────────────
    if status:
        s_col = OK_COL if "SUCCESS" in status else BAD_COL if "DEAD" in status else WARN_COL
        surf.blit(FM.render(status, True, s_col), (px + pad, yo))
        yo += lh + 4
        if done:
            blit("R or click to restart", DIM, FS)
            yo += 4

    # ── Controls ─────────────────────────────────────────────────
    for line in ["← → / Scroll   orient", "Left click      place", "R               reset", "Q / ESC         quit"]:
        surf.blit(FS.render(line, True, DIM), (px + pad, yo))
        yo += 16


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WW, WH))
    pygame.display.set_caption("BrainBlock — Human Play")
    clock = pygame.time.Clock()

    seed = args.seed if args.seed is not None else int(np.random.randint(0, 99999))
    env = BrainBlockEnv(reward_mode="shaped")
    obs, info = env.reset(seed=seed)
    action_mask = info["action_mask"]

    ep     = 1
    done   = False
    status = ""
    piece  = env._current
    vorients = VALID_ORIENTS[piece]
    oi     = 0   # orientation index

    FT = pygame.font.SysFont("Menlo", 22, bold=True)
    FS = pygame.font.SysFont("Menlo", 12)

    def reset():
        nonlocal obs, info, action_mask, done, status, piece, vorients, oi, ep, seed
        seed = args.seed if args.seed is not None else int(np.random.randint(0, 99999))
        obs, info = env.reset(seed=seed)
        action_mask = info["action_mask"]
        done = False; status = ""
        piece = env._current; vorients = VALID_ORIENTS[piece]; oi = 0
        ep += 1

    running = True
    while running:
        mx, my = pygame.mouse.get_pos()
        hover = on_board(mx, my)
        gx, gy = mouse_to_grid(mx, my) if hover else (-1, -1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif k == pygame.K_r:
                    reset()
                elif not done:
                    if k in (pygame.K_LEFT, pygame.K_UP):
                        oi = (oi - 1) % len(vorients)
                    elif k in (pygame.K_RIGHT, pygame.K_DOWN):
                        oi = (oi + 1) % len(vorients)

            elif event.type == pygame.MOUSEWHEEL and not done:
                oi = (oi + event.y) % len(vorients)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if done:
                    reset()
                elif hover and gx >= 0:
                    orient = vorients[oi]
                    if is_legal(env.board, piece, orient, gx, gy):
                        action = encode_action(orient, gx, gy)
                        _, _, terminated, truncated, next_info = env.step(action)
                        action_mask = next_info.get("action_mask", np.zeros(320, np.int8))
                        done = terminated or truncated
                        reason = next_info.get("termination_reason", "")
                        if reason == "success":
                            cov = next_info.get("coverage", 1.0)
                            status = f"SUCCESS!  {cov:.0%} covered"
                        elif reason == "dead_end":
                            cov = next_info.get("coverage", 0)
                            status = f"DEAD END  {cov:.0%} covered"
                        if not done:
                            piece = env._current
                            vorients = VALID_ORIENTS[piece]
                            oi = 0

        # ── Draw ─────────────────────────────────────────────────
        screen.fill(BG)

        title = FT.render(f"BrainBlock  —  Game #{ep}   seed: {seed}", True, TEXT)
        screen.blit(title, (MAR, 14))

        draw_board(screen, env)

        if not done and hover and gx >= 0:
            draw_preview(screen, env, gx, gy, piece, vorients[oi])

        draw_panel(screen, env, piece, oi, vorients, action_mask, status, done)

        queue_str = "  →  ".join(env._queue) if env._queue else "DONE"
        foot = FS.render(f"Queue: {queue_str}", True, DIM)
        screen.blit(foot, (MAR, WH - 32))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
