"""
BrainBlock — Pixel-Art Demo GIF generator.

Renders the PPO agent solving the 8x5 tetromino-packing puzzle as a cute
retro pixel-art animation and writes it to a GIF. Shows, per step:
  - a large beveled-block board,
  - an Info panel with episode / step / coverage / reward,
  - the upcoming piece queue (the NEXT incoming piece type),
  - the transformation the agent applied (base shape -> placed shape + label).

Multiple episodes are stitched together, each starting from a different
random piece order.

Usage:
  python pixel_demo.py \
      --model results/ppo_r2_mlp_ent005_div1_seed42/best_model.pt \
      --seeds 4 1 2 0 \
      --out demo.gif
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")

from member_umut.environment import BrainBlockEnv
from member_umut.agent import PPOAgent, PPOConfig
from common.pieces import ORIENT_TABLE, decode_action

# Palette
BG       = (16, 18, 30)
PANEL    = (26, 29, 48)
PANEL_HI = (38, 42, 66)
EMPTY    = (33, 37, 58)
EMPTY_HI = (44, 49, 74)
TEXT     = (228, 233, 245)
DIM      = (122, 132, 168)
ACCENT   = (120, 214, 255)
GOOD     = (129, 220, 150)
BAD      = (240, 110, 120)
GOLD     = (255, 213, 79)

PIECE_COLOR = {
    "I": (79, 195, 247),
    "O": (255, 213, 79),
    "L": (255, 138, 101),
    "Z": (129, 199, 132),
    "T": (206, 147, 216),
}

FONT_PATH = "/System/Library/Fonts/Monaco.ttf"

# Geometry
COLS, ROWS = 8, 5
CELL = 58            # board cell size (px)
GAP = 3              # gap between cells
BOARD_X, BOARD_Y = 34, 96
BOARD_W = COLS * CELL + (COLS - 1) * GAP
BOARD_H = ROWS * CELL + (ROWS - 1) * GAP

PANEL_X = BOARD_X + BOARD_W + 34
PANEL_W = 360
CANVAS_W = PANEL_X + PANEL_W + 30
CANVAS_H = BOARD_Y + BOARD_H + 34


# Color helpers
def lighten(c, f):
    return tuple(min(255, int(v + (255 - v) * f)) for v in c)


def darken(c, f):
    return tuple(max(0, int(v * (1 - f))) for v in c)


# Text (chunky 1-bit pixel font from a TTF)
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
    return _font_cache[size]


def _text_mask(text: str, base: int, scale: int) -> Image.Image:
    """Render text, threshold to 1-bit, nearest-upscale -> chunky pixels."""
    font = _font(base)
    bbox = font.getbbox(text)
    w = max(1, bbox[2] - bbox[0])
    h = max(1, bbox[3] - bbox[1])
    layer = Image.new("L", (w + 2, h + 2), 0)
    ImageDraw.Draw(layer).text((1 - bbox[0], 1 - bbox[1]), text, fill=255, font=font)
    layer = layer.point(lambda p: 255 if p >= 110 else 0)
    if scale != 1:
        layer = layer.resize((layer.width * scale, layer.height * scale), Image.NEAREST)
    return layer


def text_size(text: str, base: int, scale: int) -> tuple[int, int]:
    m = _text_mask(text, base, scale)
    return m.width, m.height


def draw_text(canvas, pos, text, base, scale, color, anchor="lt"):
    mask = _text_mask(text, base, scale)
    x, y = pos
    if "r" in anchor:
        x -= mask.width
    elif "m" in anchor:
        x -= mask.width // 2
    if "b" in anchor:
        y -= mask.height
    elif "c" in anchor:
        y -= mask.height // 2
    canvas.paste(Image.new("RGB", mask.size, color), (x, y), mask)
    return mask.width, mask.height


# Block drawing
def draw_block(d, x0, y0, s, color, bevel=None, flash=0.0):
    if bevel is None:
        bevel = max(2, s // 9)
    if flash:
        color = lighten(color, flash)
    light = lighten(color, 0.5)
    dark = darken(color, 0.42)
    d.rectangle([x0, y0, x0 + s - 1, y0 + s - 1], fill=color)
    d.rectangle([x0, y0, x0 + s - 1, y0 + bevel - 1], fill=light)            # top
    d.rectangle([x0, y0, x0 + bevel - 1, y0 + s - 1], fill=light)            # left
    d.rectangle([x0, y0 + s - bevel, x0 + s - 1, y0 + s - 1], fill=dark)     # bottom
    d.rectangle([x0 + s - bevel, y0, x0 + s - 1, y0 + s - 1], fill=dark)     # right


def draw_arrow(d, x0, y0, h, color):
    """Draw a small chunky right-pointing pixel arrow; return its width."""
    px = max(2, h // 7)
    cy = y0 + h // 2
    shaft_w = px * 4
    d.rectangle([x0, cy - px, x0 + shaft_w, cy + px - 1], fill=color)
    # arrow head (stepped triangle)
    head_x = x0 + shaft_w
    steps = 3
    for i in range(steps):
        half = (steps - i) * px
        d.rectangle([head_x + i * px, cy - half, head_x + (i + 1) * px - 1, cy + half - 1],
                    fill=color)
    return shaft_w + steps * px


def draw_empty(d, x0, y0, s):
    d.rectangle([x0, y0, x0 + s - 1, y0 + s - 1], fill=EMPTY)
    inset = max(2, s // 9)
    d.rectangle([x0 + inset, y0 + inset, x0 + s - 1 - inset, y0 + s - 1 - inset],
                fill=darken(EMPTY, 0.18))


def draw_mini_piece(canvas, x0, y0, cells, color, msize, max_cells_w=4, max_cells_h=2):
    """Draw a small beveled rendering of a piece shape, centered in a box."""
    xs = [dx for dx, dy in cells]
    ys = [dy for dx, dy in cells]
    w = max(xs) - min(xs) + 1
    h = max(ys) - min(ys) + 1
    minx, miny = min(xs), min(ys)
    box_w = max_cells_w * msize
    box_h = max_cells_h * msize
    ox = x0 + (box_w - w * msize) // 2
    oy = y0 + (box_h - h * msize) // 2
    d = ImageDraw.Draw(canvas)
    g = max(1, msize // 14)
    for dx, dy in cells:
        cx = ox + (dx - minx) * msize
        # flip y so the shape reads top-down like the board (y up -> draw up)
        cy = oy + (h - 1 - (dy - miny)) * msize
        draw_block(d, cx + g, cy + g, msize - 2 * g, color)
    return box_w, box_h


# Transformation label
def transform_label(piece: str, orient: int) -> str:
    if piece == "O":
        return "PLACED"
    rot = (orient % 4) * 90
    flip = orient >= 4
    if not flip and rot == 0:
        return "AS-IS"
    parts = []
    if flip:
        parts.append("FLIP")
    if rot:
        parts.append(f"ROT {rot}°")
    return " + ".join(parts) if parts else "AS-IS"


# Frame rendering
def render_frame(state) -> Image.Image:
    """state: dict with board info to draw a single frame."""
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    d = ImageDraw.Draw(img)

    # subtle vignette panel behind board
    d.rectangle([BOARD_X - 14, BOARD_Y - 14, BOARD_X + BOARD_W + 13, BOARD_Y + BOARD_H + 13],
                fill=PANEL)

    # Title
    draw_text(img, (BOARD_X - 14, 26), "BRAINBLOCK", 22, 2, ACCENT)
    draw_text(img, (BOARD_X - 14, 70), "8x5 TETROMINO PACKING - PPO AGENT", 9, 1, DIM)

    # Board
    cell_map = {}
    for pt, cells in state["placed"]:
        for cx, cy in cells:
            cell_map[(cx, cy)] = pt
    new_cells = state.get("new_cells", set())
    flash = state.get("flash", 0.0)
    for row in range(ROWS):
        for col in range(COLS):
            bx = BOARD_X + col * (CELL + GAP)
            by = BOARD_Y + (ROWS - 1 - row) * (CELL + GAP)
            pt = cell_map.get((col, row))
            if pt is None:
                draw_empty(d, bx, by, CELL)
            else:
                f = flash if (col, row) in new_cells else 0.0
                draw_block(d, bx, by, CELL, PIECE_COLOR[pt], flash=f)

    # Info panel
    px = PANEL_X
    pw = PANEL_W
    d.rectangle([px, BOARD_Y - 14, px + pw, BOARD_Y + BOARD_H + 13], fill=PANEL)
    pad = 20
    ix = px + pad
    iy = BOARD_Y - 14 + 16
    inner_w = pw - 2 * pad

    # Episode / Step
    draw_text(img, (ix, iy), "EPISODE", 9, 1, DIM)
    draw_text(img, (ix + inner_w, iy), f"#{state['episode']}", 13, 1, TEXT, anchor="rt")
    iy += 26
    draw_text(img, (ix, iy), "STEP", 9, 1, DIM)
    draw_text(img, (ix + inner_w, iy), f"{state['step']:>2}/10", 13, 1, TEXT, anchor="rt")
    iy += 30

    # Coverage bar
    draw_text(img, (ix, iy), "COVERAGE", 9, 1, DIM)
    cov = state["coverage"]
    draw_text(img, (ix + inner_w, iy), f"{cov:.0%}", 11, 1,
              GOOD if cov >= 1.0 else TEXT, anchor="rt")
    iy += 18
    bar_h = 14
    d.rectangle([ix, iy, ix + inner_w, iy + bar_h], fill=darken(EMPTY, 0.25))
    fillw = int(inner_w * cov)
    if fillw > 0:
        bc = GOOD if cov >= 1.0 else ACCENT
        d.rectangle([ix, iy, ix + fillw, iy + bar_h], fill=bc)
        d.rectangle([ix, iy, ix + fillw, iy + 3], fill=lighten(bc, 0.4))
    iy += bar_h + 16

    # Reward
    draw_text(img, (ix, iy), "REWARD", 9, 1, DIM)
    rw = state["reward"]
    rc = GOOD if rw > 0 else (BAD if rw < 0 else TEXT)
    draw_text(img, (ix + inner_w, iy), f"{rw:+.3f}", 12, 1, rc, anchor="rt")
    iy += 28

    # divider
    d.rectangle([ix, iy, ix + inner_w, iy + 1], fill=PANEL_HI)
    iy += 12

    # Queue (upcoming piece types) — NEXT incoming piece highlighted
    draw_text(img, (ix, iy), "QUEUE", 9, 1, DIM)
    nxt = state["queue"][0] if state["queue"] else None
    if nxt:
        draw_text(img, (ix + inner_w, iy), f"NEXT [{nxt}]", 10, 1,
                  PIECE_COLOR[nxt], anchor="rt")
    iy += 18
    qsize = 16
    qx = ix
    for i, p in enumerate(state["queue"][:10]):
        col = PIECE_COLOR[p]
        if i == 0:
            d.rectangle([qx - 3, iy - 3, qx + qsize + 2, iy + qsize + 2],
                        outline=ACCENT, width=2)
        draw_block(d, qx, iy, qsize, col)
        draw_text(img, (qx + qsize // 2, iy + qsize + 4), p, 8, 1,
                  DIM if i else TEXT, anchor="mt")
        qx += qsize + 16
    iy += qsize + 22

    # divider
    d.rectangle([ix, iy, ix + inner_w, iy + 1], fill=PANEL_HI)
    iy += 12

    # Last move / transformation
    draw_text(img, (ix, iy), "AGENT MOVE", 9, 1, DIM)
    iy += 20
    mv = state.get("move")
    if mv is not None:
        piece, orient = mv
        col = PIECE_COLOR[piece]
        base = sorted(ORIENT_TABLE[piece][0])
        placed = sorted(ORIENT_TABLE[piece][orient])
        msize = 18
        # base shape
        bw, bh = draw_mini_piece(img, ix, iy, base, darken(col, 0.25), msize)
        ax = ix + bw + 12
        arrow_w = draw_arrow(d, ax, iy, bh, DIM)
        # placed shape
        pwid, _ = draw_mini_piece(img, ax + arrow_w + 12, iy, placed, col, msize)
        lx = ax + arrow_w + 12 + pwid + 16
        draw_text(img, (ix + 2, iy + bh + 8), f"[{piece}]", 11, 1, col)
        draw_text(img, (lx, iy + bh // 2), transform_label(piece, orient),
                  11, 1, TEXT, anchor="lc")
    else:
        draw_text(img, (ix, iy + 6), "...", 12, 1, DIM)

    # Status line (bottom of panel)
    status = state.get("status", "")
    sy = BOARD_Y + BOARD_H + 13 - 30
    if status:
        if "SOLVED" in status:
            draw_text(img, (px + pw // 2, sy), "* " + status + " *", 13, 1, GOOD, anchor="mt")
        elif "DEAD" in status:
            draw_text(img, (px + pw // 2, sy), status, 13, 1, BAD, anchor="mt")
        else:
            draw_text(img, (px + pw // 2, sy), status, 11, 1, DIM, anchor="mt")

    return img


# Agent
def load_agent(model_path, encoder="mlp", hidden_dim=256):
    cfg = PPOConfig(encoder_type=encoder, hidden_dim=hidden_dim)
    agent = PPOAgent(cfg, device=torch.device("cpu"))
    agent.load(model_path)
    agent.network.eval()
    return agent


def choose_action(agent, obs, mask):
    with torch.no_grad():
        g = torch.tensor(obs["grid"]).unsqueeze(0)
        v = torch.tensor(obs["vec"]).unsqueeze(0)
        m = torch.tensor(mask.astype(np.float32)).unsqueeze(0)
        dist, _ = agent.network(g, v, m)
        return dist.probs.argmax(-1).item()


# Episode -> frames
def episode_frames(agent, env, seed, episode_idx):
    """Return list of (PIL.Image, duration_ms)."""
    frames = []
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]

    def base_state(**kw):
        s = {
            "placed": list(env._placed),
            "queue": list(env._queue),
            "episode": episode_idx,
            "step": env._step_count,
            "coverage": float(env.board.sum()) / 40.0,
            "reward": kw.get("reward", 0.0),
            "move": kw.get("move"),
            "status": kw.get("status", ""),
            "new_cells": kw.get("new_cells", set()),
            "flash": kw.get("flash", 0.0),
        }
        return s

    total_reward = 0.0
    last_move = None
    # opening frame
    frames.append((render_frame(base_state(reward=total_reward)), 650))

    done = False
    while not done:
        action = choose_action(agent, obs, mask)
        orient, x, y = decode_action(action)
        piece = env._current
        before = len(env._placed)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        total_reward += r
        mask = info.get("action_mask", mask)

        new_cells = set()
        if len(env._placed) > before:
            _, new_cells = env._placed[-1]
        last_move = (piece, orient)

        reason = info.get("termination_reason", "")
        status = ""
        if reason == "success":
            status = "BOARD SOLVED"
        elif reason == "dead_end":
            status = f"DEAD END {info.get('coverage', 0):.0%}"
        elif reason == "illegal_action":
            status = "ILLEGAL"

        # flash frame (just-placed cells bright)
        frames.append((render_frame(base_state(
            reward=total_reward, move=last_move, status=status,
            new_cells=new_cells, flash=0.55)), 120))
        # settle frame
        frames.append((render_frame(base_state(
            reward=total_reward, move=last_move, status=status,
            new_cells=new_cells, flash=0.0)), 300))

    # hold final solved frame
    frames[-1] = (frames[-1][0], 1500)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/ppo_r2_mlp_ent005_div1_seed42/best_model.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[4, 1, 2, 0])
    ap.add_argument("--out", default="demo.gif")
    ap.add_argument("--test-frame", default=None,
                    help="If set, render a single mid-episode PNG here and exit.")
    args = ap.parse_args()

    agent = load_agent(args.model)
    env = BrainBlockEnv(reward_mode="shaped")

    if args.test_frame:
        frames = episode_frames(agent, env, args.seeds[0], 1)
        frames[len(frames) // 2][0].save(args.test_frame)
        print("wrote", args.test_frame)
        return

    all_frames = []
    for i, seed in enumerate(args.seeds, 1):
        all_frames.extend(episode_frames(agent, env, seed, i))

    imgs = [f for f, _ in all_frames]
    durs = [d for _, d in all_frames]

    # Shared palette to avoid inter-frame flicker.
    master = imgs[len(imgs) // 2].quantize(colors=64, method=Image.MEDIANCUT)
    pal_imgs = [im.quantize(palette=master, dither=Image.NONE) for im in imgs]

    pal_imgs[0].save(
        args.out, save_all=True, append_images=pal_imgs[1:],
        duration=durs, loop=0, optimize=True, disposal=2,
    )
    print(f"wrote {args.out}  ({len(pal_imgs)} frames, {CANVAS_W}x{CANVAS_H})")


if __name__ == "__main__":
    main()
