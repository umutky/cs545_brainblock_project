"""
Shared piece definitions for BrainBlock DRL project.

Both Member A (PPO) and Member B (DQN) import from here to guarantee
identical piece geometry across pipelines.

Locked specs (§2 of ppo_action_plan):
  - Coordinate: board[y, x], x ∈ {0..7}, y ∈ {0..4}
  - Action: a = orient*40 + x*5 + y  (|A|=320)
  - Piece type index order: I=0, O=1, L=2, Z=3, T=4
  - D4 group: 8 transforms per piece, dedup via frozenset
"""

from __future__ import annotations

from itertools import product
from typing import Optional

import numpy as np

# ── Piece types (canonical order) ──────────────────────────────────────

PIECE_TYPES = ["I", "O", "L", "Z", "T"]
PIECE_IDX = {p: i for i, p in enumerate(PIECE_TYPES)}

# ── Base offsets (reference orientation, min dx=0, min dy=0) ───────────

BASE_OFFSETS: dict[str, list[tuple[int, int]]] = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "L": [(0, 0), (1, 0), (2, 0), (2, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
}

# ── Inventory ──────────────────────────────────────────────────────────

INVENTORY = ["I", "I", "O", "O", "L", "L", "Z", "Z", "T", "T"]

# ── Colors for rendering ──────────────────────────────────────────────

PIECE_COLORS = {
    "I": "#4FC3F7",
    "O": "#FFD54F",
    "L": "#FF8A65",
    "Z": "#81C784",
    "T": "#CE93D8",
    None: "#ECEFF1",
}

# ── D4 orientation generation ─────────────────────────────────────────

def _normalize(offsets: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    """Translate so min dx=0, min dy=0; return as frozenset."""
    min_x = min(dx for dx, dy in offsets)
    min_y = min(dy for dx, dy in offsets)
    return frozenset((dx - min_x, dy - min_y) for dx, dy in offsets)


def _rotate90(offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """90° CCW rotation: (dx, dy) → (-dy, dx)."""
    return [(-dy, dx) for dx, dy in offsets]


def _reflect(offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Horizontal reflection: (dx, dy) → (-dx, dy)."""
    return [(-dx, dy) for dx, dy in offsets]


def _build_orientations(
    base: list[tuple[int, int]],
    include_reflection: bool = True,
) -> list[Optional[frozenset]]:
    """
    Generate up to 8 D4 transforms; deduplicate; return list of length 8.
    Slots with duplicate or excluded shapes are set to None (masked).
    """
    seen: list[frozenset] = []
    all8: list[Optional[frozenset]] = []

    current = base
    for _ in range(4):
        norm = _normalize(current)
        if norm not in seen:
            seen.append(norm)
        all8.append(norm)
        current = _rotate90(current)

    reflected = _reflect(base)
    for _ in range(4):
        norm = _normalize(reflected)
        if include_reflection and norm not in seen:
            seen.append(norm)
        all8.append(norm)
        reflected = _rotate90(reflected)

    result: list[Optional[frozenset]] = []
    assigned: set[frozenset] = set()
    for i, fs in enumerate(all8):
        is_reflection_slot = i >= 4
        if is_reflection_slot and not include_reflection:
            result.append(None)
        elif fs not in assigned:
            result.append(fs)
            assigned.add(fs)
        else:
            result.append(None)

    return result


# Per-piece reflection policy:
#   L: reflection included — physical piece can be flipped (gives J orientations)
#   Z: reflection included — physical piece can be flipped (gives S orientations)
_INCLUDE_REFLECTION = {"I": True, "O": True, "L": True, "Z": True, "T": True}

# Pre-compute orientation tables at import time
# ORIENT_TABLE[piece_type][orient_idx] = frozenset of (dx,dy) or None
ORIENT_TABLE: dict[str, list[Optional[frozenset]]] = {
    p: _build_orientations(BASE_OFFSETS[p], _INCLUDE_REFLECTION[p])
    for p in PIECE_TYPES
}

# Sorted list of valid orient indices per piece
VALID_ORIENTS: dict[str, list[int]] = {
    p: [i for i, v in enumerate(ORIENT_TABLE[p]) if v is not None]
    for p in PIECE_TYPES
}


# ── Action encode / decode ────────────────────────────────────────────

def encode_action(orient: int, x: int, y: int) -> int:
    return orient * 40 + x * 5 + y


def decode_action(a: int) -> tuple[int, int, int]:
    orient = a // 40
    r = a % 40
    x = r // 5
    y = r % 5
    return orient, x, y


# ── Legality check ────────────────────────────────────────────────────

def is_legal(board: np.ndarray, piece: str, orient: int, x: int, y: int) -> bool:
    """Return True iff placing `piece` with `orient` at anchor (x,y) is legal."""
    cells = ORIENT_TABLE[piece][orient]
    if cells is None:
        return False
    for dx, dy in cells:
        nx, ny = x + dx, y + dy
        if nx < 0 or nx >= 8 or ny < 0 or ny >= 5:
            return False
        if board[ny, nx] != 0:
            return False
    return True


# ── Action mask ────────────────────────────────────────────────────────

def compute_action_mask(board: np.ndarray, piece: str) -> np.ndarray:
    """Return int8 array of shape (320,) with 1 for every legal action."""
    mask = np.zeros(320, dtype=np.int8)
    for orient in VALID_ORIENTS[piece]:
        cells = ORIENT_TABLE[piece][orient]
        for x, y in product(range(8), range(5)):
            if all(
                0 <= x + dx < 8 and 0 <= y + dy < 5 and board[y + dy, x + dx] == 0
                for dx, dy in cells
            ):
                mask[encode_action(orient, x, y)] = 1
    return mask
