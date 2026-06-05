"""
BrainBlock Gymnasium Environment — Member A pipeline.

Uses shared piece definitions from common/pieces.py.
Supports two reward modes:
  - "sparse"  (R1): +1 only on full completion, 0 otherwise.
  - "shaped"  (R2): potential-based shaping  r = r_base + γ·Φ(s') − Φ(s)
                     where Φ(s) = filled_cells / 40.

Action masking is built in: info["action_mask"] is always provided.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common.pieces import (
    PIECE_TYPES,
    PIECE_IDX,
    INVENTORY,
    ORIENT_TABLE,
    VALID_ORIENTS,
    PIECE_COLORS,
    encode_action,
    decode_action,
    is_legal,
    compute_action_mask,
)


def _build_obs(board: np.ndarray, current: str, queue_tail: list[str]) -> dict:
    """
    Build observation dict.
      grid: (1, 5, 8) float32 — binary occupancy
      vec:  (10,) float32 — [one-hot current piece (5), remaining counts / 2 (5)]
    """
    grid = board[np.newaxis, :, :].astype(np.float32)

    onehot = np.zeros(5, dtype=np.float32)
    onehot[PIECE_IDX[current]] = 1.0

    counts = np.zeros(5, dtype=np.float32)
    for p in queue_tail:
        counts[PIECE_IDX[p]] += 1.0
    counts /= 2.0

    vec = np.concatenate([onehot, counts])
    return {"grid": grid, "vec": vec}


class BrainBlockEnv(gym.Env):
    """
    BrainBlock 8×5 tetromino packing puzzle.

    reward_mode: "sparse" → R1, "shaped" → R2 (potential-based, default).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, reward_mode: str = "shaped", render_mode: Optional[str] = None):
        super().__init__()
        assert reward_mode in ("sparse", "shaped"), f"Unknown reward_mode: {reward_mode}"
        self.reward_mode = reward_mode
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, shape=(1, 5, 8), dtype=np.float32),
            "vec": spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(320)

        # State (set in reset)
        self.board: np.ndarray = None
        self._queue: list[str] = None
        self._step_count: int = 0
        self._placed: list[tuple[str, set]] = []

    # ──────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((5, 8), dtype=np.int8)
        queue = INVENTORY.copy()
        self.np_random.shuffle(queue)
        self._queue = queue
        self._step_count = 0
        self._placed = []

        obs = _build_obs(self.board, self._current, self._tail)
        mask = compute_action_mask(self.board, self._current)
        info = {"action_mask": mask}
        return obs, info

    # ──────────────────────────────────────────────────────────────────
    @property
    def _current(self) -> str:
        return self._queue[0]

    @property
    def _tail(self) -> list[str]:
        return self._queue[1:]

    # ──────────────────────────────────────────────────────────────────
    def step(self, action: int):
        orient, x, y = decode_action(action)
        current = self._current

        # --- Legality check ---
        if not is_legal(self.board, current, orient, x, y):
            obs = _build_obs(self.board, current, self._tail)
            mask = np.zeros(320, dtype=np.int8)
            info = {"action_mask": mask, "termination_reason": "illegal_action"}
            return obs, 0.0, True, False, info

        # --- Φ before placement (for shaped reward) ---
        phi_before = self.board.sum() / 40.0

        # --- Apply placement ---
        cells_placed = set()
        for dx, dy in ORIENT_TABLE[current][orient]:
            nx, ny = x + dx, y + dy
            self.board[ny, nx] = 1
            cells_placed.add((nx, ny))
        self._placed.append((current, cells_placed))

        # Advance queue
        self._queue.pop(0)
        self._step_count += 1

        # --- Terminal: success ---
        if len(self._queue) == 0:
            reward = self._final_reward(phi_before, success=True)
            obs = _build_obs(self.board, "I", [])  # dummy; episode over
            mask = np.zeros(320, dtype=np.int8)
            info = {
                "action_mask": mask,
                "termination_reason": "success",
                "pieces_placed": self._step_count,
                "coverage": self.board.sum() / 40.0,
            }
            return obs, reward, True, False, info

        # --- Dead-end check ---
        new_mask = compute_action_mask(self.board, self._current)
        if new_mask.sum() == 0:
            reward = self._step_reward(phi_before)
            obs = _build_obs(self.board, self._current, self._tail)
            info = {
                "action_mask": new_mask,
                "termination_reason": "dead_end",
                "pieces_placed": self._step_count,
                "coverage": self.board.sum() / 40.0,
            }
            return obs, reward, True, False, info

        # --- Normal step ---
        reward = self._step_reward(phi_before)
        obs = _build_obs(self.board, self._current, self._tail)
        info = {
            "action_mask": new_mask,
            "pieces_placed": self._step_count,
            "coverage": self.board.sum() / 40.0,
        }
        return obs, reward, False, False, info

    # ──────────────────────────────────────────────────────────────────
    def _step_reward(self, phi_before: float) -> float:
        if self.reward_mode == "sparse":
            return 0.0
        phi_after = self.board.sum() / 40.0
        gamma = 0.99
        return gamma * phi_after - phi_before

    def _final_reward(self, phi_before: float, success: bool) -> float:
        base = 1.0 if success else 0.0
        if self.reward_mode == "sparse":
            return base
        phi_after = self.board.sum() / 40.0
        gamma = 0.99
        return base + (gamma * phi_after - phi_before)

    # ──────────────────────────────────────────────────────────────────
    def render(self):
        # Import here to avoid requiring matplotlib at training time
        from common.visualize import render_board
        return render_board(
            self._placed,
            title=f"BrainBlock — {len(self._placed)}/10 | next: {self._current if self._queue else 'DONE'}",
            show=(self.render_mode == "human"),
        )

    def close(self):
        pass
