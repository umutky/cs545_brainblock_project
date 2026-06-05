"""
Actor-Critic Networks for BrainBlock PPO.

Two encoder variants:
  - "mlp":     flatten grid + vec → MLP
  - "cnn_mlp": CNN on grid, concat with vec → MLP

Both share the same ActorCritic wrapper with:
  - Actor head: 320 logits (action masking applied before softmax)
  - Critic head: scalar value
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MLPEncoder(nn.Module):
    """Flatten grid (1×5×8=40) + vec (10) → 50-dim → hidden layers."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        input_dim = 1 * 5 * 8 + 10  # 50
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        flat_grid = grid.reshape(grid.size(0), -1)  # (B, 40)
        x = torch.cat([flat_grid, vec], dim=-1)      # (B, 50)
        return self.net(x)


class CNNMLPEncoder(nn.Module):
    """
    Conv layers on grid (1×5×8) → flatten → concat vec → MLP.
    Small board → small kernels with padding to preserve spatial dims.
    """

    def __init__(self, cnn_channels: list[int] = None, hidden_dim: int = 256):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64]

        layers = []
        in_ch = 1
        for out_ch in cnn_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # After CNN: (B, 64, 5, 8) → flatten → 64*5*8 = 2560
        cnn_flat_dim = cnn_channels[-1] * 5 * 8

        self.fc = nn.Sequential(
            nn.Linear(cnn_flat_dim + 10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(grid)                     # (B, C, 5, 8)
        flat = cnn_out.reshape(cnn_out.size(0), -1)  # (B, C*5*8)
        x = torch.cat([flat, vec], dim=-1)            # (B, C*5*8 + 10)
        return self.fc(x)


class ActorCritic(nn.Module):
    """
    Actor-Critic with action masking.

    The actor outputs 320 logits. Invalid actions are masked by setting
    their logits to -1e9 before softmax.
    """

    def __init__(self, encoder_type: str = "mlp", hidden_dim: int = 256,
                 cnn_channels: list[int] = None):
        super().__init__()

        if encoder_type == "mlp":
            self.encoder = MLPEncoder(hidden_dim)
        elif encoder_type == "cnn_mlp":
            self.encoder = CNNMLPEncoder(cnn_channels, hidden_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        enc_dim = self.encoder.output_dim

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 320),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, grid: torch.Tensor, vec: torch.Tensor,
                action_mask: torch.Tensor):
        """
        Args:
            grid: (B, 1, 5, 8) float32
            vec:  (B, 10) float32
            action_mask: (B, 320) float32  (1=legal, 0=illegal)

        Returns:
            dist: Categorical distribution over actions (masked)
            value: (B,) scalar values
        """
        features = self.encoder(grid, vec)

        # Actor
        logits = self.actor(features)                     # (B, 320)
        # Mask illegal actions
        logits = logits + (action_mask - 1.0) * 1e9       # illegal → -1e9
        dist = Categorical(logits=logits)

        # Critic
        value = self.critic(features).squeeze(-1)         # (B,)

        return dist, value

    def get_action_and_value(self, grid, vec, action_mask, action=None):
        """
        Sample an action (or evaluate a given action) and return:
          action, log_prob, entropy, value
        """
        dist, value = self.forward(grid, vec, action_mask)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, grid, vec, action_mask):
        """Return value estimate only (used for GAE bootstrap)."""
        features = self.encoder(grid, vec)
        return self.critic(features).squeeze(-1)
