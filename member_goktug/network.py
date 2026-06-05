"""
Q-Network for BrainBlock DQN (Member B2).

Two encoder variants (matching Member A's architecture options):
  - "mlp":     flatten grid (40) + vec (10) → MLP → 320 Q-values
  - "cnn_mlp": CNN on grid (1×5×8) → concat vec → MLP → 320 Q-values
"""

import torch
import torch.nn as nn


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


class QNetwork(nn.Module):
    """
    DQN Q-Network with selectable encoder.

    Takes board grid (B, 1, 5, 8) and inventory vector (B, 10),
    outputs 320 Q-values (one per action).
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

        # Q-value head: encoder features → 320 Q-values
        self.q_head = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 320),
        )

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: (B, 1, 5, 8) float32
            vec:  (B, 10) float32
        Returns:
            q_values: (B, 320) float32
        """
        features = self.encoder(grid, vec)
        return self.q_head(features)
