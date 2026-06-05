"""
DQN Agent for BrainBlock (Member B2).

Improvements over member_goktug baseline:
  - Preallocated numpy replay buffer (500K capacity, no Python object overhead)
  - Selectable encoder type (mlp / cnn_mlp)
  - Timestep-based epsilon decay
  - Double DQN with masked Q-values
"""

from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from member_goktug.network import QNetwork


# ---------------------------------------------------------------------------
# Preallocated Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Preallocated circular replay buffer for off-policy training."""

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        # Preallocate numpy arrays
        self.grid = np.zeros((capacity, 1, 5, 8), dtype=np.float32)
        self.vec = np.zeros((capacity, 10), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_grid = np.zeros((capacity, 1, 5, 8), dtype=np.float32)
        self.next_vec = np.zeros((capacity, 10), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.action_mask = np.zeros((capacity, 320), dtype=np.float32)
        self.next_action_mask = np.zeros((capacity, 320), dtype=np.float32)

    def add(self, obs: dict, action: int, reward: float,
            next_obs: dict, done: bool,
            action_mask: np.ndarray, next_action_mask: np.ndarray):
        i = self.pos
        self.grid[i] = obs["grid"]
        self.vec[i] = obs["vec"]
        self.action[i] = action
        self.reward[i] = reward
        self.next_grid[i] = next_obs["grid"]
        self.next_vec[i] = next_obs["vec"]
        self.done[i] = float(done)
        self.action_mask[i] = action_mask.astype(np.float32)
        self.next_action_mask[i] = next_action_mask.astype(np.float32)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)

        def t(arr):
            return torch.as_tensor(arr[idx], device=self.device)

        return (
            t(self.grid),           # (B, 1, 5, 8)
            t(self.vec),            # (B, 10)
            t(self.action),         # (B,)
            t(self.reward),         # (B,)
            t(self.next_grid),      # (B, 1, 5, 8)
            t(self.next_vec),       # (B, 10)
            t(self.done),           # (B,)
            t(self.action_mask),    # (B, 320)
            t(self.next_action_mask),  # (B, 320)
        )

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    DQN agent with action masking and selectable encoder.

    Features:
      - Double DQN (online net picks action, target net evaluates)
      - Masked Q-values (invalid actions → -inf)
      - Soft target network update (Polyak averaging)
      - Huber loss (smooth L1)
    """

    def __init__(
        self,
        encoder_type: str = "mlp",
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        max_grad_norm: float = 0.5,
        buffer_size: int = 500_000,
        batch_size: int = 64,
        learning_starts: int = 10_000,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.learning_starts = learning_starts

        self.q_net = QNetwork(encoder_type, hidden_dim).to(self.device)
        self.target_net = QNetwork(encoder_type, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=buffer_size, device=self.device)

    def act(self, obs: Dict[str, np.ndarray], action_mask: np.ndarray,
            epsilon: float = 0.0) -> int:
        """Select action using ε-greedy with action masking."""
        if random.random() < epsilon:
            valid_actions = np.where(action_mask == 1)[0]
            if len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return 0  # fallback (dead-end handled by env)

        grid = torch.FloatTensor(obs["grid"]).unsqueeze(0).to(self.device)
        vec = torch.FloatTensor(obs["vec"]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(grid, vec).cpu().numpy()[0]

        # Mask invalid actions
        q_values[action_mask == 0] = -np.inf

        return int(np.argmax(q_values))

    def update(self) -> dict:
        """
        Sample one minibatch and perform one gradient step.
        Returns dict with training statistics.
        """
        if len(self.memory) < max(self.batch_size, self.learning_starts):
            return {"loss": 0.0}

        (grid, vec, actions, rewards, next_grid, next_vec,
         dones, action_masks, next_action_masks) = self.memory.sample(self.batch_size)

        # Current Q values: Q(s, a)
        current_q = self.q_net(grid, vec).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            # Online net picks best action for next state (masked)
            next_q_online = self.q_net(next_grid, next_vec)
            next_q_online[next_action_masks == 0] = -float('inf')
            best_actions = next_q_online.argmax(dim=1, keepdim=True)

            # Target net evaluates the value of that action
            next_q_target = self.target_net(next_grid, next_vec)
            next_q = next_q_target.gather(1, best_actions).squeeze(1)

            # Clamp for safety (dead-end states where all actions masked)
            next_q = torch.clamp(next_q, min=-1000.0)

        # Bellman target
        target_q = rewards + (self.gamma * next_q * (1.0 - dones))

        # Huber loss
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(),
                                        max_norm=self.max_grad_norm)
        self.optimizer.step()

        # Soft update target network
        for tp, lp in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)

        return {"loss": loss.item()}

    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "q_net" in checkpoint:
            self.q_net.load_state_dict(checkpoint["q_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            # Legacy format: bare state_dict
            self.q_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)
