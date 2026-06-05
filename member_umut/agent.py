"""
PPO Agent — from scratch, PyTorch.

Implements:
  - Rollout buffer for trajectory collection
  - GAE (Generalized Advantage Estimation)
  - Clipped surrogate objective
  - Entropy bonus
  - Value function loss
  - Mini-batch updates
  - Action masking integration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from member_umut.network import ActorCritic


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    mini_batch_size: int = 64
    ppo_epochs: int = 4
    encoder_type: str = "mlp"
    hidden_dim: int = 256
    cnn_channels: list = None  # default [32, 64]

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64]


class RolloutBuffer:
    """
    Fixed-size buffer for collecting rollout trajectories.

    Stores observations, actions, rewards, dones, log_probs, values, and action_masks.
    Computes GAE advantages and returns when finalized.
    """

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.full = False

        # Pre-allocate storage
        self.grids = np.zeros((capacity, 1, 5, 8), dtype=np.float32)
        self.vecs = np.zeros((capacity, 10), dtype=np.float32)
        self.action_masks = np.zeros((capacity, 320), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

    def add(self, grid, vec, action_mask, action, log_prob, reward, done, value):
        """Add a single transition to the buffer."""
        self.grids[self.pos] = grid
        self.vecs[self.pos] = vec
        self.action_masks[self.pos] = action_mask.astype(np.float32)
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        self.pos += 1
        if self.pos == self.capacity:
            self.full = True

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute GAE advantages and discounted returns."""
        last_gae = 0.0
        for t in reversed(range(self.pos)):
            if t == self.pos - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos]

    def get_batches(self, batch_size: int):
        """Yield mini-batch indices for PPO epochs."""
        size = self.pos
        indices = np.arange(size)
        np.random.shuffle(indices)
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_idx = indices[start:end]
            yield self._get_tensors(batch_idx)

    def _get_tensors(self, idx):
        """Convert a batch of indices to tensors on the target device."""
        return {
            "grid": torch.tensor(self.grids[idx], device=self.device),
            "vec": torch.tensor(self.vecs[idx], device=self.device),
            "action_mask": torch.tensor(self.action_masks[idx], device=self.device),
            "action": torch.tensor(self.actions[idx], device=self.device),
            "old_log_prob": torch.tensor(self.log_probs[idx], device=self.device),
            "advantage": torch.tensor(self.advantages[idx], device=self.device),
            "returns": torch.tensor(self.returns[idx], device=self.device),
        }

    def reset(self):
        self.pos = 0
        self.full = False


class PPOAgent:
    """
    PPO agent with action masking.

    Usage:
        agent = PPOAgent(config, device)
        # Collect rollout
        action, log_prob, value = agent.select_action(obs, mask)
        agent.buffer.add(...)
        # Update
        stats = agent.update(last_value)
    """

    def __init__(self, config: PPOConfig, device: torch.device):
        self.config = config
        self.device = device

        self.network = ActorCritic(
            encoder_type=config.encoder_type,
            hidden_dim=config.hidden_dim,
            cnn_channels=config.cnn_channels,
        ).to(device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr, eps=1e-5)
        self.buffer = RolloutBuffer(config.rollout_steps, device)

    @torch.no_grad()
    def select_action(self, obs: dict, action_mask: np.ndarray):
        """
        Select an action using the current policy.

        Returns: (action, log_prob, value) as numpy scalars.
        """
        grid = torch.tensor(obs["grid"], device=self.device).unsqueeze(0)
        vec = torch.tensor(obs["vec"], device=self.device).unsqueeze(0)
        mask = torch.tensor(action_mask.astype(np.float32), device=self.device).unsqueeze(0)

        action, log_prob, _, value = self.network.get_action_and_value(grid, vec, mask)

        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def get_value(self, obs: dict, action_mask: np.ndarray) -> float:
        """Get value estimate for GAE bootstrap."""
        grid = torch.tensor(obs["grid"], device=self.device).unsqueeze(0)
        vec = torch.tensor(obs["vec"], device=self.device).unsqueeze(0)
        mask = torch.tensor(action_mask.astype(np.float32), device=self.device).unsqueeze(0)
        value = self.network.get_value(grid, vec, mask)
        return value.item()

    def update(self, last_value: float) -> dict:
        """
        Run PPO update using collected rollout data.

        Returns dict of training statistics.
        """
        cfg = self.config

        # Compute GAE
        self.buffer.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for _ in range(cfg.ppo_epochs):
            for batch in self.buffer.get_batches(cfg.mini_batch_size):
                grid = batch["grid"]
                vec = batch["vec"]
                action_mask = batch["action_mask"]
                action = batch["action"]
                old_log_prob = batch["old_log_prob"]
                advantage = batch["advantage"]
                returns = batch["returns"]

                # Normalize advantages
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # Forward pass
                _, new_log_prob, entropy, value = self.network.get_action_and_value(
                    grid, vec, action_mask, action
                )

                # Policy loss (clipped surrogate)
                log_ratio = new_log_prob - old_log_prob
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (returns - value).pow(2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                # Stats
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                n_updates += 1

        # Reset buffer
        self.buffer.reset()

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": total_approx_kl / n_updates,
        }

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
