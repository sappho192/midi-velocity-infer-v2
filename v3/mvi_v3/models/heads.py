from __future__ import annotations

import torch
from torch import nn


class VelocityHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden_states).squeeze(-1)


class StochasticVelocityHead(nn.Module):
    """Outputs (mu, log_sigma) per note for probabilistic velocity prediction.

    At inference, multiple alternatives can be sampled.
    The `surprise` control parameter scales sigma.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.mu_proj = nn.Linear(d_model, 1)
        self.log_sigma_proj = nn.Linear(d_model, 1)
        # Initialize log_sigma bias to produce small initial sigma
        nn.init.constant_(self.log_sigma_proj.bias, -2.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        surprise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hidden_states: [batch, seq_len, d_model]
            surprise: [batch, 1] in [0, 1], scales sigma. None uses 0.5.

        Returns:
            (mu, log_sigma) each of shape [batch, seq_len]
        """
        h = self.shared(hidden_states)
        mu = self.mu_proj(h).squeeze(-1)
        log_sigma = self.log_sigma_proj(h).squeeze(-1)

        if surprise is not None:
            # Scale sigma: surprise=0 → sigma*0.1, surprise=1 → sigma*2.0
            scale = 0.1 + 1.9 * surprise  # [batch, 1]
            log_sigma = log_sigma + scale.log()

        return mu, log_sigma

    def sample(
        self,
        hidden_states: torch.Tensor,
        n_alternatives: int = 1,
        surprise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample multiple velocity alternatives.

        Args:
            hidden_states: [batch, seq_len, d_model]
            n_alternatives: number of samples to draw
            surprise: [batch, 1] in [0, 1]

        Returns:
            [n_alternatives, batch, seq_len] sampled velocities
        """
        mu, log_sigma = self.forward(hidden_states, surprise=surprise)
        sigma = log_sigma.exp()
        # [n_alternatives, batch, seq_len]
        eps = torch.randn(n_alternatives, *mu.shape, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
