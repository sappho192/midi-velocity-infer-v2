from __future__ import annotations

import torch
from torch import nn


class ControlEmbedding(nn.Module):
    """Maps scalar control parameters to a d_model conditioning vector.

    Control parameters (e.g., expressiveness, dynamics_center, surprise)
    are projected through a small MLP. The output is broadcast-added
    to token embeddings before Transformer layers.
    """

    def __init__(self, n_controls: int, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_controls, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Learnable defaults for when controls aren't provided
        self.default_controls = nn.Parameter(torch.zeros(n_controls))

    def forward(self, control_params: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        """Compute conditioning vector.

        Args:
            control_params: [batch_size, n_controls] or None (uses defaults).
            batch_size: batch size for broadcasting defaults.

        Returns:
            [batch_size, 1, d_model] conditioning vector for broadcast addition.
        """
        if control_params is None:
            control_params = self.default_controls.unsqueeze(0).expand(batch_size, -1)
        # [batch_size, d_model] -> [batch_size, 1, d_model]
        return self.mlp(control_params).unsqueeze(1)
