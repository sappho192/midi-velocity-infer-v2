from __future__ import annotations

import torch
from torch import nn


class ControlPredictorMLP(nn.Module):
    """Lightweight MLP that predicts control parameters from window features.

    Replaces the external RandomForest regressor (~890MB) with a tiny
    PyTorch model (~23KB) mapping 21-dim hand-crafted window features
    to 2-dim normalized controls (expressiveness, dynamics_center).
    """

    def __init__(self, input_dim: int = 21, hidden_dim: int = 64, output_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # controls are min-max normalized to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict controls from window features.

        Args:
            x: [batch, input_dim] window features.

        Returns:
            [batch, output_dim] predicted controls in [0, 1].
        """
        return self.net(x)
