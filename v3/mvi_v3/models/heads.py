from torch import nn


class VelocityHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, hidden_states):
        return self.layers(hidden_states).squeeze(-1)
