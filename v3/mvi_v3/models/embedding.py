import torch
from torch import nn


class NoteEmbedding(nn.Module):
    def __init__(self, continuous_dim: int, d_model: int) -> None:
        super().__init__()
        self.pitch_embedding = nn.Embedding(128, d_model)
        self.register_embedding = nn.Embedding(4, d_model)
        self.continuous_projection = nn.Sequential(
            nn.Linear(continuous_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        pitch: torch.Tensor,
        register_bucket: torch.Tensor,
        continuous: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.pitch_embedding(pitch)
            + self.register_embedding(register_bucket)
            + self.continuous_projection(continuous)
        )
