import torch
from torch import nn


class T5RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_exact = num_buckets // 2
        distance = (-relative_position).clamp(min=0)
        is_small = distance < max_exact
        large_pos = max_exact + (
            torch.log(distance.float() / max_exact + 1e-6)
            / torch.log(torch.tensor(self.max_distance / max_exact, device=distance.device))
            * (num_buckets - max_exact)
        ).to(torch.long)
        large_pos = torch.minimum(
            large_pos,
            torch.full_like(large_pos, num_buckets - 1),
        )
        return torch.where(is_small, distance, large_pos)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        context = torch.arange(seq_len, device=device)[:, None]
        memory = torch.arange(seq_len, device=device)[None, :]
        relative_position = memory - context
        buckets = self._relative_position_bucket(relative_position)
        bias = self.relative_attention_bias(buckets)
        return bias.permute(2, 0, 1)
