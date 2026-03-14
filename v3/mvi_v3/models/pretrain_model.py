"""PretrainModel for Masked Note Modeling (MNM) SSL pretraining.

Shares the same backbone architecture as TransformerVelocityModel
(embedding, position_bias, layers, output_norm) with MNM-specific heads.
"""
from __future__ import annotations

import torch
from torch import nn

from mvi_v3.config import BaselineConfig

from .embedding import NoteEmbedding
from .position import T5RelativePositionBias
from .transformer import EncoderBlock


class PretrainModel(nn.Module):
    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        n_continuous = len(config.continuous_features)

        # Backbone (same key names as TransformerVelocityModel)
        self.embedding = NoteEmbedding(n_continuous, config.d_model)
        self.position_bias = T5RelativePositionBias(config.n_heads)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(config.d_model, config.n_heads, config.ffn_dim, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.d_model)

        # Embedding dropout (regularization)
        self.embedding_dropout = (
            nn.Dropout(config.embedding_dropout) if config.embedding_dropout > 0 else None
        )

        # Learnable mask vector (pretrain-only, not part of backbone)
        self.mask_embedding = nn.Parameter(torch.randn(config.d_model) * 0.02)

        # MNM heads (pretrain-only)
        self.pitch_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 128),
        )
        self.continuous_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, n_continuous),
        )

    def forward(
        self,
        pitch: torch.Tensor,
        register_bucket: torch.Tensor,
        continuous: torch.Tensor,
        padding_mask: torch.Tensor,
        mnm_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with MNM masking.

        Args:
            pitch: [batch, seq_len] MIDI pitch values
            register_bucket: [batch, seq_len] register bins
            continuous: [batch, seq_len, n_continuous] normalized features
            padding_mask: [batch, seq_len] True=padding
            mnm_mask: [batch, seq_len] True=masked for MNM

        Returns:
            (pitch_logits, continuous_pred) where:
            - pitch_logits: [batch, seq_len, 128] pitch classification logits
            - continuous_pred: [batch, seq_len, n_continuous] feature reconstruction
        """
        x = self.embedding(pitch, register_bucket, continuous)

        if self.embedding_dropout is not None:
            x = self.embedding_dropout(x)

        # Replace masked positions with learnable mask vector
        x[mnm_mask] = self.mask_embedding

        bias = self.position_bias(x.shape[1], x.device)
        for layer in self.layers:
            x = layer(x, padding_mask, bias)
        x = self.output_norm(x)

        pitch_logits = self.pitch_head(x)
        continuous_pred = self.continuous_head(x)

        return pitch_logits, continuous_pred

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract only the backbone weights (compatible with TransformerVelocityModel).

        Includes: embedding, position_bias, layers, output_norm, embedding_dropout.
        Excludes: mask_embedding, pitch_head, continuous_head.
        """
        backbone_prefixes = ("embedding.", "position_bias.", "layers.", "output_norm.")
        if self.embedding_dropout is not None:
            backbone_prefixes = backbone_prefixes + ("embedding_dropout.",)
        return {
            k: v for k, v in self.state_dict().items()
            if k.startswith(backbone_prefixes)
        }
