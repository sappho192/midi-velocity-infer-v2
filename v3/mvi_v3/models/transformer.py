from __future__ import annotations

import torch
from torch import nn

from mvi_v3.config import BaselineConfig

from .conditioning import ControlEmbedding
from .embedding import NoteEmbedding
from .heads import ClassificationVelocityHead, StochasticVelocityHead, VelocityHead
from .position import T5RelativePositionBias


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states
        x = self.norm1(hidden_states)
        attn_mask = attn_bias.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        attn_mask = attn_mask.reshape(batch_size * attn_bias.shape[0], seq_len, seq_len)
        attended, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = residual + self.dropout(attended)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerVelocityModel(nn.Module):
    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.embedding = NoteEmbedding(len(config.continuous_features), config.d_model)
        self.position_bias = T5RelativePositionBias(config.n_heads)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(config.d_model, config.n_heads, config.ffn_dim, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.d_model)

        # Embedding dropout (regularization)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout) if config.embedding_dropout > 0 else None

        # Controllable velocity inference
        self.enable_controls = config.enable_controls
        self.control_dims = config.control_dims
        if self.enable_controls:
            self.control_embedding = ControlEmbedding(config.control_dims, config.d_model)

        # Output head
        self.head_type = config.head_type
        if config.head_type == "classification":
            self.head = ClassificationVelocityHead(config.d_model, config.num_velocity_bins)
        elif config.head_type == "stochastic" or config.stochastic_head:
            self.head = StochasticVelocityHead(config.d_model)
            self.head_type = "stochastic"
        else:
            self.head = VelocityHead(config.d_model)

        # Auxiliary reconstruction head for masked attribute regularization (Phase 3)
        self.has_aux_head = config.mask_ratio > 0
        if self.has_aux_head:
            n_continuous = len(config.continuous_features)
            self.aux_head = nn.Sequential(
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
        control_params: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(pitch, register_bucket, continuous)

        if self.embedding_dropout is not None:
            x = self.embedding_dropout(x)

        if self.enable_controls:
            # For stochastic head with 3+ control dims, extract surprise (dim 2)
            surprise = None
            ctrl_for_embedding = control_params
            if self.head_type == "stochastic" and control_params is not None and self.control_dims >= 3:
                surprise = control_params[:, 2:3]  # [batch, 1]
            x = x + self.control_embedding(ctrl_for_embedding, x.shape[0])

        bias = self.position_bias(x.shape[1], x.device)
        for layer in self.layers:
            x = layer(x, padding_mask, bias)
        x = self.output_norm(x)

        # Stochastic head with surprise routing
        if self.head_type == "stochastic" and self.enable_controls:
            return self.head(x, surprise=surprise)

        return self.head(x)

    def forward_with_aux(
        self,
        pitch: torch.Tensor,
        register_bucket: torch.Tensor,
        continuous: torch.Tensor,
        padding_mask: torch.Tensor,
        control_params: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass that also returns auxiliary reconstruction predictions.

        Returns (main_output, aux_reconstruction) where aux_reconstruction
        is [batch, seq_len, n_continuous] predicting the original features.
        """
        x = self.embedding(pitch, register_bucket, continuous)

        if self.embedding_dropout is not None:
            x = self.embedding_dropout(x)

        if self.enable_controls:
            surprise = None
            if self.head_type == "stochastic" and control_params is not None and self.control_dims >= 3:
                surprise = control_params[:, 2:3]
            x = x + self.control_embedding(control_params, x.shape[0])

        bias = self.position_bias(x.shape[1], x.device)
        for layer in self.layers:
            x = layer(x, padding_mask, bias)
        x = self.output_norm(x)

        # Auxiliary reconstruction from transformer output
        aux_recon = self.aux_head(x)

        # Main output
        if self.head_type == "stochastic" and self.enable_controls:
            main_output = self.head(x, surprise=surprise)
        else:
            main_output = self.head(x)

        return main_output, aux_recon
