"""Transformer-based reconstruction model for event-based anomaly detection."""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for window position."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AnomalyTransformer(nn.Module):
    """
    Event-based anomaly detection model.

    Architecture:
        1. Feature projection: Linear(input_dim, d_model - gap_embed_dim)
        2. Gap embedding: Embedding(num_bins, gap_embed_dim)
        3. Concatenate -> (batch, seq_len, d_model)
        4. Positional encoding
        5. TransformerEncoder (num_layers)
        6. Reconstruction head: Linear(d_model, input_dim)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        num_gap_bins: int,
        gap_embed_dim: int,
        max_pos_len: int,
    ):
        super().__init__()
        feat_proj_dim = d_model - gap_embed_dim

        self.feature_proj = nn.Linear(input_dim, feat_proj_dim)
        self.gap_embedding = nn.Embedding(num_gap_bins, gap_embed_dim)
        self.pos_encoder = PositionalEncoding(d_model, max_pos_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.reconstruction_head = nn.Linear(d_model, input_dim)

    def forward(
        self, features: torch.Tensor, gap_bins: torch.LongTensor
    ) -> torch.Tensor:
        """
        Args:
            features: (batch, seq_len, input_dim) raw feature vectors
            gap_bins: (batch, seq_len) discretized gap bin indices

        Returns:
            reconstructed: (batch, seq_len, input_dim)
        """
        feat_emb = self.feature_proj(features)
        gap_emb = self.gap_embedding(gap_bins)
        x = torch.cat([feat_emb, gap_emb], dim=-1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.reconstruction_head(x)


def build_model(model_cfg: dict) -> AnomalyTransformer:
    """Instantiate AnomalyTransformer from model config dict."""
    gap_cfg = model_cfg["gap_embedding"]
    pos_cfg = model_cfg["positional_encoding"]

    return AnomalyTransformer(
        input_dim=model_cfg["input_dim"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        activation=model_cfg["activation"],
        num_gap_bins=gap_cfg["num_bins"],
        gap_embed_dim=gap_cfg["embed_dim"],
        max_pos_len=pos_cfg["max_len"],
    )
