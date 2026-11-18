"""
Spatio-Temporal Graph Attention Network (ST-GAT) Architecture (Phase 2).
Jointly models temporal traffic dynamics via recurrent sequence encoding (GRU)
and spatial road network shockwave propagation via dynamic Multi-Head Graph Attention (GAT).

Architectural Pipeline:
-----------------------
Input: X in R^{B x T_in x N x F_in} (B: batch, T_in: 12 lags, N: sensors, F_in: features)
  │
  ▼
Temporal Sequence Encoder (GRU / 1D-CNN over T_in for each sensor)
  │
  ▼
Spatial Multi-Head Graph Attention (GAT over sensor graph N x N)
  │
  ▼
Residual Connection & Feed-Forward Normalization
  │
  ▼
Multi-Horizon Direct Forecasting Heads (15m, 30m, 60m)
  │
  ▼
Output: Y_hat in R^{B x N x H} (Speed predictions in mph)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models_phase2.gat_layer import MultiHeadGATLayer


class TemporalEncoder(nn.Module):
    """
    Encodes the historical time-series lags for each sensor node independently.
    Uses a causal GRU to summarize T_in time steps into a dense hidden vector.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super(TemporalEncoder, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T_in, N, in_features)
        Returns:
            out: Encoded temporal tensor of shape (B, N, hidden_dim)
        """
        B, T_in, N, F_in = x.shape
        # Reshape to (B * N, T_in, F_in) to process all nodes in parallel
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * N, T_in, F_in)
        
        _, h_n = self.gru(x_reshaped)  # h_n shape: (num_layers, B * N, hidden_dim)
        
        # Take the top layer hidden state at the final time step
        h_last = h_n[-1]  # shape: (B * N, hidden_dim)
        
        # Reshape back to (B, N, hidden_dim)
        out = h_last.view(B, N, self.hidden_dim)
        return out


class SpatioTemporalGAT(nn.Module):
    """
    End-to-End Spatio-Temporal Graph Attention Network for Traffic Speed Prediction.
    """

    def __init__(
        self,
        num_nodes: int,
        in_features: int = 5,
        temporal_hidden_dim: int = 64,
        spatial_hidden_dim: int = 32,
        num_heads: int = 4,
        num_horizons: int = 3,
        dropout: float = 0.1,
        alpha: float = 0.2,
    ):
        super(SpatioTemporalGAT, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.temporal_hidden_dim = temporal_hidden_dim
        self.spatial_hidden_dim = spatial_hidden_dim
        self.num_heads = num_heads
        self.num_horizons = num_horizons

        # 1. Temporal sequence encoder
        self.temporal_encoder = TemporalEncoder(
            in_features=in_features,
            hidden_dim=temporal_hidden_dim,
            num_layers=1,
            dropout=dropout,
        )

        # 2. Spatial Multi-Head GAT layer 1
        # Input: temporal_hidden_dim -> Output: spatial_hidden_dim * num_heads
        self.gat_layer1 = MultiHeadGATLayer(
            in_features=temporal_hidden_dim,
            out_features=spatial_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            concat=True,
        )

        gat_out_dim = spatial_hidden_dim * num_heads

        # 3. Spatial Multi-Head GAT layer 2 (averaging heads)
        self.gat_layer2 = MultiHeadGATLayer(
            in_features=gat_out_dim,
            out_features=temporal_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            concat=False,
        )

        # Layer normalization & residual projections
        self.norm1 = nn.LayerNorm(gat_out_dim)
        self.norm2 = nn.LayerNorm(temporal_hidden_dim)
        self.residual_proj = nn.Linear(temporal_hidden_dim, gat_out_dim)

        # 4. Multi-Horizon Direct Prediction Heads (15m, 30m, 60m)
        self.head_15m = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.head_30m = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.head_60m = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward computation.
        
        Args:
            x: Historical speed & time features: shape (B, T_in, N, in_features)
            adj: Highway sensor adjacency matrix: shape (N, N)
            return_attention: Whether to return dynamic spatial attention weights
            
        Returns:
            predictions: Tensor of shape (B, N, 3) representing forecasts for (+15m, +30m, +60m)
            attention: Optional attention tensor from GAT layer
        """
        # 1. Temporal Encoding -> (B, N, temporal_hidden_dim)
        h_temp = self.temporal_encoder(x)

        # 2. Spatial GAT Layer 1 with Residual Connection
        h_gat1, att1 = self.gat_layer1(h_temp, adj, return_attention_weights=return_attention)
        h_gat1 = self.norm1(h_gat1 + self.residual_proj(h_temp))

        # 3. Spatial GAT Layer 2 with Residual Connection
        h_gat2, att2 = self.gat_layer2(h_gat1, adj, return_attention_weights=return_attention)
        h_spatial = self.norm2(h_gat2 + h_temp)

        # 4. Multi-Horizon Forecast Heads
        pred_15m = self.head_15m(h_spatial)  # (B, N, 1)
        pred_30m = self.head_30m(h_spatial)  # (B, N, 1)
        pred_60m = self.head_60m(h_spatial)  # (B, N, 1)

        # Concat horizons: (B, N, 3)
        predictions = torch.cat([pred_15m, pred_30m, pred_60m], dim=-1)

        if return_attention:
            return predictions, att1
        return predictions, None

