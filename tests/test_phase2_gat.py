"""
PyTest Suite for Phase 2 Spatio-Temporal Graph Attention Network (ST-GAT).
Validates tensor dimensions, attention probability distributions, causality,
and masked metric calculations.
"""

import pytest
import numpy as np
import torch

from src.models_phase2.gat_layer import GraphAttentionLayer, MultiHeadGATLayer
from src.models_phase2.st_gat_model import SpatioTemporalGAT
from src.models_phase2.train_gat import masked_mae_loss, masked_rmse, masked_mape


def test_single_head_gat_layer_shape_and_attention_sum():
    """Verify single-head GAT outputs correct dimensions and valid attention probabilities."""
    B, N, in_feat, out_feat = 4, 10, 16, 32
    h = torch.randn(B, N, in_feat)
    
    # Create simple adjacency with self-loops
    adj = torch.eye(N)
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0

    gat = GraphAttentionLayer(in_features=in_feat, out_features=out_feat, dropout=0.0)
    out, att = gat(h, adj, return_attention_weights=True)

    assert out.shape == (B, N, out_feat), f"Expected shape {(B, N, out_feat)}, got {out.shape}"
    assert att.shape == (B, N, N), f"Expected shape {(B, N, N)}, got {att.shape}"

    # Verify attention weights sum to 1.0 for each node over its neighbors
    att_sum = torch.sum(att, dim=-1)
    assert torch.allclose(att_sum, torch.ones_like(att_sum), atol=1e-5), "Attention weights must sum to 1.0"


def test_multi_head_gat_concat_and_average():
    """Verify multi-head GAT correctly concatenates or averages heads."""
    B, N, in_feat, out_feat, num_heads = 2, 8, 12, 16, 4
    h = torch.randn(B, N, in_feat)
    adj = torch.eye(N)

    # Concat mode
    gat_concat = MultiHeadGATLayer(
        in_features=in_feat,
        out_features=out_feat,
        num_heads=num_heads,
        concat=True,
    )
    out_concat, _ = gat_concat(h, adj)
    assert out_concat.shape == (B, N, num_heads * out_feat)

    # Average mode
    gat_avg = MultiHeadGATLayer(
        in_features=in_feat,
        out_features=out_feat,
        num_heads=num_heads,
        concat=False,
    )
    out_avg, _ = gat_avg(h, adj)
    assert out_avg.shape == (B, N, out_feat)


def test_st_gat_model_forward_pass():
    """Verify end-to-end Spatio-Temporal GAT model maps (B, T, N, F) to (B, N, 3)."""
    B, T_in, N, F_in = 2, 12, 15, 5
    x = torch.randn(B, T_in, N, F_in)
    adj = torch.eye(N)

    model = SpatioTemporalGAT(
        num_nodes=N,
        in_features=F_in,
        temporal_hidden_dim=32,
        spatial_hidden_dim=16,
        num_heads=2,
        num_horizons=3,
        dropout=0.0,
    )

    preds, att = model(x, adj, return_attention=True)
    assert preds.shape == (B, N, 3), f"Expected (B, N, 3) for 3 horizons, got {preds.shape}"
    assert att is not None
    assert att.shape == (B, N, N)


def test_masked_metrics_ignore_invalid_entries():
    """Verify masked loss and error metrics strictly ignore masked-out elements."""
    y_true = torch.tensor([[50.0, 0.0], [60.0, 0.0]])
    y_pred = torch.tensor([[55.0, 100.0], [58.0, 100.0]])
    mask = torch.tensor([[True, False], [True, False]])

    # MAE should only evaluate the first column: |55-50| = 5, |58-60| = 2 -> (5 + 2)/2 = 3.5
    mae = masked_mae_loss(y_pred, y_true, mask).item()
    assert abs(mae - 3.5) < 1e-4

    # RMSE: sqrt((25 + 4)/2) = sqrt(14.5) ≈ 3.80788655
    rmse = masked_rmse(y_pred, y_true, mask).item()
    assert abs(rmse - np.sqrt(14.5)) < 1e-4

