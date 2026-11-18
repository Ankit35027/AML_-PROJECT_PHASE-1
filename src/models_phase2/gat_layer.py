"""
Graph Attention Network (GAT) Layer Module (Phase 2).
Implements multi-head graph self-attention based on Veličković et al. (ICLR 2018),
adapted for directed highway sensor graphs with continuous physical edge weights.

Mathematical Formulation:
-------------------------
For sensor node features h_i in R^F and adjacency A in R^{N x N}:
1. Linear projection: z_i = W h_i, where W in R^{F' x F}
2. Self-attention energy: e_{ij} = LeakyReLU( a^T [ z_i || z_j ] )
3. Attention coefficients (masked by graph adjacency):
   alpha_{ij} = exp(e_{ij}) / sum_{k in N_i} exp(e_{ik})
4. Output representation (multi-head concat or average):
   h_i' = sigma( sum_{j in N_i} alpha_{ij} z_j )
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Single-head Graph Attention Network (GAT) layer.
    Supports both unweighted adjacency masks and continuous distance-based edge weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Learnable weight matrix W: [F_in, F_out]
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Learnable attention vector a: [2 * F_out, 1]
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for single-head GAT.
        
        Args:
            h: Node feature tensor of shape (B, N, in_features) or (N, in_features)
            adj: Adjacency matrix of shape (N, N) where adj[i, j] > 0 indicates an edge i -> j
            return_attention_weights: If True, returns the computed attention matrix
            
        Returns:
            h_prime: Updated node features of shape (..., out_features)
            attention: Optional attention coefficients of shape (N, N) or (B, N, N)
        """
        has_batch = h.dim() == 3
        if not has_batch:
            h = h.unsqueeze(0)  # Shape: (1, N, in_features)

        B, N, _ = h.shape

        # 1. Linear projection: z = h W -> (B, N, out_features)
        z = torch.matmul(h, self.W)

        # 2. Pairwise feature concatenation for attention
        # z_i: (B, N, 1, out_features) replicated along columns
        # z_j: (B, 1, N, out_features) replicated along rows
        z_i = z.unsqueeze(2).expand(B, N, N, self.out_features)
        z_j = z.unsqueeze(1).expand(B, N, N, self.out_features)
        
        # Concat: (B, N, N, 2 * out_features)
        z_cat = torch.cat([z_i, z_j], dim=-1)

        # Compute raw energy: e_{ij} = LeakyReLU( a^T [z_i || z_j] ) -> (B, N, N)
        e = self.leakyrelu(torch.matmul(z_cat, self.a).squeeze(-1))

        # 3. Mask out non-edges using adjacency matrix
        # If adj[i, j] == 0, replace with large negative number so softmax -> 0
        zero_vec = -9e15 * torch.ones_like(e)
        adj_expanded = adj.unsqueeze(0).expand(B, N, N)
        adj_mask = adj_expanded > 0.0

        masked_e = torch.where(adj_mask, e, zero_vec)

        # Softmax over neighborhood j for each node i
        attention = F.softmax(masked_e, dim=-1)
        attention = self.dropout_layer(attention)

        # 4. Aggregation: h' = sum_j alpha_{ij} * z_j -> (B, N, out_features)
        h_prime = torch.matmul(attention, z)

        if self.concat:
            out = F.elu(h_prime)
        else:
            out = h_prime

        if not has_batch:
            out = out.squeeze(0)
            if attention is not None:
                attention = attention.squeeze(0)

        if return_attention_weights:
            return out, attention
        return out, None


class MultiHeadGATLayer(nn.Module):
    """
    Multi-Head Graph Attention Layer.
    Executes K independent attention heads and concatenates or averages their outputs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.out_features = out_features

        self.heads = nn.ModuleList([
            GraphAttentionLayer(
                in_features=in_features,
                out_features=out_features,
                dropout=dropout,
                alpha=alpha,
                concat=concat,
            )
            for _ in range(num_heads)
        ])

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass combining K attention heads.
        """
        head_outputs = []
        attentions = []

        for head in self.heads:
            out, att = head(h, adj, return_attention_weights=return_attention_weights)
            head_outputs.append(out)
            if return_attention_weights:
                attentions.append(att)

        if self.concat:
            # Concatenate along feature dimension -> (B, N, num_heads * out_features)
            total_out = torch.cat(head_outputs, dim=-1)
        else:
            # Average heads -> (B, N, out_features)
            total_out = torch.mean(torch.stack(head_outputs, dim=0), dim=0)

        avg_attention = None
        if return_attention_weights:
            avg_attention = torch.mean(torch.stack(attentions, dim=0), dim=0)

        return total_out, avg_attention

