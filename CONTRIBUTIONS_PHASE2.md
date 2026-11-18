# Phase 2 Team Workstream Division & Git Commit Attribution Guide

**Project**: Historical METR-LA Traffic Speed Forecasting  
**Phase**: Phase 2 Spatio-Temporal Graph Attention Networks (Deep Learning)  
**Team**: 4 Members  

---

## 1. Workstream Responsibilities

```
                         TEAM PROJECT LEADERSHIP
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
  Ankit Garg (WS1)           Yashpal (WS2)             Anshika (WS3)
   [ankit35027]             [Yashpal-2004]            [anshika292005]
 [Graph Data & Lineage]  [Spatial GAT Layer]    [Temporal Backbone & Heads]
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                                   ▼
                         Samriddhi Shah (WS4)
                          [samriddhi-techy]
                        [Training, Benchmarking,
                           Dashboard & Systems]
```

### Member 1: Ankit Garg (`ankit35027`) — Lead, Graph Data Engineering & Lineage (WS1)
- **Primary Modules**: `src/models_phase2/graph_dataset.py`, `src/data/loader.py`
- **Email**: `ankitgarg35027@gmail.com`
- **Key Responsibilities**:
  1. Construct the PyTorch `SpatioTemporalTrafficDataset` converting METR-LA matrices into spatio-temporal graph batches $(B, T=12, N, F=5)$.
  2. Implement causal feature windowing (past 12 steps $t-11 \dots t$) and multi-step target extractions ($t+3, t+6, t+12$).
  3. Ensure strict chronological data isolation: 70% Train, 10% Validation, 20% Test with training-only standardizers.
  4. Build adjacency matrix extractors for both the 15-sensor pilot and the full 207-sensor Caltrans network.
  5. Audit sensor dropouts and causal forward-fill imputation for historical feature lags.

### Member 2: Yashpal (`Yashpal-2004`) — Lead, Spatial Graph Attention (GAT) Architecture (WS2)
- **Primary Modules**: `src/models_phase2/gat_layer.py`, `tests/test_phase2_gat.py`
- **Email**: `yashpal.2024@nst.rishihood.edu.in`
- **Key Responsibilities**:
  1. Implement single-head `GraphAttentionLayer` using PyTorch tensor operations.
  2. Implement learnable projection matrix $\mathbf{W}$ and attention vector $\mathbf{a}$ with LeakyReLU non-linearities.
  3. Mask attention coefficients using the road network adjacency matrix $\mathbf{A}$ and verify row-softmax normalization ($\sum_j \alpha_{ij} = 1.0$).
  4. Develop `MultiHeadGATLayer` supporting $K=4$ independent attention heads with concatenation and averaging modes.
  5. Build unit tests verifying that attention probabilities sum to 1.0 and non-connected nodes receive zero attention.

### Member 3: Anshika (`anshika292005`) — Lead, Temporal Backbone & Multi-Horizon Heads (WS3)
- **Primary Modules**: `src/models_phase2/st_gat_model.py`, `src/models_phase2/__init__.py`
- **Email**: `anshika292005@users.noreply.github.com`
- **Key Responsibilities**:
  1. Implement the causal recurrent temporal encoder (`TemporalEncoder` using GRU) to summarize historical lags without future leakage.
  2. Integrate the temporal representations with the spatial GAT layer via residual connections and layer normalization.
  3. Design direct multi-horizon forecasting heads (independent projection heads for $+15$m, $+30$m, $+60$m).
  4. Ensure end-to-end tensor flow mapping $(B, T=12, N, F=5)$ to multi-horizon predictions $(B, N, 3)$.

### Member 4: Samriddhi Shah (`samriddhi-techy`) — Lead, Training Systems, Benchmarking & Integration (WS4)
- **Primary Modules**: `src/models_phase2/train_gat.py`, `requirements_phase2.txt`, `PHASE2_ROADMAP.md`, `TASK_BOARD_PHASE2.md`
- **Email**: `samriddhishah25@gmail.com`
- **Key Responsibilities**:
  1. Author the end-to-end training harness with AdamW optimizer, learning rate schedulers (`ReduceLROnPlateau`), gradient clipping, and validation early stopping.
  2. Maintain model checkpointing and serialization (`models/st_gat_best_checkpoint.pt`).
  3. Implement masked loss functions (`masked_mae_loss`, `masked_rmse`, `masked_mape`) with $\epsilon$-clamping ($\epsilon = 2.0$ mph).
  4. Generate empirical benchmark comparison tables evaluating ST-GAT against Phase 1 baselines (Persistence, Ridge, XGBoost).
  5. Coordinate Phase 2 scaffolding, project tracking, and dependencies.

---

## 2. Commit Roster

| Member | Full Name | GitHub Account | Email | Assigned Scope |
| :--- | :--- | :--- | :--- | :--- |
| **Member 1** | Ankit Garg | `ankit35027` | `ankitgarg35027@gmail.com` | `src/models_phase2/graph_dataset.py` |
| **Member 2** | Yashpal | `Yashpal-2004` | `yashpal.2024@nst.rishihood.edu.in` | `src/models_phase2/gat_layer.py`, `tests/test_phase2_gat.py` |
| **Member 3** | Anshika | `anshika292005` | `anshika292005@users.noreply.github.com` | `src/models_phase2/st_gat_model.py`, `src/models_phase2/__init__.py` |
| **Member 4** | Samriddhi Shah | `samriddhi-techy` | `samriddhishah25@gmail.com` | `src/models_phase2/train_gat.py`, `requirements_phase2.txt`, `PHASE2_ROADMAP.md`, `TASK_BOARD_PHASE2.md` |
