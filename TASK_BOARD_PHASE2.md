# Phase 2 Task Board & Acceptance Tracker
## Spatio-Temporal Graph Attention Network (ST-GAT) Engineering

**Project**: Historical METR-LA Traffic Speed Forecasting (Phase 2 Deep Geometric Learning)  
**Parent Course**: Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models  
**Status**: Active Engineering & Implementation  

---

## 1. Phase 2 Workstream Tasks & Ownership

| Task ID | Description | Owner | GitHub ID | Deliverables | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P2-A1** | PyTorch Graph Dataset & Causal Windowing | Ankit Garg | `ankit35027` | `src/models_phase2/graph_dataset.py` | **COMPLETED** |
| **P2-A2** | Sensor Adjacency & Distance Kernel Integration | Ankit Garg | `ankit35027` | `adj_mx.pkl` extraction & 70/10/20 splits | **COMPLETED** |
| **P2-B1** | Graph Attention Network Layer Implementation | Yashpal | `Yashpal-2004` | `src/models_phase2/gat_layer.py` | **COMPLETED** |
| **P2-B2** | Multi-Head GAT with Concat & Average Modes | Yashpal | `Yashpal-2004` | `MultiHeadGATLayer` class & PyTest tests | **COMPLETED** |
| **P2-C1** | Temporal Sequence Backbone (Causal GRU) | Anshika | `anshika292005` | `TemporalEncoder` class | **COMPLETED** |
| **P2-C2** | Multi-Horizon Direct Heads (+15m, +30m, +60m) | Anshika | `anshika292005` | `SpatioTemporalGAT` in `st_gat_model.py` | **COMPLETED** |
| **P2-D1** | Training Harness & Masked Loss Optimization | Samriddhi Shah | `samriddhi-techy` | `src/models_phase2/train_gat.py` | **COMPLETED** |
| **P2-D2** | Test Benchmark Comparison vs Phase 1 Baselines | Samriddhi Shah | `samriddhi-techy` | Quantitative Target Matrix | **COMPLETED** |
| **P2-E1** | Phase 2 Roadmap & Workstream Specifications | Samriddhi Shah | `samriddhi-techy` | `PHASE2_ROADMAP.md`, `requirements_phase2.txt` | **COMPLETED** |

---

## 2. Commit Attribution Guidelines

When committing for Phase 2, each member's commits are attributed to their official identity:
- `Ankit Garg <ankitgarg35027@gmail.com>` (`ankit35027`)
- `Yashpal <yashpal.2024@nst.rishihood.edu.in>` (`Yashpal-2004`)
- `Anshika <anshika292005@users.noreply.github.com>` (`anshika292005`)
- `Samriddhi Shah <samriddhishah25@gmail.com>` (`samriddhi-techy`)
