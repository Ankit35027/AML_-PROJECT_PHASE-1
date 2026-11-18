# Phase 2 Task Board & Acceptance Tracker
## Spatio-Temporal Graph Attention Network (ST-GAT) Engineering

**Project**: Historical METR-LA Traffic Speed Forecasting (Phase 2 Deep Geometric Learning)  
**Parent Course**: Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models  
**Status**: 100% COMPLETED, TRAINED, EVALUATED & MERGED TO MAIN  

---

## 1. Phase 2 Workstream Tasks & Ownership

| Task ID | Description | Owner | GitHub ID | Deliverables | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P2-A1** | PyTorch Graph Dataset & Causal Windowing | Ankit Garg | `ankit35027` | `src/models_phase2/graph_dataset.py` | **VERIFIED & MERGED** |
| **P2-A2** | Sensor Adjacency & Distance Kernel Integration | Ankit Garg | `ankit35027` | `adj_mx.pkl` extraction & 70/10/20 splits | **VERIFIED & MERGED** |
| **P2-B1** | Graph Attention Network Layer Implementation | Yashpal | `Yashpal-2004` | `src/models_phase2/gat_layer.py` | **VERIFIED & MERGED** |
| **P2-B2** | Multi-Head GAT with Concat & Average Modes | Yashpal | `Yashpal-2004` | `MultiHeadGATLayer` class & PyTest tests | **VERIFIED & MERGED** |
| **P2-C1** | Temporal Sequence Backbone (Causal GRU) | Anshika | `anshika292005` | `TemporalEncoder` class | **VERIFIED & MERGED** |
| **P2-C2** | Multi-Horizon Direct Heads (+15m, +30m, +60m) | Anshika | `anshika292005` | `SpatioTemporalGAT` in `st_gat_model.py` | **VERIFIED & MERGED** |
| **P2-D1** | Training Harness & Masked Loss Optimization | Samriddhi Shah | `samriddhi-techy` | `src/models_phase2/train_gat.py` | **VERIFIED & MERGED** |
| **P2-D2** | Test Benchmark Comparison vs Phase 1 Baselines | Samriddhi Shah | `samriddhi-techy` | `metrics_phase2_gat.csv` & test parquet | **VERIFIED & MERGED** |
| **P2-E1** | Phase 2 Roadmap & Workstream Specifications | Samriddhi Shah | `samriddhi-techy` | `PHASE2_ROADMAP.md`, `requirements_phase2.txt` | **VERIFIED & MERGED** |

---

## 2. Verified Empirical Benchmark Outcomes

Evaluated on 90,149 valid test instances across 207 highway sensors:
- **15-Min Horizon**: MAE 2.4124 mph (vs XGBoost 2.5909 mph & Persistence 3.0011 mph) -> **6.89% error reduction**
- **30-Min Horizon**: MAE 2.8945 mph (vs XGBoost 3.1668 mph & Persistence 3.6157 mph) -> **8.60% error reduction**
- **60-Min Horizon**: MAE 3.5821 mph (vs XGBoost 4.0083 mph & Persistence 4.7302 mph) -> **10.63% error reduction**

All artifacts, checkpoints (`models/st_gat_best_checkpoint.pt`), and evaluation figures are pushed directly to `main`.

---

## 2. Commit Attribution Guidelines

When committing for Phase 2, each member's commits are attributed to their official identity:
- `Ankit Garg <ankitgarg35027@gmail.com>` (`ankit35027`)
- `Yashpal <yashpal.2024@nst.rishihood.edu.in>` (`Yashpal-2004`)
- `Anshika <anshika292005@users.noreply.github.com>` (`anshika292005`)
- `Samriddhi Shah <samriddhishah25@gmail.com>` (`samriddhi-techy`)
