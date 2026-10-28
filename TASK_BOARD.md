# Project Task Board & Acceptance Tracker

**Project**: Historical METR-LA Traffic Speed Forecasting (Phase 1 Baseline & Classical ML System)  
**Parent Course**: Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models  
**Status**: Active Engineering & Empirical Benchmarking  

---

## 1. Workstream Task Allocation & Milestone Status

| ID | Task Description | Owner | Dependencies | Deliverables | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TSK-A1** | Scaffolding, configuration management, seed locking | Member 4 | None | `src/utils/config.py`, `.github/workflows/ci.yml` | **COMPLETED** |
| **TSK-A2** | Ingestion of raw METR-LA speed matrix & sensor graph | Member 1 | TSK-A1 | `src/data/loader.py`, `data_lineage.json` | **COMPLETED** |
| **TSK-A3** | Missingness audit, sensor alignment & discrepancy audit | Member 1 | TSK-A2 | `src/data/audit.py`, `data_audit.json` | **COMPLETED** |
| **TSK-B1** | Causal feature window builder ($t-11 \dots t \to t+3, 6, 12$) | Member 2 | TSK-A3 | `src/features/build_features.py` | **COMPLETED** |
| **TSK-B2** | Scientific unit tests (leakage, splits, offset alignment) | Member 2 | TSK-B1 | `tests/test_splits_and_leakage.py` | **COMPLETED** |
| **TSK-C1** | Exploratory Data Analysis & Diurnal / ACF figures | Member 1 | TSK-A3 | `notebooks/01_eda.py`, `reports/figures/eda/` | **COMPLETED** |
| **TSK-C2** | Historical Persistence & Ridge Regression Baselines | Member 2 | TSK-B1 | `src/models/persistence.py`, `src/models/ridge_baseline.py` | **COMPLETED** |
| **TSK-C3** | Pilot XGBoost Regressors (15m, 30m, 60m horizons) | Member 3 | TSK-B1 | `src/models/train_xgboost.py` | **COMPLETED** |
| **TSK-D1** | Standardized Multi-Horizon Evaluation & Metrics Engine | Member 3 | TSK-C2, C3 | `src/evaluation/metrics.py`, `metrics_by_horizon.csv` | **COMPLETED** |
| **TSK-D2** | Spatial Lag Feature Ablation (Neighbors vs Isolated) | Member 3 | TSK-D1 | `src/evaluation/ablation.py`, `metrics_ablation.csv` | **COMPLETED** |
| **TSK-D3** | Peak vs Off-Peak Stratified Analysis & Residual Plots | Member 3 | TSK-D1 | `metrics_peak_offpeak.csv`, `reports/figures/evaluation/` | **COMPLETED** |
| **TSK-E1** | Streamlit Interactive Forecasting & Evaluation App | Member 4 | TSK-D1, D3 | `app/streamlit_app.py`, lightweight evaluation assets | **COMPLETED** |
| **TSK-F1** | IEEE Conference Paper LaTeX Manuscript & Bibliography | All (Lead M4) | TSK-A to D | `reports/paper/main.tex`, `references.bib` | **COMPLETED** |
| **TSK-F2** | 15-Slide Academic PowerPoint Presentation Deck | All (Lead M4) | TSK-A to D | `presentation/AML_Phase1_Traffic_Prediction.pptx` | **COMPLETED** |
| **TSK-F3** | Comprehensive Defense & Viva Preparation Guide | Member 4 | TSK-A to D | `VIVA_GUIDE.md` | **COMPLETED** |
| **TSK-F4** | Production Runbook & Repository Documentation | Member 4 | All | `README.md` | **COMPLETED** |

---

## 2. Formal Acceptance Criteria (Runbook Validation)

- [x] **A1: Data Provenance**: Original benchmark dataset source documented with cryptographic SHA-256 hashes in `data_lineage.json`.
- [x] **A2: Sensor Alignment**: Graph sensor sequence matches HDF5 matrix columns 1-to-1 (`207 == 207`).
- [x] **A3: Missingness Accounting**: `NaN` (0), `0.0` (575,302 / 8.11%), and impossible speeds (0) cataloged in `data_audit.json`.
- [x] **B1: Causal Windowing**: Input features strictly drawn from historical past $t-11 \dots t$; target horizons strictly $t+3$ (+15m), $t+6$ (+30m), $t+12$ (+60m).
- [x] **B2: Split Isolation**: Chronological 70% Train, 10% Validation, 20% Test with zero temporal overlap. Scaler fitted strictly on Train.
- [x] **B3: Unit Tests**: Pytest test suite covering temporal boundaries, lack of data leakage, target index offsets, and manual calculation cross-checks.
- [x] **C1: Real Experimental Baselines**: Persistence, Ridge, and XGBoost trained on real data without fabricated metrics.
- [x] **D1: Paired Multi-Horizon Evaluation**: Common evaluation mask across all models; MAE, RMSE, and masked $\epsilon$-MAPE reported.
- [x] **D2: Topological Feature Ablation**: Road graph spatial lag comparison scientifically executed and reported.
- [x] **E1: Streamlit Dashboard**: Operates completely offline on precomputed evaluation artifacts without live retraining.
- [x] **F1: IEEE Manuscript & Deck**: Complete LaTeX paper with genuine figures/tables, and 15-slide PowerPoint deck with speaker notes.
