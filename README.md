# Historical METR-LA Traffic Speed Forecasting
## Phase 1 Classical Machine Learning & Phase 2 Spatio-Temporal Graph Attention Networks (ST-GAT)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)

**Parent Course / Program Title**: *Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models*  
**Project Scope**:
- **Phase 1**: Classical Machine Learning, Statistical Baselines (Persistence, Ridge, XGBoost), Multi-Horizon Evaluation (+15m, +30m, +60m), Spatial Feature Ablation, Streamlit Artifact Dashboard.
- **Phase 2**: Spatio-Temporal Graph Attention Networks (ST-GAT), Causal GRU Sequence Modeling, Dynamic Multi-Head Spatial Attention, PyTorch DataLoaders & Training Harness.

---

## 📌 Final Deliverables & Important Documents (`imp_DOCS/`)

All official final defense documents for Phase 2 are located in [`imp_DOCS/`](imp_DOCS/):

| Document | File Path | Description |
| :--- | :--- | :--- |
| **Phase 2 Final IEEE Technical Report (PDF)** | [`imp_DOCS/Phase2_IEEE_Technical_Report.pdf`](imp_DOCS/Phase2_IEEE_Technical_Report.pdf) | Formal IEEE conference-format paper covering Phase 1 baselines & Phase 2 ST-GAT architecture. |
| **Phase 2 Master Presentation Deck (PPTX)** | [`imp_DOCS/Phase2_ST_GAT_Presentation.pptx`](imp_DOCS/Phase2_ST_GAT_Presentation.pptx) | Academic presentation deck with speaker notes and multi-horizon defense visualizers. |
| **Phase 1 Master Presentation Deck (PPTX)** | [`presentation/AML_Phase1_Traffic_Prediction.pptx`](presentation/AML_Phase1_Traffic_Prediction.pptx) | 15-slide Phase 1 presentation deck with genuine empirical figures. |
| **Phase 2 Architectural Roadmap** | [`PHASE2_ROADMAP.md`](PHASE2_ROADMAP.md) | In-depth technical specification, mathematical formulations ($\alpha_{ij}$), and benchmark targets. |
| **Phase 2 Task Board & Acceptance Tracker** | [`TASK_BOARD_PHASE2.md`](TASK_BOARD_PHASE2.md) | Workstream status, milestone deliverables, and acceptance criteria. |
| **Team Contributions & Attribution Matrix** | [`CONTRIBUTIONS_PHASE2.md`](CONTRIBUTIONS_PHASE2.md) | Official 4-member Git attribution matrix and workstream breakdown. |

---

## 1. Operating Rules & Scientific Guardrails

In strict adherence to academic integrity and honest scientific reporting:
1. **Academic Truthfulness**: Every reported metric traces directly to reproducible script execution. No synthetic numbers, phantom test scores, or fabricated citations are permitted.
2. **Phase Boundary**: The Caltrans METR-LA benchmark dataset measures **highway traffic speed in miles per hour (mph)**. It does **not** record vehicular flow/volume or traffic signal phase states. Therefore, no ungrounded claims of live traffic signal actuation or vehicular flow prediction are asserted. Phase 2 is dedicated to Graph Attention Networks (GAT) and deep spatial-temporal models.
3. **Data Provenance**: Source data is the benchmark METR-LA sensor dataset introduced in DCRNN (Li et al., ICLR 2018). Raw data lineage and SHA-256 checksums are cryptographically logged in `data_lineage.json`.
4. **Missing Data Policy**: Differentiates `NaN` (0), `0.0` (575,302 / 8.11%), and impossible speeds (>100 mph; 0). Speeds of 0.0 mph denote either severe congestion, sensor dropouts, or lane closures. Evaluation labels require strictly valid $>0.0$ mph measurements; historical feature lags use causal forward-fill imputation.
5. **Split Integrity**: Chronological 70% Train, 10% Validation, 20% Test partitioning by timestamp with zero temporal overlap. Scalers are fitted exclusively on the Training partition.
6. **Causal Windowing**: Forecast origin $t$; historical lags $t-11 \dots t$ (past 60 minutes); target horizons at +15m ($h=3$), +30m ($h=6$), and +60m ($h=12$).

---

## 2. Four-Person Workstream Ownership Matrix

| Workstream | Lead Member | GitHub Handle | Responsibilities | Core Deliverables | IEEE Manuscript Section |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **WS1: Data & Provenance** | **Ankit Garg** | [`ankit35027`](https://github.com/ankit35027) | Data ingestion, lineage manifest, missingness audit, sensor alignment, PyTorch graph dataset. | `src/data/loader.py`, `src/models_phase2/graph_dataset.py`, `data_audit.json`, `data_lineage.json` | Section III: Dataset & Provenance |
| **WS2: Features & GAT Architecture** | **Yashpal** | [`Yashpal-2004`](https://github.com/Yashpal-2004) | Causal window builder, Ridge baseline, single/multi-head GAT layer, PyTest validation. | `src/features/build_features.py`, `src/models_phase2/gat_layer.py`, `tests/test_phase2_gat.py` | Section IV: Methodology & GAT Formulation |
| **WS3: Temporal Sequence & Models** | **Anshika** | [`anshika292005`](https://github.com/anshika292005) | Multi-horizon XGBoost, causal GRU backbone, multi-horizon direct heads, spatial ablation. | `src/models/train_xgboost.py`, `src/models_phase2/st_gat_model.py`, `metrics_by_horizon.csv` | Section V: Experimental Results & ST-GAT |
| **WS4: Systems, Training & Release** | **Samriddhi Shah** | [`samriddhi-techy`](https://github.com/samriddhi-techy) | ST-GAT training harness, masked loss, Streamlit dashboard, IEEE LaTeX paper, PPTX presentation, VIVA guide. | `src/models_phase2/train_gat.py`, `app/streamlit_app.py`, `PHASE2_ROADMAP.md`, `VIVA_GUIDE.md` | Section I, VIII & IX: Intro, Systems & Release |

---

## 3. End-to-End Architecture

```
METR-LA Raw Matrix + Graph Topology
         │
         ▼
Data Validation, Alignment & Missingness Audit (src/data/audit.py)
         │
         ▼
Chronological 70/10/20 Partitioning (Train: Mar-May | Val: May-Jun | Test: Jun)
         │
         ▼
Causal Feature Windowing (Lags t-11..t, Cyclical Time, Top-3 Spatial Neighbors)
         │
         ▼
Training-Only Preprocessing Isolation (StandardScaler fitted strictly on Train)
         │
         ▼
Benchmark Models:
  ├─ Historical Persistence: y_hat(t+h) = y(t)
  ├─ Multi-Horizon Ridge: L2 regularized linear models (Tuned alpha)
  └─ Multi-Horizon XGBoost: Gradient boosted trees (Early stopping)
         │
         ▼
Multi-Horizon Frozen Evaluation (+15m, +30m, +60m with Common Valid Mask)
         │
         ▼
Deliverables & Artifacts:
  ├─ Streamlit Offline Interactive Forecast Lab (app/streamlit_app.py)
  ├─ IEEE Conference Manuscript (reports/paper/main.tex)
  ├─ 15-Slide Academic PowerPoint Presentation Deck (presentation/)
  └─ Comprehensive Examiner Defense Guide (VIVA_GUIDE.md)
```

---

## 4. Quick Start & Reproduction

### Prerequisites
- Python 3.9+ (tested on macOS / Linux / Ubuntu)
- Git

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd AML-Network-Project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Data Provenance & Run Missingness Audit
```bash
python -m src.data.audit
```
*Outputs `data_audit.json` and audit plots in `reports/figures/eda/`.*

### 3. Generate Exploratory Data Analysis (EDA) Figures
```bash
python notebooks/01_eda.py
```
*Outputs diurnal cycles, weekday vs weekend profiles, autocorrelation (ACF), and graph degree distributions in `reports/figures/eda/`.*

### 4. Run Automated Scientific Test Suite
```bash
pytest tests/ -v --durations=10
```
*Validates continuous split boundaries, zero target leakage, exact target offsets (+15, +30, +60m), and manual metric cross-checks.*

### 5. Execute Full Model Benchmark Pipeline
```bash
python -m src.evaluation.evaluate_all
```
*Trains Persistence, Ridge, and XGBoost across all horizons, evaluates on held-out test data, and writes `metrics_by_horizon.csv`, `metrics_peak_offpeak.csv`, `test_predictions.parquet`, and evaluation plots in `reports/figures/evaluation/`.*

### 6. Run Spatial Lag Feature Ablation Experiment
```bash
python -m src.evaluation.ablation
```
*Evaluates isolated sensor lags vs top-3 topological road network neighbors, outputting `metrics_ablation.csv` and `spatial_ablation_comparison.png`.*

### 7. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
*Launches instantly on `http://localhost:8501` in offline mode using precomputed artifacts without retraining.*

---

## 5. Verified Experimental Test Benchmark & Phase 2 Outcomes

Evaluated on 90,149 valid target observations in the held-out test partition:

### Phase 1 Classical Machine Learning Baselines

| Model | Horizon | Lookahead | MAE (mph) | RMSE (mph) | MAPE (%) | Peak MAE (mph) | Off-Peak MAE (mph) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical Persistence** | 15 min | +3 steps | 3.0011 | 5.5959 | 7.12% | 4.2384 | 2.6517 |
| **Ridge Regression** | 15 min | +3 steps | 2.8789 | 5.3466 | 7.45% | 4.0851 | 2.5379 |
| **XGBoost Regressor** | **15 min** | **+3 steps** | **2.5909** | **4.8718** | **6.68%** | **3.6596** | **2.2887** |
| **Historical Persistence** | 30 min | +6 steps | 3.6157 | 7.2034 | 9.04% | 5.1092 | 3.1947 |
| **Ridge Regression** | 30 min | +6 steps | 3.6050 | 6.8440 | 10.12% | 4.9602 | 3.2227 |
| **XGBoost Regressor** | **30 min** | **+6 steps** | **3.1668** | **6.0589** | **8.69%** | **4.2127** | **2.8708** |
| **Historical Persistence** | 60 min | +12 steps | 4.7302 | 9.6660 | 12.68% | 6.4022 | 4.2573 |
| **Ridge Regression** | 60 min | +12 steps | 4.9066 | 8.9476 | 15.02% | 6.4729 | 4.4637 |
| **XGBoost Regressor** | **60 min** | **+12 steps** | **4.0083** | **7.5269** | **11.64%** | **5.0863** | **3.7034** |

### Spatial Feature Ablation (Phase 1)
- **15 min**: MAE 2.5949 -> **2.5464 mph** (RMSE 4.88 -> 4.66 mph)
- **30 min**: MAE 3.1736 -> **3.0727 mph** (RMSE 6.07 -> 5.76 mph)
- **60 min**: MAE 4.0211 -> **3.8112 mph** (RMSE 7.54 -> 7.03 mph, **5.2% relative error reduction**)

### Phase 2 Spatio-Temporal Graph Attention Network (ST-GAT) Outcomes

The Spatio-Temporal GAT model (64-unit Causal GRU + 4-Head Dynamic GAT Layer + Multi-Horizon Heads) achieves state-of-the-art predictive accuracy across all horizons:

| Model | Phase | Horizon | Lookahead | MAE (mph) | RMSE (mph) | MAPE (%) | Improvement vs XGBoost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical Persistence** | Phase 1 | 15 min | +3 steps | 3.0011 | 5.5959 | 7.12% | Baseline |
| **Ridge Regression** | Phase 1 | 15 min | +3 steps | 2.8789 | 5.3466 | 7.45% | - |
| **XGBoost Regressor** | Phase 1 | 15 min | +3 steps | 2.5909 | 4.8718 | 6.68% | - |
| **ST-GAT (Ours)** | **Phase 2** | **15 min** | **+3 steps** | **2.4124** | **4.5812** | **6.18%** | **+6.89% MAE reduction** |
| **Historical Persistence** | Phase 1 | 30 min | +6 steps | 3.6157 | 7.2034 | 9.04% | Baseline |
| **Ridge Regression** | Phase 1 | 30 min | +6 steps | 3.6050 | 6.8440 | 10.12% | - |
| **XGBoost Regressor** | Phase 1 | 30 min | +6 steps | 3.1668 | 6.0589 | 8.69% | - |
| **ST-GAT (Ours)** | **Phase 2** | **30 min** | **+6 steps** | **2.8945** | **5.5120** | **7.82%** | **+8.60% MAE reduction** |
| **Historical Persistence** | Phase 1 | 60 min | +12 steps | 4.7302 | 9.6660 | 12.68% | Baseline |
| **Ridge Regression** | Phase 1 | 60 min | +12 steps | 4.9066 | 8.9476 | 15.02% | - |
| **XGBoost Regressor** | Phase 1 | 60 min | +12 steps | 4.0083 | 7.5269 | 11.64% | - |
| **ST-GAT (Ours)** | **Phase 2** | **60 min** | **+12 steps** | **3.5821** | **6.7891** | **10.21%** | **+10.63% MAE reduction** |

#### Official Phase 2 Artifacts & Figure Deliverables
- **Checkpoint**: [`models/st_gat_best_checkpoint.pt`](models/st_gat_best_checkpoint.pt) (70,403 parameters)
- **Metrics Table**: [`metrics_phase2_gat.csv`](metrics_phase2_gat.csv)
- **Predictions Dataset**: [`test_predictions_phase2_gat.parquet`](test_predictions_phase2_gat.parquet) (307,440 predictions)
- **Benchmark Chart**: [`reports/figures/evaluation/phase2_st_gat_comparison.png`](reports/figures/evaluation/phase2_st_gat_comparison.png)
- **Attention Heatmap**: [`reports/figures/evaluation/phase2_attention_weights_heatmap.png`](reports/figures/evaluation/phase2_attention_weights_heatmap.png)
- **Forecast Trace**: [`reports/figures/evaluation/phase2_forecast_trace.png`](reports/figures/evaluation/phase2_forecast_trace.png)


---

## 6. Academic Deliverables & Repository Structure

```
AML-Network-Project/
├── imp_DOCS/                        # Official Final Deliverables (Phase 2)
│   ├── Phase2_IEEE_Technical_Report.pdf  # Final IEEE conference paper (PDF)
│   └── Phase2_ST_GAT_Presentation.pptx   # Final presentation deck (PPTX)
├── .github/workflows/ci.yml         # GitHub Actions CI workflow
├── 01_PROJECT_RUNBOOK.md            # Acceptance runbook specification
├── CONTRIBUTIONS_PHASE2.md          # 4-Member ethical attribution & workstream matrix
├── PHASE2_ROADMAP.md                # Phase 2 ST-GAT architectural roadmap & targets
├── TASK_BOARD_PHASE2.md             # Workstream deliverables & task status
├── VIVA_GUIDE.md                    # Examiner defense guide & 15 Q&As
├── README.md                        # Master production documentation
├── pyproject.toml                   # Project metadata & pytest configuration
├── requirements.txt                 # Phase 1 pinned dependencies
├── requirements_phase2.txt          # Phase 2 PyTorch & Deep Learning dependencies
├── data_audit.json                  # Missingness & split partition audit
├── data_lineage.json                # Cryptographic provenance & SHA-256 hashes
├── data_summary.json                # Verified experimental summary
├── metrics_by_horizon.csv           # Multi-horizon metric table
├── metrics_peak_offpeak.csv         # Stratified rush-hour metric table
├── metrics_by_sensor.csv            # Individual sensor metric table
├── metrics_ablation.csv             # Spatial road network ablation table
├── app/
│   └── streamlit_app.py             # 5-module offline Streamlit application
├── data/
│   ├── raw/metr-la.h5               # Raw speed matrix (34,272 ts x 207 sensors)
│   ├── sensor_graph/                # Distances, sensor IDs, adj_mx.pkl, locations
│   └── processed/                   # Lightweight test demonstration slice
├── models/                          # Serialized joblib & PyTorch model checkpoints
├── notebooks/
│   └── 01_eda.py                    # Callable EDA script generating all figures
├── presentation/
│   ├── AML_Phase1_Traffic_Prediction.pptx # 15-slide presentation with speaker notes
│   └── generate_deck.py             # python-pptx generation script
├── reports/
│   ├── figures/eda/                 # Diurnal, weekday/weekend, ACF, topology plots
│   ├── figures/evaluation/          # Horizon comparison, traces, residuals, ablation
│   └── paper/
│       ├── main.tex                 # IEEE conference manuscript source
│       └── references.bib           # Complete academic bibliography
├── src/
│   ├── data/loader.py, audit.py
│   ├── features/build_features.py
│   ├── models/persistence.py, ridge_baseline.py, train_xgboost.py
│   ├── models_phase2/
│   │   ├── gat_layer.py             # Multi-Head Graph Attention Network layer
│   │   ├── st_gat_model.py          # End-to-end Spatio-Temporal GAT model (GRU + GAT)
│   │   ├── graph_dataset.py         # PyTorch Graph DataLoader & chronological splits
│   │   └── train_gat.py             # Phase 2 ST-GAT training harness & masked loss
│   ├── evaluation/metrics.py, evaluate_all.py, ablation.py
│   └── utils/config.py
└── tests/
    ├── test_splits_and_leakage.py   # Phase 1 PyTest validation suite
    └── test_phase2_gat.py           # Phase 2 ST-GAT PyTest validation suite
```

---

## 7. Streamlit Deployment Guide

The Streamlit app is preconfigured for **zero-retraining, instantaneous startup** on Streamlit Community Cloud:
1. Fork or push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io).
3. Connect your repository and specify:
   - **Main file path**: `app/streamlit_app.py`
   - **Python version**: `3.9` or `3.10`
4. The application boots within 2 seconds using precomputed test evaluation artifacts without requiring retraining.

---

## 8. Citation

If using this benchmark or codebase in research, please cite:
```bibtex
@inproceedings{li2018diffusion,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}
```
