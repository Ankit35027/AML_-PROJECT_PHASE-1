# Contributions & Team Ownership Matrix

**Project Title**: Historical METR-LA Traffic Speed Forecasting (Phase 1 Baseline & Classical ML System)  
**Parent Course**: Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models  
**Institutional Affiliation**: College of Engineering / Department of Computer Science & AI  

---

## 1. Ethical Attribution & Git Collaboration Principles

In compliance with strict academic integrity standards:
1. **No Fabricated Commits or Identities**: Every code change, review, and experimental analysis must reflect genuine member participation. Single automation tools or pairing assistants must never spoof member email addresses or author identities.
2. **Co-Authorship Trailers**: If members pair-program on a module or commit jointly, standard Git trailers (`Co-authored-by: Name <email>`) must be utilized with explicit consent.
3. **Reproducibility Mandate**: No member may report or incorporate synthetic numbers or fabricated metrics into papers, slides, or software. All claims must execute deterministically from stored scripts and data manifests.

---

## 2. Four-Person Workstream Division

| Workstream | Member Designation | Focus Area | Core Modules & Deliverables | Primary Manuscript Section |
| :--- | :--- | :--- | :--- | :--- |
| **WS1: Data & Provenance** | Member 1 | Data Ingestion, Lineage, Missingness Audit & EDA | `src/data/loader.py`, `src/data/audit.py`, `notebooks/01_eda.py`, `data_audit.json`, `data_lineage.json`, EDA Figures | Section III: Dataset & Provenance Analysis |
| **WS2: Feature Engineering & Baselines** | Member 2 | Causal Windowing, Historical Persistence, Ridge Baseline & Test Suite | `src/features/build_features.py`, `src/models/persistence.py`, `src/models/ridge_baseline.py`, `tests/` | Section IV: Causal Windowing & Benchmark Formulation |
| **WS3: Non-Linear ML & Evaluation** | Member 3 | Multi-Horizon XGBoost, Hyperparameter Tuning, Spatial Ablation & Metrics | `src/models/train_xgboost.py`, `src/evaluation/metrics.py`, `src/evaluation/ablation.py`, `metrics_by_horizon.csv` | Section V: Empirical Results & Spatial Ablation |
| **WS4: Systems, Dashboard & Presentation** | Member 4 | Interactive Streamlit App, Paper Compilation, Slide Deck & Viva Guide | `app/streamlit_app.py`, `reports/paper/`, `presentation/`, `VIVA_GUIDE.md`, `README.md` | Section I & VI: Introduction, Deployment & Future Work |

---

## 3. Workstream Detailed Responsibilities

### WS1: Data & Provenance (Member 1)
- **Data Ingestion**: Verify raw HDF5 speed matrix download from benchmark sources without data distortion.
- **Data Lineage Manifest**: Maintain `data_lineage.json` with file paths, sizes, and SHA-256 cryptographic hashes.
- **Missingness Audit**: Audit `0.0` readings vs genuine sensor dropouts and physical anomalies (>100 mph).
- **Exploratory Data Analysis**: Implement reproducible plotting scripts for 24-hour diurnal patterns, weekday vs weekend profiles, speed histograms, and autocorrelation functions (ACF).

### WS2: Feature Engineering & Baselines (Member 2)
- **Causal Window Builder**: Build `build_examples()` using strictly historical observations ($t-11 \dots t$) and multi-step targets ($t+3, t+6, t+12$).
- **Partitioning & Preprocessing**: Enforce chronological 70/10/20 train/validation/test splits; ensure scaler fit is strictly isolated to training data.
- **Statistical Baselines**: Implement Historical Persistence (last-value) and L2-regularized Ridge Regression.
- **Automated Validation**: Author unit tests verifying causal boundaries, off-by-one offsets, and absence of target leakage.

### WS3: Non-Linear ML & Evaluation (Member 3)
- **Gradient Boosted Decision Trees**: Train horizon-specific XGBoost regressors on the pilot sensor cohort.
- **Evaluation Suite**: Implement standardized multi-horizon metric functions (MAE, RMSE, masked $\epsilon$-MAPE) with identical evaluation masks.
- **Spatial Feature Ablation**: Implement and evaluate 1st-order road network topological neighbor lag features vs isolated sensor lags.
- **Stratified Analysis**: Calculate peak (rush-hour) vs off-peak and individual sensor error breakdowns.

### WS4: Systems, Dashboard & Presentation (Member 4)
- **Streamlit Application**: Develop multi-page offline interactive dashboard featuring Overview, Dataset Explorer, Forecast Lab, Benchmark Metrics, and Road Network Preview.
- **Paper & Presentation**: Author IEEE-format conference manuscript in LaTeX and 15-slide PowerPoint deck with genuine experimental plots.
- **Viva Preparation**: Compile `VIVA_GUIDE.md` detailing theoretical equations, architectural trade-offs, and examiner defense scenarios.
- **Release & CI**: Maintain GitHub Actions CI workflow, `requirements.txt`, `pyproject.toml`, and comprehensive `README.md`.
