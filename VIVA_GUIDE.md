# Comprehensive Viva Defense & Examiner Q&A Guide

**Project Title**: Historical METR-LA Traffic Speed Forecasting (Phase 1 Baseline & Classical ML System)  
**Parent Course**: Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models  
**Presentation / Viva Duration**: 7-Minute Defense + Examiner Q&A  

---

## 1. Executive Summary & Scientific Scope

### The Project Mandate
This project investigates short- and medium-term highway traffic speed prediction on the Caltrans Los Angeles freeway network (METR-LA). The parent course title references Graph Attention Networks (GAT) and flow models; **Phase 1** establishes the rigorous, academically truthful classical machine learning benchmarks (Historical Persistence, $L_2$-regularized Ridge Regression, and Gradient Boosted Trees XGBoost).

### Core Boundaries & Academic Integrity
1. **Measurement Unit**: Speeds in miles per hour (mph) collected from inductive loop detectors.
2. **Phase Boundary**: The METR-LA benchmark contains **only speed measurements**. It does **not** contain vehicular flow/volume or traffic signal phase states. Therefore, no ungrounded claims of live signal control or unmeasured vehicular flow are asserted.
3. **Causal Windowing**: Every feature is strictly computed from elapsed history $t-11 \dots t$ (past 60 minutes) to predict multi-step forward horizons $t+3$ (+15 min), $t+6$ (+30 min), and $t+12$ (+60 min).
4. **Chronological Partitioning**: Strict 70% Train, 10% Validation, 20% Test chronological split with zero temporal overlap. Scalers are fitted exclusively on the Training partition.

---

## 2. Mathematical Formulations & Core Equations

### 1. Causal Feature Windowing & Temporal Cyclical Encodings
For observation timestamp $t$, the input feature vector $\mathbf{x}_t \in \mathbb{R}^{21}$ (or $\mathbb{R}^{30}$ with spatial neighbors) comprises:
$$\mathbf{x}_t = \left[ y_{t-11}, y_{t-10}, \dots, y_t, \, \mu_{\text{hist}}, \, \sigma_{\text{hist}}, \, \min_{\text{hist}}, \, \max_{\text{hist}}, \, \Delta_{\text{trend}}, \, \sin(\text{TOD}), \, \cos(\text{TOD}), \, \sin(\text{DOW}), \, \cos(\text{DOW}) \right]$$
Where continuous cyclical time encodings prevent midnight boundary discontinuity:
$$\sin(\text{TOD}) = \sin\left(\frac{2\pi \cdot (\text{hour} \times 60 + \text{minute})}{1440}\right), \quad \cos(\text{TOD}) = \cos\left(\frac{2\pi \cdot (\text{hour} \times 60 + \text{minute})}{1440}\right)$$

### 2. Historical Persistence Benchmark
$$\hat{y}_{t+h} = y_t, \quad \forall h \in \{3, 6, 12\}$$
Assumes the current measured speed at origin $t$ persists unchanged over horizon $h$.

### 3. Ridge Regression ($L_2$ Regularization)
Direct horizon-specific linear models minimizing penalized squared loss:
$$\mathcal{L}_{\text{Ridge}}(\mathbf{w}_h) = \sum_{i=1}^{N_{\text{train}}} (y_{i,t+h} - \mathbf{w}_h^T \mathbf{x}_{i,t})^2 + \alpha_h \|\mathbf{w}_h\|_2^2$$
Where $\alpha_h$ is tuned on validation MAE ($\alpha_{15m}=100.0, \alpha_{30m}=0.1, \alpha_{60m}=0.1$).

### 4. XGBoost Objective (Second-Order Taylor Approximation)
Ensemble of $M$ decision trees minimizing regularized loss:
$$\mathcal{L}^{(m)} \approx \sum_{i=1}^N \left[ g_i f_m(\mathbf{x}_i) + \frac{1}{2} h_i f_m^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
Where $g_i = \partial_{\hat{y}^{(m-1)}} l(y_i, \hat{y}^{(m-1)})$ and $h_i = \partial^2_{\hat{y}^{(m-1)}} l(y_i, \hat{y}^{(m-1)})$.

### 5. Evaluation Metrics & $\epsilon$-Masked MAPE Policy
$$\text{MAE} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} |y_i - \hat{y}_i|, \quad \text{RMSE} = \sqrt{\frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} (y_i - \hat{y}_i)^2}$$
$$\text{MAPE} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \frac{|y_i - \hat{y}_i|}{\max(y_i, \epsilon)} \times 100\%$$
Where $\epsilon = 2.0$ mph ensures numerical stability and prevents extreme ratio inflation when speeds approach 0 mph during traffic jams.

### 6. Phase 2 Graph Attention Coefficient (GAT Transition)
$$\alpha_{ij}^{(t)} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i^{(t)} \parallel \mathbf{W}h_j^{(t)}]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i^{(t)} \parallel \mathbf{W}h_k^{(t)}]\right)\right)}$$

---

## 3. Verified Experimental Results Summary

Evaluated on 90,149 valid target observations in the held-out test split:

| Model | Horizon | MAE (mph) | RMSE (mph) | MAPE (%) | Peak MAE | Off-Peak MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Persistence** | 15 min | 3.0011 | 5.5959 | 7.12% | 4.2384 | 2.6517 |
| **Ridge** | 15 min | 2.8789 | 5.3466 | 7.45% | 4.0851 | 2.5379 |
| **XGBoost** | **15 min** | **2.5909** | **4.8718** | **6.68%** | **3.6596** | **2.2887** |
| **Persistence** | 30 min | 3.6157 | 7.2034 | 9.04% | 5.1092 | 3.1947 |
| **Ridge** | 30 min | 3.6050 | 6.8440 | 10.12% | 4.9602 | 3.2227 |
| **XGBoost** | **30 min** | **3.1668** | **6.0589** | **8.69%** | **4.2127** | **2.8708** |
| **Persistence** | 60 min | 4.7302 | 9.6660 | 12.68% | 6.4022 | 4.2573 |
| **Ridge** | 60 min | 4.9066 | 8.9476 | 15.02% | 6.4729 | 4.4637 |
| **XGBoost** | **60 min** | **4.0083** | **7.5269** | **11.64%** | **5.0863** | **3.7034** |

### Spatial Feature Ablation (Isolated Lags vs. 1st-Order Topological Neighbors)
- **15 min**: MAE 2.5949 -> **2.5464 mph** (RMSE 4.88 -> 4.66 mph)
- **30 min**: MAE 3.1736 -> **3.0727 mph** (RMSE 6.07 -> 5.76 mph)
- **60 min**: MAE 4.0211 -> **3.8112 mph** (RMSE 7.54 -> 7.03 mph, **5.2% relative error reduction**)

---

## 4. Top 15 Likely Examiner Questions & Model Defense Answers

### Q1: Why did you choose classical ML instead of directly implementing Graph Attention Networks (GAT)?
**Examiner's intent**: Testing whether the student skipped ahead without building an empirical baseline.  
**Answer**:  
*"In applied machine learning research, jumping directly into complex deep neural architectures without an empirical lower-bound benchmark leads to uncalibrated claims of superiority. Classical baselines—specifically Historical Persistence, regularized linear regression (Ridge), and gradient boosted trees (XGBoost)—provide the definitive performance floor. Our Phase 1 results establish that XGBoost achieves an MAE of 2.59 mph at 15m. In Phase 2, our Graph Attention Network must demonstrate statistically significant improvements over this 2.59 mph baseline to justify its computational complexity."*

### Q2: Why is Ridge Regression worse than Persistence at the 60-minute horizon?
**Examiner's intent**: Testing understanding of linear model limitations and collinearity.  
**Answer**:  
*"At $h=12$ steps (+60 min), traffic conditions have largely decoupled from the immediate past $t-11 \dots t$. The 12 historical lags are highly collinear. A linear hyper-plane attempts to extrapolate global weights across this collinear feature space, amplifying variance on non-linear congestion shifts. In contrast, Persistence simply outputs $y_t$, which remains bounded within physical speed limits (e.g. 55-65 mph), avoiding the extrapolation errors that penalize Ridge."*

### Q3: Why did you not treat all 0.0 mph values as missing/NaN?
**Examiner's intent**: Testing data engineering and traffic domain awareness.  
**Answer**:  
*"In Caltrans PeMS loop detectors, 0.0 mph can represent genuine bumper-to-bumper standstill traffic during a severe incident, in addition to hardware dropouts or lane closures. Blindly replacing all zeros with NaN would discard real congestion events and bias the dataset toward high speeds. Instead, we performed an explicit audit: for evaluation, labels must be strictly valid ($>0.0$ mph) to allow fair comparison and defined MAPE; for historical feature lags, we applied causal forward-fill imputation using only elapsed past readings."*

### Q4: How do you guarantee there is zero data leakage in your feature pipeline?
**Examiner's intent**: Checking cross-validation and windowing rigor.  
**Answer**:  
*"We implemented three strict architectural barriers verified by our automated pytest suite:  
1. **Chronological split**: 70% Train, 10% Val, 20% Test by timestamp; no random shuffling.  
2. **Causal windowing**: For origin $t$, feature columns are indexed strictly from $t-11$ to $t$. Targets are offsets $t+3, t+6, t+12$.  
3. **Training-only scaler fit**: The StandardScaler is fitted exclusively on the Training partition ($N=359,505$ samples). Validation and Test sets are transformed using frozen training means and standard deviations, verified by `test_scaler_fit_strictly_on_train_only`."*

### Q5: Why is MAPE computed with an $\epsilon$-threshold?
**Examiner's intent**: Probing metric edge cases and division by zero.  
**Answer**:  
*"The standard MAPE formula divides by ground truth speed $y_t$: $|y - \hat{y}| / y$. When highway speeds drop to 1 or 2 mph during heavy congestion, a small absolute error of 2 mph produces an astronomical percentage error of 100% or 200%, severely distorting the metric. By setting $\epsilon = 2.0$ mph ($\max(y_t, 2.0)$), we prevent division by zero and extreme skew while preserving metric monotonicity."*

### Q6: What did your spatial lag ablation prove?
**Examiner's intent**: Testing the transition from tabular ML to graph-based thinking.  
**Answer**:  
*"Our ablation compared an isolated sensor model (21 features) against a model incorporating lags from the top-3 nearest road network neighbors from the adjacency matrix (30 features). Incorporating topological neighbors reduced 60-minute MAE from 4.02 to 3.81 mph. This proves that congestion shockwaves physically propagate along freeway corridors over a 60-minute window, providing direct empirical motivation for Phase 2 Spatio-Temporal Graph Attention Networks."*

### Q7: If you only evaluated 15 pilot sensors, why not all 207?
**Examiner's intent**: Checking honesty about experimental scope.  
**Answer**:  
*"To guarantee rapid iteration, thorough hyperparameter validation, and full test evaluation without crashing local memory, we selected a representative 15-sensor pilot spanning major corridors (I-5, I-10, I-405, SR-110). We disclose this prominently in our report, app, and code manifest. We do not claim 207-sensor coverage for this experiment. However, our modular data loader and window builder scale seamlessly to all 207 sensors when high-performance compute clusters are provisioned in Phase 2."*

### Q8: What features were most influential in the XGBoost models?
**Examiner's intent**: Checking model interpretability.  
**Answer**:  
*"Feature importance analysis via Gini gain reveals that the most recent speed lag $y_t$ (column 11) is overwhelmingly dominant at +15m. However, as the horizon extends to +60m, the relative importance of $y_t$ decays, while the historical rolling mean $\mu_{\text{hist}}$ and cyclical time-of-day features ($\sin(\text{TOD}), \cos(\text{TOD})$) increase substantially, reflecting diurnal traffic rhythm."*

### Q9: Why does error spike during rush hours (peak vs. off-peak)?
**Examiner's intent**: Testing understanding of traffic flow dynamics.  
**Answer**:  
*"During off-peak hours (e.g. midday or night), traffic operates in a stable free-flow regime where speeds hover consistently around 60-65 mph, making forecasting relatively straightforward (MAE ~2.29 mph). During peak hours, traffic density approaches road capacity, inducing non-linear phase transitions: slight perturbations cause stop-and-go shockwaves, inflating error to 3.66 mph at 15m and 5.09 mph at 60m."*

### Q10: How does your Streamlit application handle performance without crashing?
**Examiner's intent**: Evaluating software engineering and caching.  
**Answer**:  
*"Our Streamlit app is designed offline-first. It caches precomputed evaluation tables and a 4.7MB test prediction parquet file. When a user changes sensor or origin time, the app performs instant in-memory filtering rather than retraining models on the fly. This enables cold-start startup in under 2 seconds and guarantees identical, deterministic results."*

### Q11: What is the exact difference between Phase 1 and Phase 2 in your project?
**Examiner's intent**: Ensuring alignment with course syllabus.  
**Answer**:  
*"Phase 1 delivers the classical machine learning foundation: causal windowing, missingness auditing, Historical Persistence, Ridge, and XGBoost benchmarks, evaluated on multi-horizon speed targets. Phase 2 introduces deep geometric deep learning: Graph Attention Networks (GAT) to learn dynamic, time-varying edge attention weights between sensors, coupled with temporal convolutions (TCN/GRU) to jointly model spatio-temporal dependencies."*

### Q12: Why can't you claim live traffic signal control with METR-LA?
**Examiner's intent**: Testing academic honesty against the course title.  
**Answer**:  
*"The course title references signal control, but the benchmark METR-LA dataset contains strictly freeway loop detector speeds. It does not provide municipal traffic signal timings, queue lengths, or intersection signal phases. Claiming signal actuation on data that lacks signal states is scientifically dishonest. In Phase 1, we explicitly document this boundary: we predict highway traffic speeds only."*

### Q13: How did you handle hyperparameter tuning for Ridge and XGBoost?
**Examiner's intent**: Verifying validation integrity.  
**Answer**:  
*"All hyperparameter tuning was conducted strictly on the chronological Validation split (May 23 to June 4, 2012; 51,060 samples). For Ridge, we evaluated candidate alphas [0.1, 1.0, 10.0, 50.0, 100.0] on validation MAE. For XGBoost, we trained with early stopping (patience = 25 rounds) monitoring validation MAE. The final test set was evaluated once with frozen parameters."*

### Q14: How does your team divide responsibilities among the four members?
**Examiner's intent**: Assessing individual contribution and teamwork.  
**Answer**:  
*"We established four distinct workstreams documented in `CONTRIBUTIONS.md` and `TASK_BOARD.md`:  
- **Member 1 (WS1)**: Dataset ingestion, lineage manifest, missingness audit, EDA.  
- **Member 2 (WS2)**: Causal window builder, split validation, Persistence/Ridge baselines, pytest suite.  
- **Member 3 (WS3)**: XGBoost regressors, evaluation metrics engine, peak/off-peak analysis, spatial ablation.  
- **Member 4 (WS4)**: Streamlit app, IEEE LaTeX paper, PPTX presentation deck, and systems integration."*

### Q15: What is the single biggest lesson learned from Phase 1?
**Examiner's intent**: Open-ended critical reflection.  
**Answer**:  
*"The biggest insight is that traffic is inherently spatial over medium horizons. While an isolated sensor model performs reasonably at 15 minutes, predicting 60 minutes ahead requires road network topology. Adding just 3 nearest neighbors cut 60m error by over 5%, proving that spatial graph modeling is essential for long-range forecasting."*

---

## 5. Seven-Minute Viva Presentation Script & Choreography

| Minute | Presenter | Slide | Key Talking Points |
| :--- | :--- | :--- | :--- |
| **0:00 - 1:00** | **Member 1** | Slides 1 - 2 | Introduce project title, four-person ownership, course alignment, and scientific guardrails (speed prediction only; no false signal claims). |
| **1:00 - 2:00** | **Member 1** | Slides 3 - 4 | Data provenance: 207 sensors, 4 months, SHA-256 hash. Missingness audit: 8.11% zeros, label policy (>0.0 valid for eval, causal forward-fill for features). |
| **2:00 - 3:15** | **Member 2** | Slides 5 - 7 | Causal window builder ($t-11 \dots t \to t+3, 6, 12$), strict 70/10/20 chronological split, training-only scaler fit. Highlight 6 passing pytest unit tests. |
| **3:15 - 4:45** | **Member 3** | Slides 8 - 11 | Baseline models (Persistence, Ridge, XGBoost). Present test metrics table (XGBoost 2.59, 3.17, 4.01 mph). Show 24h forecast trace and peak vs off-peak inflation. |
| **4:45 - 5:45** | **Member 3** | Slide 12 | Present Spatial Lag Ablation: top-3 neighbors reduce 60m MAE from 4.02 to 3.81 mph, proving spatial shockwave transmission. |
| **5:45 - 6:30** | **Member 4** | Slide 13 | Demonstrate Streamlit interactive dashboard: offline architecture, instant cold-start, live walkthrough of Forecast Lab and sensor map. |
| **6:30 - 7:00** | **Member 4** | Slides 14 - 15 | Roadmap to Phase 2 (dynamic attention in GAT to beat 2.59 mph baseline). Conclude and invite questions from the committee. |
