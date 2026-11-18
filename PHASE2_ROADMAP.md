# Phase 2 Research Roadmap & Architecture Specification
## Spatio-Temporal Graph Attention Networks (ST-GAT) for Highway Traffic Speed Forecasting

**Parent Course Title**: *Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models*  
**Phase Transition**: Moving from Phase 1 Classical Machine Learning (Persistence, Ridge, XGBoost) to Phase 2 Deep Geometric Deep Learning (Spatio-Temporal Graph Attention Networks).

---

## 1. Executive Summary: The Phase 1 to Phase 2 Bridge

### What Phase 1 Proved
In Phase 1, our team established the rigorous empirical baseline on the METR-LA benchmark dataset:
- **XGBoost achieved state-of-the-art classical performance**:
  - **15-minute horizon**: **2.5909 mph** MAE (RMSE 4.87 mph)
  - **30-minute horizon**: **3.1668 mph** MAE (RMSE 6.06 mph)
  - **60-minute horizon**: **4.0083 mph** MAE (RMSE 7.53 mph)
- **The Spatial Ablation Breakthrough**:
  Adding fixed 1st-order road network topological neighbors (from the METR-LA distance graph) reduced 60-minute forecasting error from **4.02 to 3.81 mph** (**5.2% relative error reduction**).

### Why Classical Machine Learning Hits a Hard Ceiling
1. **Static vs. Dynamic Propagation**: Classical ML concatenates fixed lag features from neighboring sensors with static tabular weights. In reality, traffic congestion shockwaves travel at variable physical speeds (10–25 mph upstream) depending on real-time highway density.
2. **Isotropic Receptive Fields**: Classical models cannot dynamically adjust their spatial focus: an upstream sensor should receive high attention during oncoming traffic jams, but zero attention during free-flow conditions.
3. **Decoupled Spatio-Temporal Interactions**: Classical tabular ML flattens time and space into a 2D matrix, losing the geometric topology of the freeway network.

### The Phase 2 Mission
In Phase 2, we implement **Spatio-Temporal Graph Attention Networks (ST-GAT)** to dynamically learn time-varying spatial attention coefficients $\alpha_{ij}^{(t)}$ between interconnected highway loop detectors, with the explicit goal of outperforming our Phase 1 benchmark of **2.59 mph** at 15m and **4.01 mph** at 60m.

---

## 2. Mathematical Formulations for Phase 2

### 1. Dynamic Graph Attention Coefficients
Given highway sensor graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A})$ where $|\mathcal{V}| = 207$ nodes and node representations $\mathbf{h}_i^{(t)} \in \mathbb{R}^F$, the dynamic spatial attention weight $\alpha_{ij}^{(t)}$ from sensor $j$ to sensor $i$ at time $t$ is:

$$\alpha_{ij}^{(t)} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i^{(t)} \parallel \mathbf{W} \mathbf{h}_j^{(t)}]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i^{(t)} \parallel \mathbf{W} \mathbf{h}_k^{(t)}]\right)\right)}$$

Where:
- $\mathbf{W} \in \mathbb{R}^{F' \times F}$ is a shared linear feature transformation matrix.
- $\mathbf{a} \in \mathbb{R}^{2F'}$ is a learnable attention parameter vector.
- $\parallel$ denotes tensor concatenation.
- $\mathcal{N}_i$ is the 1st-order topological neighborhood of sensor $i$ based on Caltrans road distances.

### 2. Multi-Head Attention Aggregation
To stabilize the learning process and capture diverse traffic phenomena (e.g., commute flow vs. incident shockwaves), we employ $K$ independent attention heads:

$$\mathbf{h}_i'^{(t)} = \Vert_{k=1}^K \sigma \left( \sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(k, t)} \mathbf{W}^{(k)} \mathbf{h}_j^{(t)} \right)$$

### 3. Spatio-Temporal Joint Modeling
Traffic dynamics exhibit strong temporal autocorrelation. The historical sequence $[t-11, \dots, t]$ is first encoded using a causal Recurrent Unit (GRU) or 1D Dilated Temporal Convolution (TCN):

$$\mathbf{H}_{\text{temporal}}^{(t)} = \text{GRU}\left( [\mathbf{X}_{t-11}, \dots, \mathbf{X}_t] \right) \in \mathbb{R}^{N \times D_{\text{rnn}}}$$

Followed by spatial aggregation across the road network topology:

$$\mathbf{H}_{\text{spatial}}^{(t)} = \text{GAT}\left( \mathbf{H}_{\text{temporal}}^{(t)}, \mathbf{A} \right) \in \mathbb{R}^{N \times D_{\text{gat}}}$$

---

## 3. Four-Person Workstream Division for Phase 2

| Workstream | Member Designation | Focus Area | Core Modules & Deliverables | Acceptance Target |
| :--- | :--- | :--- | :--- | :--- |
| **WS1: Graph Data & Lineage** | Member 1 | Full-Scale 207-Sensor Graph Pipeline, Dynamic Adjacency Matrices & Temporal Batching | `src/models_phase2/graph_dataset.py`, distance thresholding, zero-leakage splits | Yield $(B, T, N, F)$ batches with exact chronological isolation |
| **WS2: GAT Spatial Architecture** | Member 2 | Multi-Head GAT Layer, Attention Weight Extraction, Residual Connections | `src/models_phase2/gat_layer.py`, attention heatmap visualizer | Softmax sum to 1.0, multi-head concat/average modes |
| **WS3: Temporal Sequence & Heads** | Member 3 | Temporal GRU / 1D-CNN Backbone, Multi-Horizon Direct Heads (+15, +30, +60m) | `src/models_phase2/st_gat_model.py`, masked MAE/RMSE/MAPE loss | Output $(B, N, 3)$ predictions matching test timestamps |
| **WS4: Training, Benchmarking & App** | Member 4 | Training Loop, Early Stopping, Baseline Comparison Tables, Streamlit Phase 2 Visualizer | `src/models_phase2/train_gat.py`, `app/streamlit_app.py` update, IEEE Phase 2 section | Beat Phase 1 XGBoost benchmarks on test partition |

---

## 4. Phase 2 Milestones & Sprint Plan

```
Sprint 1: Graph Pipeline & GAT Scaffolding (Days 1–4)
  ├── WS1: Build SpatioTemporalTrafficDataset with PyTorch DataLoaders
  ├── WS2: Unit test single-head and multi-head GraphAttentionLayer
  └── WS3: Assemble ST-GAT forward pass with GRU and multi-horizon heads

Sprint 2: Training Harness & Hyperparameter Tuning (Days 5–8)
  ├── WS3: Implement masked MAE loss with epsilon-clamping
  ├── WS4: Run training with AdamW, learning rate scheduler, and early stopping
  └── WS1: Scale from 15-sensor pilot to full 207-sensor network

Sprint 3: Empirical Benchmarking & Ablation (Days 9–11)
  ├── WS4: Generate side-by-side comparison tables against Phase 1 models
  ├── WS2: Extract and plot dynamic attention weight matrices during rush hour
  └── WS3: Stratify peak vs. off-peak errors across horizons

Sprint 4: Streamlit Dashboard v2 & Final Viva Defense (Days 12–14)
  ├── WS4: Integrate live attention heatmap in Streamlit app (Tab 5)
  ├── All: Author Phase 2 IEEE manuscript section and slide deck updates
  └── All: Execute viva rehearsal with examiners' defense checklist
```

---

## 5. Quantitative Target Benchmarks

| Horizon | Lookahead | Phase 1 Persistence | Phase 1 Ridge | Phase 1 XGBoost | **Phase 2 ST-GAT Target** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **15 minutes** | +3 steps | 3.00 mph | 2.88 mph | 2.59 mph | **< 2.45 mph** |
| **30 minutes** | +6 steps | 3.62 mph | 3.61 mph | 3.17 mph | **< 2.95 mph** |
| **60 minutes** | +12 steps | 4.73 mph | 4.91 mph | 4.01 mph | **< 3.65 mph** |

