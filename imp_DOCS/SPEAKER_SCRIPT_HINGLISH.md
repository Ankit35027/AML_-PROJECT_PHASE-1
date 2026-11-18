# 🎤 4-Member Viva & Presentation Master Script (in Natural Hinglish)
## Historical METR-LA Traffic Speed Forecasting (Phase 1 Baselines & Phase 2 ST-GAT)

**Deck Structure**: 9 Slides Total (Widescreen 16:9 in `imp_DOCS/Phase2_ST_GAT_Presentation.pptx`)  
**Division**: Each member presents strictly **2 to 3 slides** (~2.5 to 3 minutes each)  
**Total Presentation Time**: 10 to 12 Minutes  

---

```
                       PRESENTATION FLOW & TIMING
  ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
  │ Member 1: Ankit │ Member 2: Yash  │ Member 3: Ansh  │ Member 4: Sam   │
  │  (Slides 1 - 2) │  (Slides 3 - 4) │  (Slides 5 - 6) │  (Slides 7 - 9) │
  │   0:00 - 2:30   │   2:30 - 5:00   │   5:00 - 7:30   │   7:30 - 11:00  │
  │  Data & Audit   │ Phase 1 Summary │   ST-GAT Model  │ Results & App   │
  │                 │  & GAT Concept  │    & Heatmaps   │   & Conclusion  │
  └─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

---

## 👤 MEMBER 1: ANKIT GARG (`ankit35027`)
**Assigned Slides**: Slide 1 & Slide 2 (2 Slides)  
**Workstream**: WS1 - Data Ingestion, Lineage, Missingness Protocol & Chronological Splits  
**Speaking Time**: 0:00 to 2:30 Minutes  

### 📌 Slide 1: Title & Executive Overview
> **Script**:
> *"Good morning respected professors and committee members. Mera naam Ankit Garg hai, aur main apne team members Yashpal, Anshika, aur Samriddhi Shah ke saath hamara AML project present kar raha hoon: **'Historical METR-LA Traffic Speed Forecasting'**.*
>
> *Presentation start karne se pehle main ek bohot zaroori **Academic Guardrail** clarify karna chahta hoon: Hamara course title traffic signal control mention karta hai, lekin benchmark Caltrans METR-LA dataset freeway ke inductive loop detectors ki speed record karta hai (miles per hour mein). Isme municipal traffic signals ya intersection queue lengths record nahi hoti. Isliye academic truthfulness maintain karte hue, hamara research strictly **multi-horizon traffic speed prediction (+15m, +30m, +60m)** par focused hai, bina kisi ungrounded signal control claim ke.*
>
> *Humne is project ko do rigorous phases mein execute kiya hai:*
> - *Phase 1: Zero-leakage Classical Machine Learning baselines (Persistence, Ridge, XGBoost) aur spatial ablation.*
> - *Phase 2: Spatio-Temporal Graph Attention Networks (ST-GAT) jo highway graph par dynamic edge weights learn karta hai.*
>
> *Sabhi chaar members ne dedicated workstreams deliver kiye hain jo hamari GitHub repository par committed hain."*

### 📌 Slide 2: METR-LA Dataset & Missingness Accounting
> **Script**:
> *"Ab aate hain hamare dataset aur data integrity protocol par.*
>
> *METR-LA benchmark mein Los Angeles County ke primary freeways—jaise I-5, I-10, I-405, aur US-101—par lage **207 loop detectors** ka data hai. March 1 se June 27, 2012 tak har 5 minute ka continuous record hai, jisme total **34,272 timestamps** hain. Raw data ka SHA-256 cryptographic hash humne verify karke `data_lineage.json` mein store kiya hai.*
>
> *Data ingestion ke dauran humne ek deep missingness audit perform kiya: Dataset mein 0 NaNs the aur 0 impossible speeds (>100 mph) thi. Lekin **8.11% readings (575,302 values) exactly 0.0 mph** thi. Real-world freeway par 0.0 mph ka matlab ya toh sensor failure hota hai ya road closure. Isliye humne protocol banaya:*
> 1. *Input features ke liye hum causal forward-fill use karte hain taaki missingness handle ho sake.*
> 2. *Ground-truth target evaluation mein hum strictly valid readings (>0.0 mph) evaluate karte hain taaki dropouts score distort na karein.*
>
> *Sabse important part hai hamara **Chronological 70/10/20 Split**: Traffic data mein random k-fold validation lagana blunder hota hai kyunki future data past mein leak ho jata hai. Humne time ke hisaab se 70% Train (Mar-May), 10% Validation (May-Jun), aur 20% Held-Out Test (June) divide kiya. StandardScaler sirf training data par fit kiya gaya.*
>
> *Ab main Yashpal ko hand over karta hoon Phase 1 ki empirical summary aur GAT formulation explain karne ke liye."*

---

## 👤 MEMBER 2: YASHPAL (`Yashpal-2004`)
**Assigned Slides**: Slide 3 & Slide 4 (2 Slides)  
**Workstream**: WS2 - Causal Sliding Windows, Classical Baselines & GAT Formulation  
**Speaking Time**: 2:30 to 5:00 Minutes  

### 📌 Slide 3: Phase 1 Summary: Classical ML & Spatial Ablation (Dedicated Phase 1 Slide)
> **Script**:
> *"Thank you Ankit. Respected examiners, main Slide 3 mein **Phase 1 ki complete empirical summary** present kar raha hoon.*
>
> *Phase 1 mein hamara maksad tha solid classical ML foundation banana. Humne origin timestamp $t$ par past 12 time steps (pichle 60 minutes: $t-11 \dots t$) ki speed aur cyclical time features (sin/cos of time-of-day aur day-of-week) ka causal sliding window banaya. Target lookaheads the +15 min ($h=3$), +30 min ($h=6$), aur +60 min ($h=12$).*
>
> *Humne teen classical models benchmark kiye 90,149 frozen test instances par:*
> 1. *Historical Persistence: Bas pichli speed carry forward karta hai.*
> 2. *Ridge Regression: L2-regularized linear model.*
> 3. *XGBoost Regressor: 100 gradient boosted trees.*
>
> *Results bohot revealing the:*
> - *15 min lookahead par XGBoost ne 2.59 mph MAE diya (Persistence 3.00 mph).*
> - *60 min lookahead par XGBoost ne 4.01 mph achieve kiya. Yahan Ridge regression (4.91 mph) naive Persistence (4.73 mph) se bhi kharab ho gaya kyunki collinear features par linear model fail ho jata hai.*
>
> *Lekin Phase 1 ka sabse bada scientific breakthrough tha hamara **'Spatial Feature Ablation'**: Jab humne sensor ke khud ke lags ke sath road network ke top-3 neighbors ke lags inject kiye, toh 60-minute MAE 4.02 se ghat kar **3.81 mph ho gaya—pura 5.2% relative error reduction!***
>
> *Isne physically prove kiya ki highway congestion aage aur peeche shockwaves ki tarah propagate karta hai. Aur yahi proof hamari foundation bana Phase 2 ke Graph Attention Networks ke liye."*

### 📌 Slide 4: Phase 2 Formulation: Moving Beyond Static Tabular ML
> **Script**:
> *"Ab sawal yeh aata hai: Agar XGBoost 2.59 mph achieve kar raha tha, toh hume Graph Attention Network ki zaroorat kyu padi?*
>
> *Classical tabular machine learning ek hard ceiling par hit karta hai do main reasons ki wajah se:*
> 1. *Pehla: Static Weights vs Dynamic Shockwaves. Tabular model har neighbor ko ek fixed regression weight assign karta hai. Lekin real life mein highway jam 10 se 25 mph ki speed se travel karta hai. Ek upstream sensor subah rush hour mein bohot critical hota hai, lekin raat ko 2 baje bilkul irrelevant. Classical model time ke hisaab se spatial attention change nahi kar sakta.*
> 2. *Dusra: Tabular concatenation road network ke physical graph topology ko destroy kar deta hai.*
>
> *Is problem ko solve karne ke liye Phase 2 mein humne **Dynamic Graph Attention Network (GAT)** formulate kiya:*
> - *Har sensor pair $(i, j)$ ke beech attention energy compute hoti hai: $e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}h_i \parallel \mathbf{W}h_j])$.*
> - *Fir road network adjacency matrix se mask karke row-softmax calculate hota hai, jisse dynamic spatial attention coefficient $\alpha_{ij}^{(t)}$ milta hai.*
> - *Humne $K=4$ independent attention heads use kiye hain taaki commute traffic aur sudden accident shockwaves alag-alag capture ho sakein.*
> - *Unit tests se verified hai ki har node par attention weights ka sum strictly $1.0$ hota hai.*
>
> *Ab Anshika aapko hamara end-to-end deep learning architecture aur attention heatmaps explain karengi."*

---

## 👤 MEMBER 3: ANSHIKA (`anshika292005`)
**Assigned Slides**: Slide 5 & Slide 6 (2 Slides)  
**Workstream**: WS3 - Temporal Backbone (GRU), SpatioTemporalGAT Architecture & Attention Heatmaps  
**Speaking Time**: 5:00 to 7:30 Minutes  

### 📌 Slide 5: ST-GAT End-to-End Pipeline & Training Harness
> **Script**:
> *"Thank you Yashpal. Respected committee members, main Anshika hoon aur main hamara deep learning model architecture aur training harness present kar rahi hoon.*
>
> *Hamara deep architecture—**SpatioTemporalGAT**—teen main building blocks se bana hai:*
> 1. *Temporal Sequence Backbone: Input batch $(B, T=12, N=207, F=5)$ hota hai. Ek causal Gated Recurrent Unit (GRU) jisme 64 hidden units hain, pichle 12 time steps ke temporal patterns ko encode karta hai.*
> 2. *Spatial GAT Layer: 4-head dynamic attention layer sensor adjacency tensor par operate karti hai aur highway neighbors ke beech real-time spatial representations aggregate karti hai.*
> 3. *Multi-Horizon Direct Heads: Linear projection heads ek hi forward pass mein +15m, +30m, aur +60m speeds output karti hain bina autoregressive error compounding ke.*
>
> *Pura architecture bohot lightweight hai: **Sirf 70,403 trainable parameters**, jisse yeh fast infer karta hai aur low-resource hardware par bhi smoothly deploy ho jata hai.*
>
> *Model training ke liye humne ek custom **Masked MAE Loss function** implement kiya:*
> $$\mathcal{L}_{\text{masked}} = \frac{\sum |y - \hat{y}| \cdot M}{\sum M + \epsilon}$$
> *Yahan mask $M$ sirf valid speeds ($>0.0$ mph) ko select karta hai. Iska fayda yeh hua ki loop detector dropouts aur missing values gradient updates ko distort nahi karte. Humne AdamW optimizer, ReduceLROnPlateau scheduler, aur gradient clipping (max norm = 5.0) use kiya. Validation early stopping ke baad best model weights `models/st_gat_best_checkpoint.pt` mein save kiye gaye."*

### 📌 Slide 6: Dynamic Attention Heatmaps: Tracking Shockwaves (Figure Embed)
> **Script**:
> *"Ab examiners ke saamne hamare deep model ka sabse expressive proof: **Dynamic Attention Heatmap**, jo Slide 6 ke right side par visible hai.*
>
> *Neural networks aksar black-box mane jaate hain, lekin Graph Attention Network hame direct physical interpretability deta hai. Is plot mein do alag-alag time periods ka comparison hai:*
>
> *1. Raat ke 2:00 baje (Midnight Free-Flow - Right Side):*
> *Jab highway khali hota hai aur gaadiyan 65 mph par chal rahi hoti hain, toh attention weight matrix 85% diagonal self-loop par concentrate ho jata hai ($\alpha_{ii} \approx 0.85$). Iska physical meaning yeh hai ki free-flow condition mein har sensor ki speed persistent hoti hai aur neighbor sensor ka koi correlation nahi hota.*
>
> *2. Subah ke 8:00 baje (Morning Rush Hour - Left Side):*
> *Jab commute traffic shuru hota hai aur freeway par bottleneck banta hai, toh attention weights turant diffuse hokar upstream aur downstream sensors par fail jaate hain ($\alpha_{ij} \approx 0.25 - 0.40$). Model actively detect kar leta hai ki peeche se traffic ruk raha hai aur shockwaves propagate ho rahi hain!*
>
> *Yeh heatmap traffic physics ka direct visual confirmation hai ki model ne real-world shockwaves dynamically learn kar li hain. Ab main Samriddhi ko invite karti hoon final outcomes aur Streamlit dashboard present karne ke liye."*

---

## 👤 MEMBER 4: SAMRIDDHI SHAH (`samriddhi-techy`)
**Assigned Slides**: Slide 7, Slide 8 & Slide 9 (3 Slides)  
**Workstream**: WS4 - Systems, Benchmark Verification, Streamlit App & Viva Defense Release  
**Speaking Time**: 7:30 to 11:00 Minutes  

### 📌 Slide 7: Phase 2 Empirical Outcomes: ST-GAT Surpasses Baselines (Figure Embed)
> **Script**:
> *"Thank you Anshika. Respected committee members, main Samriddhi Shah hoon aur main Phase 2 ke verified empirical test outcomes present kar rahi hoon.*
>
> *Slide 7 par hamara held-out test benchmark comparison chart hai jo 90,149 instances par evaluate kiya gaya hai. Hamare ST-GAT model ne har single horizon par Phase 1 ke best XGBoost baseline ko convincingly beat kiya hai:*
> - *15 Minutes Lookahead (+3 steps):*
>   *XGBoost 2.5909 mph tha -> **ST-GAT ne 2.4124 mph** achieve kiya (**6.89% error reduction**; RMSE 4.58 mph, MAPE 6.18%).*
> - *30 Minutes Lookahead (+6 steps):*
>   *XGBoost 3.1668 mph tha -> **ST-GAT ne 2.8945 mph** achieve kiya (**8.60% error reduction**; RMSE 5.51 mph, MAPE 7.82%).*
> - *60 Minutes Lookahead (+12 steps) — The Major Breakthrough:*
>   *XGBoost 4.0083 mph tha -> **ST-GAT ne 3.5821 mph** achieve kiya (**10.63% relative error reduction**; RMSE 6.79 mph, MAPE 10.21%).*
>
> *Is trend ko dhyan se dekhiye: Short horizon (15 min) par traffic localized hota hai, isliye XGBoost aur ST-GAT paas the. Lekin jaise horizon 60 minutes par jata hai, distant shockwaves poore freeway network mein failti hain. Tabular model wahan fail hone lagta hai aur error amplify hota hai, jabki hamara ST-GAT dynamic spatial attention ki madad se shockwaves ko capture karke error compounding ko rok leta hai!"*

### 📌 Slide 8: Continuous Forecast Traces & Offline Streamlit Dashboard (Figure Embed)
> **Script**:
> *"Slide 8 ke left side par hamara 24-hour continuous speed forecast trace plot hai (+15 min lookahead).*
>
> *Aap dekh sakte hain ki jab rush hours mein highway speed achanak 65 mph se girkar 25 mph par drop hoti hai, toh ST-GAT forecast (green dashed line) actual ground truth speed (black line) ko bina artificial lag ya oversmoothing ke follow karta hai.*
>
> *Examiners ke real-time verification ke liye humne ek interactive **Streamlit Application (`app/streamlit_app.py`)** deploy kiya hai. Yeh dashboard offline-first architecture par chalta hai: precomputed test parquet files cached hain, jiski wajah se yeh Cloud par **2 second ke andar boot ho jata hai bina retraining ke**.*
>
> *Dashboard ke paanch comprehensive modules hain:*
> 1. *Module 1: Executive Scope & Data Lineage hashes.*
> 2. *Module 2: Missingness audit & 207-sensor time-series filter.*
> 3. *Module 3: Multi-Horizon Forecast Lab jahan aap kisi bhi sensor aur timestamp ko select karke live ground truth vs prediction compare kar sakte hain.*
> 4. *Module 4: Peak vs Off-Peak residual error analysis.*
> 5. *Module 5: Road network geography, dynamic GAT attention heatmaps, aur Phase 2 test metrics table."*

### 📌 Slide 9: Project Conclusion & Committee Defense Verification (All Members)
> **Script**:
> *"To conclude our presentation: Hamari four-member engineering team ne project ke dono phases ko highest scientific integrity ke sath deliver kiya hai.*
>
> *Hamare key achievements:*
> 1. *Phase 1 mein reproducible classical ML baselines establish kiye aur spatial ablation se 5.2% error reduction demonstrate kiya.*
> 2. *Phase 2 mein SpatioTemporalGAT implement karke 60-minute lookahead par 10.63% error reduction achieve kiya.*
> 3. *Hamare saare deliverables—IEEE publication-grade paper (`imp_DOCS/Phase2_IEEE_Technical_Report.pdf`), master presentation deck, interactive Streamlit app, PyTorch checkpoint (`models/st_gat_best_checkpoint.pt`), aur 10/10 automated passing unit tests—hamare GitHub repository ke `main` branch par live committed hain.*
> 4. *Sabhi commits chaaron members ke respective GitHub IDs (`ankit35027`, `Yashpal-2004`, `anshika292005`, `samriddhi-techy`) par properly attributed hain.*
>
> *Hum pure project mein strict academic honesty maintain karte hain: sirf verified highway speed prediction, bina kisi ungrounded signal control claim ke.*
>
> *Now, our entire research team is ready and warmly welcomes any questions from the examining committee. Thank you very much!"*

---

## 💡 Top 5 Tricky Examiner Questions (And Quick Hinglish Answers)

| Question | Examiner's Trap | Perfect Hinglish Answer |
| :--- | :--- | :--- |
| **1. "Course title mein 'Signal Control' hai, aapne signal control kyu nahi kiya?"** | Checking academic honesty. | *"Sir, METR-LA dataset freeway loop detectors ka hai, isme sirf speed (mph) hai. Traffic signal timings ya queue lengths data mein exist hi nahi karti. Bina data ke signal control claim karna academic dishonesty hoti, isliye humne strictly speed forecasting par focus kiya."* |
| **2. "XGBoost tabular hai, ST-GAT deep learning. ST-GAT ne 60m par itna bada improvement (10.63%) kyu diya?"** | Testing traffic physics understanding. | *"15 min par local autoregression dominate karti hai, isliye XGBoost close tha. Lekin 60 min par congestion shockwaves highway par propagate hoti hain. Tabular model static lag leta hai isliye error amplify hota hai; ST-GAT dynamic attention weights $\alpha_{ij}^{(t)}$ compute karke shockwave capture kar leta hai."* |
| **3. "8.11% zero speeds ko kaise handle kiya?"** | Checking leakage and missingness protocol. | *"Features ke liye causal forward-fill use kiya taaki causality maintain rahe. Ground truth test evaluation mein sirf valid speeds (>0.0 mph) evaluate ki, aur training ke dauran custom Masked MAE loss lagaya taaki sensor dropouts par gradients zero rahein."* |
| **4. "Ridge regression 60 min par Persistence se bhi kharab kyu ho gaya?"** | Testing linear algebra and collinearity. | *"60 minutes par consecutive historical lags highly collinear hote hain. Ridge regression collinear features par hyper-plane extrapolate karne ki koshish karta hai, jisse non-linear congestion shifts mein error blow up ho gaya (4.91 mph vs Persistence 4.73 mph)."* |
| **5. "Graph attention matrix ka physical interpretation kya hai?"** | Testing interpretability of Slide 6 heatmap. | *"Raat ko free-flow mein 85% attention diagonal self-loop par rehta hai kyunki speed persistent hoti hai. Morning rush hour mein attention diffuse hokar road neighbors par fail jata hai ($\alpha_{ij} \approx 0.35$), jo physically shockwave propagation prove karta hai."* |
