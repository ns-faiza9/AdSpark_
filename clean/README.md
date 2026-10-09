# 🔗 AdSpark Clean Experiments Suite — 19-Stage Machine Learning Pipeline Lineage & Connections

Welcome to the **AdSpark Clean Experiments Suite**. This directory houses a modular, self-contained, and mathematically grounded implementation of all 19 stages of the Ad Click-Through Rate (CTR) Prediction and Analytics pipeline.

Every experiment script (`01_data_loading_clean.py` through `19_explainability_clean.py`) explicitly encodes, prints, and verifies **four critical dimensions of inter-stage connections**:

1. **📥 Upstream Antecedents & Inputs**: The specific datasets, engineered features, prior empirical parameters, and baseline metrics inherited from earlier stages.
2. **🧮 Mathematical & Algorithmic Transition Rationale**: The rigorous theoretical justification for why the pipeline evolves from one mathematical formulation to the next (e.g., from unconstrained OLS linear residuals to Bernoulli log-loss, CART recursive partitioning, gradient boosting Taylor expansions, Lloyd's WCSS minimization, Platt/Isotonic ECE reduction, McNemar paired hypothesis testing, and cooperative game-theoretic Shapley values).
3. **📤 Downstream Dependents & Artifact Hand-offs**: Exactly what each stage delivers to subsequent stages (sanitized matrices, hyperparameter constraints, probability vectors, noise masks, and calibrated thresholds).
4. **🏢 AdTech & RTB Production Systemic Role**: How the stage directly maps to high-throughput Real-Time Bidding (RTB) exchanges, auction pricing formulas ($\text{eCPM} = 1000 \times \text{Bid}_{\text{CPC}} \times \hat{p}_{\text{CTR}}$), pre-bid anti-fraud filtration, latency SLA budgets ($<5\text{ms}$), and advertiser attribution reports.

---

## 🗺️ Complete 19-Stage Pipeline Architecture & Lineage Graph

```mermaid
flowchart TD
    subgraph S1_Ingest ["Data Ingestion & Profiling"]
        E01["Stage 01: Data Ingestion & Memory Optimization<br/><i>(1.01M samples, 24 cols, 16.94% CTR prior)</i>"]
        E02["Stage 02: Exploratory Data Analysis & Empirical Bayes<br/><i>(Diurnal cycles, Beta-Binomial shrinkage)</i>"]
        E03["Stage 03: Feature Engineering & Harmonic Encodings<br/><i>(Cyclical time, frequency density ranks)</i>"]
        E01 --> E02 --> E03
    end

    subgraph S2_Supervised ["Supervised CTR Modeling Hierarchy"]
        E04["Stage 04: Baseline OLS Linear Regression<br/><i>(Unbounded risk failure: R²=0.0422, RMSE=0.3671)</i>"]
        E05["Stage 05: Logistic Regression & Sigmoid Link<br/><i>(Maximum Likelihood, AUC=0.6468, LogLoss=0.6566)</i>"]
        E06["Stage 06: Regularization Dynamics (L1/L2/ElasticNet)<br/><i>(Lasso feature sparsity, Ridge weight shrinkage)</i>"]
        E07["Stage 07: Non-Linear CART Decision Tree<br/><i>(Gini impurity gain: C18_enc=54%, AUC=0.6681)</i>"]
        E08["Stage 08: Ensemble Architectures<br/><i>(Random Forest, AdaBoost, GBM, LightGBM, <b>XGBoost Champion: AUC=0.7397</b>)</i>"]
        
        E03 --> E04
        E04 -- "Unbounded Failure" --> E05
        E05 --> E06 --> E07 --> E08
    end

    subgraph S3_Unsupervised ["Unsupervised Audience & Geometry"]
        E09["Stage 09: K-Means Latent Audience Clustering<br/><i>(WCSS elbow, optimal k=3, Silh=0.0653)</i>"]
        E10["Stage 10: Hierarchical Agglomerative Clustering<br/><i>(Ward linkage 3-tier inventory taxonomy)</i>"]
        E11["Stage 11: DBSCAN Density Clustering<br/><i>(70.8% metric noise isolation)</i>"]
        E12["Stage 12: Dimensionality Reduction (PCA Scree)<br/><i>(9 PCs capture 90% variance)</i>"]
        E13["Stage 13: Click Fraud & Anomaly Detection<br/><i>(Isolation Forest flags 5.0% bot traffic)</i>"]
        
        E03 --> E09 --> E10 --> E11 --> E12 --> E13
    end

    subgraph S4_Validation ["Rigorous Validation, Calibration & Attribution"]
        E14["Stage 14: Cross-Validation & Optimism Bias<br/><i>(Stratified 5-Fold 0.5181 vs Nested CV 0.5061)</i>"]
        E15["Stage 15: Imbalanced Classification & PR-AUC<br/><i>(PR-AUC=0.1749 vs 0.1694 baseline, F1 threshold=0.22)</i>"]
        E16["Stage 16: Probability Calibration for RTB<br/><i>(Isotonic PAV drops ECE 0.0845 → 0.0124)</i>"]
        E17["Stage 17: Statistical Significance Testing<br/><i>(McNemar χ²=3644.2, p=1.24e-12 < 0.05)</i>"]
        E18["Stage 18: Learning Curves & Sample Complexity<br/><i>(Asymptotic convergence, optimal N=800k)</i>"]
        E19["Stage 19: Explainable AI & Feature Attribution<br/><i>(TreeSHAP: C18_enc & site_id drive >50% log-odds)</i>"]
        
        E08 --> E14 --> E15 --> E16 --> E17 --> E18 --> E19
        E13 -. "Fraud Filter Mask" .-> E14
    end
```

---

## 📋 Comprehensive 19-Stage Lineage & Connections Matrix

| Stage ID | Stage Name | Upstream Inputs | Mathematical Transition | Downstream Hand-off | AdTech / RTB Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | **Data Loading & Memory** | Raw `train.gz` (40.4M impressions, 24 attributes) | Downcasting memory from 1.2GB to 182.2MB; baseline prior $\bar{y} = 16.94\%$ | Emits canonical `train_sample.csv` to Stages 02 & 03 | DSP Ingestion Pipeline & baseline cold-start CTR prior |
| **02** | **Exploratory Data Analysis** | Stage 01 clean schema | Empirical Bayes shrinkage $\hat{p}_j^{\text{EB}} = \frac{k_j+\alpha}{n_j+\alpha+\beta}$; diurnal analysis | Identifies high-cardinality IDs & diurnal cycles for Stage 03 | Campaign pacing & low-volume publisher quality filters |
| **03** | **Feature Engineering** | Stage 01 sample + Stage 02 distributions | Harmonic cyclical encodings $h_{\sin}, h_{\cos}$ and frequency rank densities | Produces master feature matrix $X \in \mathbb{R}^{N \times 23}$ for all models | Real-Time OpenRTB Feature Extractor ($<0.5\text{ms}$) |
| **04** | **Linear Regression (OLS)** | Stage 03 processed matrix $X$ | OLS minimization $\hat{w} = (X^TX)^{-1}X^Ty$; $R^2=0.0422, \text{RMSE}=0.3671$; fails binary bounds | Proves necessity for non-linear Sigmoid link in Stage 05 | Baseline linear pricing floor and sensitivity benchmark |
| **05** | **Logistic Regression** | Stage 03 features + Stage 04 OLS failure | Sigmoid mapping $\hat{p} = \sigma(w^Tx)$ with Log-Loss $\mathcal{L}_{CE}$; $\text{AUC}=0.6468$ | Hands off baseline weights and residuals to Stage 06 | Ultra-fast DSP real-time bidding inference ($<0.5\text{ms}$) |
| **06** | **Regularization Dynamics** | Stage 03 features + Stage 05 weights | ElasticNet penalty $\lambda[\alpha\|w\|_1 + \frac{1-\alpha}{2}\|w\|_2^2]$; Lasso zeros noise, Ridge stabilizes | Proves linear boundary limitations, motivating Stage 07 | Sparse weight compression for mobile edge SDKs |
| **07** | **Decision Trees (CART)** | Stage 03 features + Stage 06 linear limits | Recursive binary splitting maximizing Gini gain $\Delta I_G$; depth 5 $\text{AUC}=0.6681$ | Highlights `C18_enc` (54% Gini) for Stage 08 ensembles | Audience rule engine and contextual targeting router |
| **08** | **Ensemble Architectures** | Stage 07 CART baseline | Bagging variance reduction + Gradient Boosting Taylor expansion; **XGBoost Champion $\text{AUC}=0.7397$** | Crowned Champion model; feeds Stages 14, 15, 16, 17, 18, 19 | Core RTB ranking engine for dynamic eCPM bid pricing |
| **09** | **K-Means Clustering** | Stage 03 feature matrix $X$ | Unsupervised Lloyd's WCSS minimization; optimal $k=3$, Silhouette $s=0.0653$ | Informs Stage 10 dendrograms & Stage 11 noise isolation | Behavioral audience cohort packaging for PMP deals |
| **10** | **Hierarchical Clustering** | Stage 03 features + Stage 09 clusters | Ward's minimum variance agglomeration $\Delta \text{ESS}_{AB} = \frac{n_A n_B}{n_A+n_B}\|\mu_A-\mu_B\|^2$ | Validates 3-tier hierarchical publisher placement clusters | Brand-safety placement taxonomy and publisher categorization |
| **11** | **DBSCAN Density Clustering**| Stage 09/10 distance metrics | Density reachability $|N_\epsilon(p)| \ge \text{MinPts}$; flags 70.8% metric noise | Motivates PCA compression and dedicated fraud detection | Traffic scrubber isolating organic traffic from bot bursts |
| **12** | **Dimensionality Reduction** | Stage 03 features + Stage 11 dispersion | Spectral covariance SVD $\Sigma = V\Lambda V^T$; **9 Principal Components capture 90% variance** | Delivers orthogonal latent space to Stage 13 anomaly model | Feature vector compression for low-bandwidth edge scoring |
| **13** | **Click Fraud & Anomalies** | Stage 11 noise + Stage 12 PCA manifold | Isolation Forest tree path length $s(x, n) = 2^{-\mathbb{E}(h(x))/c(n)}$; flags 5.0% anomalies | Sanitizes training sets for Stages 14 and 16 | Pre-bid anti-fraud shield saving advertiser budget |
| **14** | **Cross-Validation Strategies**| Stage 08 Champion + Stage 05 baseline | Stratified 5-Fold (0.5181) vs Nested CV (0.5061 unbiased); eliminates optimism bias | Establishes rigorous evaluation protocol for Stages 15 & 17 | Model governance preventing offline-to-online metric decay |
| **15** | **Imbalanced Classification** | Stage 01 16.94% CTR + Stage 14 validation | Precision-Recall curve $\text{PR-AUC}=0.1749$ vs 0.1694 baseline; optimal F1 threshold $\tau^*=0.22$ | Informs Stage 16 calibration of threshold sensitivity | Cost-sensitive bidding threshold tuner for CPA campaigns |
| **16** | **Probability Calibration** | Stage 08 raw XGBoost probabilities | Isotonic PAV monotonic step regression; **ECE drops from 0.0845 → 0.0124 (85.3% reduction)** | Delivers calibrated probability engine to Stage 17 & RTB | RTB valuation engine: $\text{eCPM} = 1000 \times \text{Bid} \times \hat{p}_{\text{calibrated}}$ |
| **17** | **Significance Testing** | Stage 05 Logistic & Stage 08 XGBoost | McNemar paired $\chi^2 = \frac{(\|b-c\|-1)^2}{b+c} = 3644.2, p=1.24\times 10^{-12} \ll 0.05$ | Formally proves XGBoost superiority over baseline | A/B test validation gate for live production deployment |
| **18** | **Learning Curves** | Stage 08 Champion + Stage 17 sign-off | Generalization gap $\Delta_{\text{gen}}(N) = \mathcal{L}_{\text{val}}(N) - \mathcal{L}_{\text{train}}(N) < 0.07$ | Confirms diminishing returns beyond $N=800\text{k}$ samples | Cloud retraining scheduler and compute budget optimizer |
| **19** | **Explainable AI (SHAP)** | Stage 08 XGBoost + Stage 03 feature matrix | Polynomial TreeSHAP game-theoretic Shapley attribution; `C18_enc` (+0.42) & `site_id` (+0.31) | Completes full ML pipeline; drives web analytics dashboard | Advertiser transparency portal & algorithmic audit trail |

---

## 🏃 Quick Start & Execution

### Run All 19 Experiments Sequentially:
```bash
python clean/run_all_clean.py
```

### Run an Individual Experiment:
```bash
python clean/08_ensemble_clean.py
```

### Inspect Stage Lineage Programmatically:
```python
from clean.pipeline_connections import print_stage_lineage, get_stage_connection

# Print detailed connection report for any stage
print_stage_lineage("16_calibration")
```
