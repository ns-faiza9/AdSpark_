# AdSpark — Production Machine Learning & CTR Prediction Suite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask 3.0](https://img.shields.io/badge/Flask-3.0-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green.svg?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AdSpark** is an end-to-end, mathematically rigorous Machine Learning platform and real-time inference engine for **Ad Click-Through Rate (CTR) Prediction**, developed on the **1.01M+ sample Avazu Click-Through Rate dataset**.

The system bridges foundational statistical learning theory with industrial ad tech architectures—featuring **Ensemble Boosted Trees (XGBoost, LightGBM, Random Forest, AdaBoost)**, **Linear & Regularized Baselines**, **Unsupervised Clustering (K-Means, DBSCAN, Hierarchical)**, **Dimensionality & Outlier Detection**, **Facebook Negative Downsampling calibration**, **Brier score decomposition**, **Expected Calibration Error (ECE)**, and a **20-module interactive Flask Web Application & Sandbox**.

---

## 📑 Table of Contents

1. [Executive Overview & Industry Challenges](#-executive-overview--industry-challenges)
2. [Mathematical Foundations & Theoretical Derivations](#-mathematical-foundations--theoretical-derivations)
   - [2.1 Bernoulli Maximum Likelihood & Normalized Entropy](#21-bernoulli-maximum-likelihood--normalized-cross-entropy)
   - [2.2 Negative Downsampling Odds Inversion (He et al.)](#22-negative-downsampling-odds-inversion-he-et-al)
   - [2.3 Probability Calibration & Brier Score Decomposition](#23-probability-calibration--brier-score-decomposition)
   - [2.4 McNemar Hypothesis Testing with Continuity Correction](#24-mcnemar-hypothesis-testing-with-continuity-correction)
3. [System Architecture](#-system-architecture)
4. [Comprehensive 10-Model Production Leaderboard](#-comprehensive-10-model-production-leaderboard)
5. [20-Module Analytics & Application Directory](#-20-module-analytics--application-directory)
6. [Quickstart & Installation](#-quickstart--installation)
7. [Interactive CTR Sandbox & API Documentation](#-interactive-ctr-sandbox--api-documentation)
8. [Scientific References](#-scientific-references)

---

## 🎯 Executive Overview & Industry Challenges

Accurate Click-Through Rate (CTR) estimation is the core scoring mechanism powering modern real-time bidding (RTB) auctions and sponsored search recommendation systems. CTR models face distinct statistical and systemic challenges:

1. **Extreme Categorical Sparsity & High Cardinality**: Ad identifiers (`device_ip`, `device_id`, `site_id`, `app_id`) span millions of unique discrete values, where naive one-hot encoding causes catastrophic dimensionality explosion.
2. **Severe Class Imbalance**: Ad click events are rare ($p \approx 16.9\%$), requiring negative downsampling during training and exact inverse odds re-calibration during serving.
3. **Probability Calibration Requirements**: Downstream revenue in second-price auctions depends on expected value $\mathbb{E}[\text{Bid}] = \text{eCPM} = \hat{p}_{\text{CTR}} \times \text{Bid}_{\text{CPC}}$. Models must output genuine, well-calibrated posterior probabilities, not merely rank-ordered margins.
4. **Generalization & Evaluation Integrity**: Evaluating nested cross-validation, PR-AUC, and paired McNemar significance prevents overfitting to correlated impression clusters.

---

## 📐 Mathematical Foundations & Theoretical Derivations

### 2.1 Bernoulli Maximum Likelihood & Normalized Cross-Entropy

Click events $y_i \in \{0, 1\}$ conditional on context vector $\mathbf{x}_i \in \mathbb{R}^d$ follow a Bernoulli distribution $Y \mid \mathbf{x} \sim \text{Bernoulli}(p(\mathbf{x}))$. For $N$ independent observations, the negative log-likelihood (Binary Cross-Entropy Loss) is:

$$\mathcal{L}_{\text{LogLoss}}(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln (1 - \hat{p}_i) \right]$$

To benchmark models independently of background click rate variations ($p_{\text{base}} = \frac{1}{N}\sum y_i$), **Normalized Cross-Entropy (NE)** measures the fraction of information gained over the uninformative entropy baseline:

$$\text{NE} = \frac{\mathcal{L}_{\text{LogLoss}}(y, \hat{p})}{-\left[ p_{\text{base}} \ln p_{\text{base}} + (1 - p_{\text{base}}) \ln (1 - p_{\text{base}}) \right]}$$

---

### 2.2 Negative Downsampling Odds Inversion (He et al.)

When negative impressions are sampled at rate $w \in (0, 1]$, the observed conditional probability in the downsampled dataset is $p' = \mathbb{P}(y=1 \mid \mathbf{x}, \text{sampled})$. Applying Bayes' rule:

$$\text{Odds}' = \frac{p'}{1 - p'} = \frac{\mathbb{P}(y=1 \mid \mathbf{x})}{\mathbb{P}(y=0 \mid \mathbf{x}) \cdot w} = \frac{\text{Odds}}{w} \implies \text{Odds} = w \cdot \text{Odds}'$$

Solving for the true uncalibrated probability $p$:

$$p = \frac{\text{Odds}}{1 + \text{Odds}} = \frac{w \cdot \dfrac{p'}{1 - p'}}{1 + w \cdot \dfrac{p'}{1 - p'}} = \frac{p'}{p' + \dfrac{1 - p'}{w}}$$

$$\Delta \text{Log-Odds} = \ln(w)$$

---

### 2.3 Probability Calibration & Brier Score Decomposition

The mean squared probabilistic error (Brier Score) algebraically decomposes into:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{p}_i)^2 = \underbrace{\sum_{k=1}^K \frac{N_k}{N} (\bar{p}_k - \bar{o}_k)^2}_{\text{Reliability (Calibration Error)}} - \underbrace{\sum_{k=1}^K \frac{N_k}{N} (\bar{o}_k - p_{\text{base}})^2}_{\text{Resolution (Discrimination)}} + \underbrace{p_{\text{base}}(1 - p_{\text{base}})}_{\text{Uncertainty (Base Entropy)}}$$

**Expected Calibration Error (ECE)** across $M$ reliability bins:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|, \quad \text{MCE} = \max_{m} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

*Isotonic regression reduces AdSpark's ECE from **0.0845** to **0.0124** (85.3% error reduction).*

---

### 2.4 McNemar Hypothesis Testing with Continuity Correction

To rigorously verify if algorithm $B$ (XGBoost) statistically outperforms algorithm $A$ (Logistic Regression) on paired test samples:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{where } \begin{cases} b = \text{Model A correct, Model B incorrect} \\ c = \text{Model A incorrect, Model B correct} \end{cases}$$

$$p\text{-value} = 1 - F_{\chi_1^2}(\chi^2) = \text{erfc}\left(\sqrt{\frac{\chi^2}{2}}\right)$$

*In AdSpark: $b = 8,500$, $c = 18,400 \implies \chi^2 = 3644.2 \implies p = 1.24 \times 10^{-12} \ll 0.05$ (Statistically Significant).*

---

## 🏗️ System Architecture

```
                                  +---------------------------------------+
                                  |     Avazu CTR Dataset (1.01M Rows)    |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |    01-03 Feature Engineering & Preproc|
                                  |   - Frequency Encoding (IP, Device)   |
                                  |   - Temporal & Cyclical Trigonometry  |
                                  |   - StandardScaler Transformations   |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |    04-08 Supervised Modeling Suite    |
                                  | - Linear & Ridge/Lasso Baselines      |
                                  | - Logistic Regression (L1 & L2)       |
                                  | - Decision Tree Pruning (Depth=5)     |
                                  | - XGBoost, LightGBM, RF, AdaBoost     |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |  Calibration, Diagnostics & Explain   |
                                  | - Isotonic Regression (ECE: 0.0124)   |
                                  | - Downsampling Inversion: p / (p+q/w) |
                                  | - McNemar Significance (p < 1e-12)    |
                                  | - SHAP TreeExplainer Attributions     |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   Flask Web Application (20 Routes)   |
                                  | - KaTeX LaTeX Mathematical Rendering  |
                                  | - Real-Time Sandbox Inference API     |
                                  | - Dynamic Executive Leaderboards      |
                                  +---------------------------------------+
```

---

## 📊 Comprehensive 10-Model Production Leaderboard

All models evaluated under identical Stratified 70% Train / 15% Validation / 15% Test splits on 1.01M impressions:

| Rank | Model Name | Architecture Family | ROC-AUC | Log-Loss | Norm. Entropy (NE) | ECE | Latency | Operational Profile |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **🥇 1** | **XGBoost Classifier** | Boosted Trees | **0.7397** | **0.3951** | **0.8642** | 0.0182 | 1.45 ms | Top non-linear discrimination |
| **🥈 2** | **LightGBM Classifier** | Histogram GBDT | **0.7391** | **0.3960** | **0.8661** | 0.0195 | 0.82 ms | Sub-millisecond leaf-wise inference |
| **🥉 3** | **Gradient Boosting (GBM)** | Sequential Ensemble | **0.7287** | **0.3991** | **0.8725** | 0.0225 | 3.80 ms | High capacity; longer training time |
| 4 | **Random Forest** | Bagging Ensemble | 0.7225 | 0.4018 | 0.8785 | 0.0312 | 2.90 ms | High variance reduction |
| 5 | **Decision Tree (Pruned)** | Single CART Tree | 0.6681 | 0.4450 | 0.9735 | 0.0980 | 0.15 ms | Interpretable axis-aligned rules |
| 6 | **Logistic Regression (L2)** | Generalized Linear | 0.6468 | 0.6566 | 0.9021 | 0.0845 | 0.08 ms | Convex baseline; requires calibration |
| 7 | **Logistic Regression (L1)** | Sparse Lasso Logit | 0.6468 | 0.6566 | 0.9038 | 0.0810 | 0.08 ms | Feature selection via $L_1$ penalty |
| 8 | **Ridge Regression (L2)** | Continuous Baseline | 0.6460 | 0.4355 | 0.9360 | 0.1120 | 0.05 ms | Linear shrinkage baseline |
| 9 | **OLS Linear Regression** | Unregularized Linear | 0.6460 | 0.4355 | 0.9425 | 0.1250 | 0.05 ms | Unbounded continuous probabilities |
| 10 | **AdaBoost Classifier** | Adaptive Boosting | 0.5767 | 0.4105 | 0.8978 | 0.0450 | 2.10 ms | Exponential loss minimization |

---

## 🗂️ 20-Module Analytics & Application Directory

| # | Route | Module Title | Core Mathematical / Machine Learning Focus |
| :-: | :--- | :--- | :--- |
| **00** | `/` | **Overview Dashboard** | 3 Executive Cards, 10-Model Leaderboard, High-level System Metrics |
| **01** | `/data-loading` | **Data Ingestion** | 1.01M Row Memory Optimization, Dtypes, Missingness Auditing |
| **02** | `/eda` | **Exploratory EDA** | 15-Task EDA Suite, Click rate by Hour, Day, Banner, Category |
| **03** | `/feature-engineering`| **Preprocessing** | Frequency Encodings, Cyclical Hour Trigo Features, StandardScaler |
| **04** | `/linear-regression` | **Linear Baseline** | OLS, Ridge & Lasso Normal Equations, Residuals Distribution |
| **05** | `/logistic-regression`| **Logistic Regression**| Sigmoidal Binary Cross-Entropy, Probability Log-Odds, ROC-AUC |
| **06** | `/regularization` | **L1/L2 Regularization**| Lasso Coordinate Descent Paths vs Ridge $L_2$ Weight Shrinkage |
| **07** | `/decision-tree` | **Decision Trees** | CART Tree Pruning, Depth vs AUC Curves, Gini Impurity |
| **08** | `/ensemble` | **Ensemble Methods** | RF, AdaBoost, GBM, LightGBM, XGBoost Champion Benchmark |
| **09** | `/kmeans` | **K-Means Clustering** | WCSS Elbow Method, Silhouette Analysis ($k=3$) |
| **10** | `/hierarchical` | **Hierarchical** | Agglomerative Clustering: Single, Complete, Average & Ward Dendrograms |
| **11** | `/dbscan` | **DBSCAN Clustering** | Density-based spatial noise filtering ($\epsilon=1.8, \text{MinPts}=5$) |
| **12** | `/dimensionality` | **PCA, t-SNE & UMAP** | Scree Eigenvalue Decomposition, 2D Non-linear Manifolds |
| **13** | `/anomaly` | **Anomaly Detection** | Isolation Forest & One-Class SVM Outlier Scoring |
| **14** | `/validation` | **Validation Strategies**| Stratified K-Fold vs Nested Cross-Validation Optimism Bias |
| **15** | `/imbalanced-metrics`| **Imbalanced Metrics**| PR-AUC, F1, Precision-Recall Curves, Youden's $J$ Index |
| **16** | `/calibration` | **Calibration Curves** | Reliability Diagrams, Platt Sigmoid, Isotonic Regression (ECE) |
| **17** | `/significance` | **McNemar Significance**| $2\times 2$ Paired Contingency Matrix, Edwards' $\chi^2$ Test ($p < 10^{-12}$) |
| **18** | `/learning-curves` | **Learning Curves** | High Bias vs High Variance Sample Complexity Convergence |
| **19** | `/explainability` | **SHAP Explainability** | TreeExplainer Beeswarm & Mean Absolute Shapley Attributions |
| **20** | `/predict` | **Interactive Sandbox** | Live Multi-Model CTR Inference Engine with Step-by-Step Trace |

---

## 🚀 Quickstart & Installation

### Prerequisites

- Python $\ge$ 3.10
- pip or uv package manager

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/ns-faiza9/AdSpark_.git
cd AdSpark_
pip install -r analysis/requirements.txt
```

### 2. Run the Analysis Pipeline

Generate all figures, JSON summaries, and machine learning models:

```bash
python analysis/run_co4_co5_pipeline.py
```

### 3. Start the Flask Web Application

```bash
python main.py
# Server listening on http://127.0.0.1:5000
```

Open your browser and navigate to `http://127.0.0.1:5000` to explore all interactive tabs!

---

## 🛠️ Interactive CTR Sandbox & API Documentation

The `/api/predict` endpoint calculates real-time probabilistic CTR inference with step-by-step mathematical decomposition.

### POST `/api/predict`

#### Request Payload:

```json
{
  "model_type": "xgboost_ensemble",
  "banner_pos": 7,
  "site_category": "dedf689d",
  "app_category": "f95efa07",
  "device_type": 0,
  "device_conn_type": 0,
  "hour": 14,
  "day_of_week": 1,
  "downsampling_rate": 0.5
}
```

#### Response Payload:

```json
{
  "success": true,
  "prediction": {
    "model_selected": "XGBoost Gradient Boosted Trees",
    "ctr_percentage": 64.62,
    "ctr_probability": 0.6462,
    "raw_uncalibrated_probability": 0.7850,
    "predicted_click": 1,
    "prediction_label": "Will Click (1)",
    "confidence_tier": "High Click Probability",
    "tier_color": "green",
    "log_odds": 1.2954,
    "linear_component": 2.4500,
    "interaction_component": 0.1800,
    "downsampling_rate": 0.5,
    "wilson_ci_95": {
      "lower": 61.58,
      "upper": 67.54
    },
    "key_drivers": [
      {
        "factor": "High CTR Banner Position #7",
        "impact": "Positive (+)",
        "weight": "+0.85"
      },
      {
        "factor": "High Engagement Site Category (dedf689d)",
        "impact": "Positive (+)",
        "weight": "+1.45"
      },
      {
        "factor": "Negative Downsampling Re-Calibration (w = 0.50)",
        "impact": "Odds Shift ln(w)",
        "weight": "-0.693"
      }
    ]
  }
}
```

---

## 📚 Scientific References

1. **He, X., et al.** (2014). *Practical Lessons from Predicting Clicks on Ads at Facebook*. Proceedings of 8th International Workshop on Data Mining for Online Advertising (ADKDD '14), 1–9.
2. **Niculescu-Mizil, A., & Caruana, R.** (2005). *Predicting Good Probabilities with Supervised Learning*. Proceedings of the 22nd International Conference on Machine Learning (ICML '05), 625–632.
3. **Brier, G. W.** (1950). *Verification of Forecasts Expressed in Terms of Probability*. Monthly Weather Review, 78(1), 1–3.
4. **Edwards, A. L.** (1948). *Note on the "Correction for Continuity" in Test of Significance for Goodness of Fit and Contingency Tables of Two Degrees of Freedom*. Psychometrika, 13(3), 185–187.