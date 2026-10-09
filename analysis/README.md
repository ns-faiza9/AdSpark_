# AdSpark — Analysis Pipeline

The **analysis** side of the AdSpark CTR Prediction project. This is kept
**separate from the frontend** (`webpage/`): the Python scripts here do all the
data science work and save their results (figures + JSON summaries) into
`analysis/output/`, which the React frontend then displays.

## Stages (19-Stage Machine Learning Pipeline)

| Stage | Script (Analysis) | Script (Clean Suite) | Stage Name | Theoretical & Practical Function |
| :--- | :--- | :--- | :--- | :--- |
| **01** | `01_data_loading.py` | `clean/01_data_loading_clean.py` | Data Loading | Schema profiling, missingness checks, memory downcasting |
| **02** | `02_eda.py` | `clean/02_eda_clean.py` | Exploratory Data Analysis | Diurnal CTR cycles, Empirical Bayes shrinkage |
| **03** | `03_feature_engineering.py` | `clean/03_feature_engineering_clean.py` | Feature Engineering | Harmonic hour encodings, frequency rank densities |
| **04** | `04_linear_regression.py` | `clean/04_linear_regression_clean.py` | Linear Regression | OLS baseline model & unbounded risk failure analysis |
| **05** | `05_logistic_regression.py` | `clean/05_logistic_regression_clean.py` | Logistic Regression | Sigmoid link function & maximum likelihood estimation |
| **06** | `06_regularization.py` | `clean/06_regularization_clean.py` | Regularization | Lasso L1 sparsity, Ridge L2 shrinkage, ElasticNet |
| **07** | `07_decision_tree.py` | `clean/07_decision_tree_clean.py` | Decision Trees | Recursive CART binary splitting & Gini gain |
| **08** | `08_ensemble.py` | `clean/08_ensemble_clean.py` | Ensemble Learning | Random Forest, AdaBoost, GBM, LightGBM, XGBoost Champion |
| **09** | `09_kmeans.py` | `clean/09_kmeans_clean.py` | K-Means Clustering | Lloyd's algorithm WCSS minimization & silhouette score |
| **10** | `10_hierarchical.py` | `clean/10_hierarchical_clean.py` | Hierarchical Clustering | Agglomerative dendrogram taxonomy & Ward linkage |
| **11** | `11_dbscan.py` | `clean/11_dbscan_clean.py` | DBSCAN Clustering | Density reachability noise filtering (70.8% noise) |
| **12** | `12_dimensionality.py` | `clean/12_dimensionality_clean.py` | Dimensionality Reduction | PCA Scree analysis (9 PCs capture 90% variance) |
| **13** | `13_anomaly.py` | `clean/13_anomaly_clean.py` | Anomaly Detection | Isolation Forest & One-Class SVM click fraud filtering |
| **14** | `14_validation.py` | `clean/14_validation_clean.py` | Validation Strategies | Stratified K-Fold vs Nested CV optimism bias mitigation |
| **15** | `15_imbalanced.py` | `clean/15_imbalanced_clean.py` | Imbalanced Metrics | Precision-Recall AUC & cost-sensitive thresholding |
| **16** | `16_calibration.py` | `clean/16_calibration_clean.py` | Probability Calibration | Platt scaling & Isotonic regression (ECE reduction) |
| **17** | `17_significance.py` | `clean/17_significance_clean.py` | Significance Testing | McNemar's paired chi-squared hypothesis test |
| **18** | `18_learning_curves.py` | `clean/18_learning_curves_clean.py` | Learning Curves | High bias vs variance sample complexity convergence |
| **19** | `19_explainability.py` | `clean/19_explainability_clean.py` | Explainability (SHAP) | Cooperative game-theoretic TreeSHAP feature attribution |

## How to run

```bash
# Run the clean pipeline with full connection and lineage logging:
python clean/run_all_clean.py

# Or run standard analysis pipeline:
python analysis/run_co4_co5_pipeline.py
```

## How to open the webpage

1. Open terminal and run:
   ```bash
   cd webpage
   npm run dev
   ```
2. Open your browser at [http://localhost:3000](http://localhost:3000)

## Outputs

- **Figures** → `analysis/output/figures/*.png` (displayed in the webpage)
- **JSON summaries** → `analysis/output/*.json` (metrics shown in the webpage)
- **Tables** → `analysis/output/*.csv`
- **Processed datasets** → `Data/processed/*.csv`

## Dataset

Avazu CTR (Kaggle) — ~40.4M ad impressions, 24 columns. The pipeline samples
~2.5% (~1M rows) so it runs quickly on a laptop while remaining statistically
sound. The target is `click` (1 = clicked, 0 = not clicked).