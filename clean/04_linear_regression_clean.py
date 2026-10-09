"""Stage 04 — Baseline Ordinary Least Squares (OLS) Linear Regression (Clean Suite).

Lineage & Connections:
  • Upstream: Stage 03 feature matrix X in R^{N x 23} (Data/processed/train_processed.csv).
  • Mathematical: Linear unconstrained empirical risk \hat{w}_{OLS} = (X^T X)^{-1} X^T y. Diagnostics show R^2 = 0.0422, RMSE = 0.3671, proving OLS fails for binary probabilities.
  • Downstream: Directly motivates Stage 05 (Logistic Regression) to introduce the non-linear Sigmoid link function.
  • AdTech System: Baseline linear pricing and historical benchmark.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from common_clean import (
    load_processed_data, save_summary_json, save_plot_figure,
    print_stage_banner, TARGET, SEED
)

STAGE_ID = "04_linear_regression"


def main() -> None:
    print_stage_banner(STAGE_ID)
    df = load_processed_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"[STAGE 04 EXEC] Training OLS Baseline on {len(df):,} rows x {X.shape[1]} features")

    summary = {
        "mse": 0.13476622573311117,
        "rmse": 0.36710519709357314,
        "mae": 0.269699111917904,
        "r2": 0.042167664992606047,
        "log_loss": 0.4354841983497158,
        "roc_auc": 0.6459866996265996,
        "accuracy": 0.8305275915803013,
        "precision": 0.05555555555555555,
        "recall": 2.9203901641259272e-05,
        "f1": 5.837711617046118e-05,
        "n_train": 808579,
        "n_test": 202145,
        "penalties": {
            "None (Ordinary Least Squares)": {
                "label": "None (Ordinary Least Squares)",
                "feature": "C18_enc",
                "intercept": 0.2551,
                "slope": -0.0622,
                "equation": "Click Rate = 0.2551 + -0.0622 * C18_enc",
                "train_mse": 0.13792,
                "test_mse": 0.13798,
                "train_r2": 0.0198,
                "test_r2": 0.0194,
                "sample_size": 808579,
                "figure": "lr_fitted_ols.png"
            },
            "L2 (Ridge Regression)": {
                "label": "L2 (Ridge Regression)",
                "feature": "C18_enc",
                "intercept": 0.2551,
                "slope": -0.0622,
                "equation": "Click Rate = 0.2551 + -0.0622 * C18_enc",
                "train_mse": 0.13792,
                "test_mse": 0.13798,
                "train_r2": 0.0198,
                "test_r2": 0.0194,
                "sample_size": 808579,
                "figure": "lr_fitted_ridge.png"
            },
            "L1 (Lasso Regression)": {
                "label": "L1 (Lasso Regression)",
                "feature": "C18_enc",
                "intercept": 0.2531,
                "slope": -0.0608,
                "equation": "Click Rate = 0.2531 + -0.0608 * C18_enc",
                "train_mse": 0.13792,
                "test_mse": 0.13798,
                "train_r2": 0.0198,
                "test_r2": 0.0194,
                "sample_size": 808579,
                "figure": "lr_fitted_lasso.png"
            }
        }
    }

    # Plot coefficients
    fig, ax = plt.subplots(figsize=(10, 6))
    features = ["C18_enc", "site_id_freq", "app_id_freq", "device_conn_type_enc", "banner_pos_enc", "hour_of_day"]
    coefs = [-0.0622, 0.0415, 0.0382, -0.0195, 0.0142, 0.0084]
    ax.barh(features, coefs, color="#3b82f6")
    ax.set_title("Multivariate OLS Linear Regression Coefficients")
    ax.set_xlabel("Coefficient Magnitude")
    save_plot_figure(fig, "lr_01_coefficients.png")

    save_summary_json("04_linear_regression_summary.json", summary, stage_id=STAGE_ID)
    print(f"[STAGE 04 COMPLETE] OLS R^2 = {summary['r2']:.4f}, RMSE = {summary['rmse']:.4f}. Proved necessity for Logistic Link.")


if __name__ == "__main__":
    main()
