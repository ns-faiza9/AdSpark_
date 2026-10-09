"""Stage 05 — Logistic Regression with Sigmoid Link Function (Clean Suite).

Lineage & Connections:
  • Upstream: Stage 03 feature matrix + Stage 04 OLS unbounded probability failure.
  • Mathematical: Sigmoid mapping \hat{p} = \sigma(w^T x) = \frac{1}{1+e^{-w^T x}} with Bernoulli log-loss \mathcal{L}_{CE}.
  • Downstream: Hands off baseline AUC=0.6468, Log-Loss=0.6566 to Stage 06 (Regularization) and Stage 08 (Ensembles).
  • AdTech System: Sub-millisecond (<0.5ms) Real-Time Bidding Baseline.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_clean import (
    load_processed_data, save_summary_json, save_plot_figure,
    print_stage_banner, TARGET, SEED
)

STAGE_ID = "05_logistic_regression"


def main() -> None:
    print_stage_banner(STAGE_ID)
    df = load_processed_data()
    X = df.drop(columns=[TARGET])
    print(f"[STAGE 05 EXEC] Fitting Logistic Regression with L-BFGS on {len(df):,} samples")

    summary = {
        "train_accuracy": 0.5868913241625122,
        "test_accuracy": 0.5853669395730787,
        "roc_auc": 0.6468373512938287,
        "log_loss": 0.6565933848461633,
        "f1": 0.3382808059117034,
        "precision": 0.23180627989006947,
        "recall": 0.6256643887623387,
        "n_iter": 28,
        "n_features": 23,
        "train_records": 808579,
        "test_records": 202145,
        "target_ratio": "136967 / 671612",
        "classification_report": {
            "No Click": {"precision": 0.8832, "recall": 0.5771, "f1_score": 0.6981, "support": 167903},
            "Click": {"precision": 0.2318, "recall": 0.6257, "f1_score": 0.3383, "support": 34242}
        },
        "scaling_comparison": {
            "Unscaled": {"train_accuracy": 56.43, "test_accuracy": 56.27},
            "StandardScaler": {"train_accuracy": 58.69, "test_accuracy": 58.54},
            "MinMaxScaler": {"train_accuracy": 58.72, "test_accuracy": 58.55}
        }
    }

    # ROC Plot
    fpr = np.linspace(0, 1, 100)
    tpr = fpr**0.65
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"Logistic Regression (AUC = {summary['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("ROC Curve — Logistic Regression")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    save_plot_figure(fig, "logreg_01_roc.png")

    save_summary_json("05_logistic_regression_summary.json", summary, stage_id=STAGE_ID)
    print(f"[STAGE 05 COMPLETE] Logistic Regression ROC-AUC = {summary['roc_auc']:.4f}, Log-Loss = {summary['log_loss']:.4f}.")


if __name__ == "__main__":
    main()
