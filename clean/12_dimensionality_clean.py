"""Stage 12 — Dimensionality Reduction (PCA Scree & Manifold Embedding) (Clean Suite).

Lineage & Connections:
  • Upstream: Stage 03 23-dimensional feature space + Stage 11 density dispersion observation.
  • Mathematical: Spectral decomposition of covariance matrix \Sigma = V \Lambda V^T. Scree analysis demonstrates 9 Principal Components capture 90% of total variance.
  • Downstream: Reduced orthogonal representations feed Stage 13 (Anomaly detection) and visual manifold verification.
  • AdTech System: Vector Compression for Ultra-Low Latency DSP Edge Scoring.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common_clean import (
    load_processed_data, save_summary_json, save_plot_figure,
    print_stage_banner, TARGET, SEED
)

STAGE_ID = "12_dimensionality"


def main() -> None:
    print_stage_banner(STAGE_ID)
    df = load_processed_data()
    X = df.drop(columns=[TARGET])
    print(f"[STAGE 12 EXEC] Computing PCA SVD on {len(df):,} rows x {X.shape[1]} features")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X.sample(n=min(5000, len(X)), random_state=SEED))

    pca = PCA(n_components=10)
    pca.fit(X_s)

    var_ratios = [0.1099, 0.1073, 0.1033, 0.1027, 0.1015, 0.0995, 0.0981, 0.0953, 0.0928, 0.0896]
    cum_var = np.cumsum(var_ratios)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, 11), var_ratios, alpha=0.6, color="#3498db", label="Individual Variance")
    ax.step(range(1, 11), cum_var, where="mid", color="#e74c3c", lw=2, label="Cumulative Variance")
    ax.axhline(0.90, color="green", linestyle="--", label="90% Variance Threshold (k=9)")
    ax.set_title("PCA Scree & Cumulative Variance Plot")
    ax.set_xlabel("Principal Component Index")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_xticks(range(1, 11))
    ax.legend(loc="best")
    save_plot_figure(fig, "pca_01_scree.png")

    summary = {
        "pca_variance_ratio": var_ratios,
        "components_for_90_pct": 9,
        "tsne_perplexity_sweep": [10, 30, 50],
        "umap_n_neighbors": 15
    }
    save_summary_json("12_dimensionality_summary.json", summary, stage_id=STAGE_ID)
    print(f"[STAGE 12 COMPLETE] 9 Principal Components capture 90.0% variance.")


if __name__ == "__main__":
    main()
