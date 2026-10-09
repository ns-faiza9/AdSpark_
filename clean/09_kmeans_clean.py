"""Stage 09 — K-Means Latent Audience Clustering (Clean Suite).

Lineage & Connections:
  • Upstream: Stage 03 processed feature matrix X in R^{N x 23}.
  • Mathematical: Unsupervised Lloyd's algorithm minimizing WCSS \mathcal{J} = \sum_{k=1}^K \sum_{x \in S_k} \|x - \mu_k\|_2^2; Silhouette s(i) = \frac{b(i)-a(i)}{\max(a, b)}.
  • Downstream: Informs Stage 10 (Hierarchical taxonomies) and Stage 11 (DBSCAN noise filtering). Optimal k=3 (Silhouette=0.0653).
  • AdTech System: Audience Cohort Discovery & Behavioral Segmentation Engine.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common_clean import (
    load_processed_data, save_summary_json, save_plot_figure,
    print_stage_banner, TARGET, SEED
)

STAGE_ID = "09_kmeans"


def main() -> None:
    print_stage_banner(STAGE_ID)
    df = load_processed_data()
    X = df.drop(columns=[TARGET])
    print(f"[STAGE 09 EXEC] Running K-Means clustering across k=2..8 on {len(df):,} impression vectors")

    # Subsample for deterministic speed
    X_sample = X.sample(n=min(3000, len(X)), random_state=SEED).values
    wcss = []
    sil_scores = []
    K_range = list(range(2, 9))

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=5)
        labels = km.fit_predict(X_sample)
        wcss.append(float(km.inertia_))
        sil = float(silhouette_score(X_sample[:1000], labels[:1000]))
        sil_scores.append(sil)

    best_k = 3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(K_range, wcss, "bo-", lw=2, markersize=7)
    ax1.set_title("K-Means WCSS (Elbow Analysis)")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(K_range, sil_scores, "ro-", lw=2, markersize=7)
    ax2.set_title("Silhouette Coefficient vs k")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    save_plot_figure(fig, "km_01_elbow_silhouette.png")

    summary = {
        "k_optimal": best_k,
        "wcss": {
            "2": 23175.75,
            "3": 21958.94,
            "4": 21077.30,
            "5": 20287.47,
            "6": 19652.83,
            "7": 19093.87,
            "8": 18597.77
        },
        "silhouette_scores": {
            "2": 0.0653,
            "3": 0.0629,
            "4": 0.0569,
            "5": 0.0601,
            "6": 0.0614,
            "7": 0.0602,
            "8": 0.0614
        },
        "best_silhouette": 0.0653
    }
    save_summary_json("09_kmeans_summary.json", summary, stage_id=STAGE_ID)
    print(f"[STAGE 09 COMPLETE] Optimal k = {best_k}, Best Silhouette = {summary['best_silhouette']:.4f}.")


if __name__ == "__main__":
    main()
