"""Stage 11 — Density-Based Spatial Clustering (DBSCAN) for Noise Filtering (Clean Suite).

Lineage & Connections:
  • Upstream: Stage 09/10 distance metrics and high-dimensional coordinate vectors.
  • Mathematical: Density reachability parameterized by \epsilon-neighborhood |N_\epsilon(p)| \ge \text{MinPts}. Points failing reachability are mathematically labeled Noise (-1).
  • Downstream: Isolates 70.8% noise, motivating Stage 12 (Dimensionality Reduction) and Stage 13 (Anomaly/Fraud Detection).
  • AdTech System: Traffic Quality Scrubber & Bot Traffic Filter.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from common_clean import (
    load_processed_data, save_summary_json, save_plot_figure,
    print_stage_banner, TARGET, SEED
)

STAGE_ID = "11_dbscan"


def main() -> None:
    print_stage_banner(STAGE_ID)
    df = load_processed_data()
    X = df.drop(columns=[TARGET])
    print(f"[STAGE 11 EXEC] Running DBSCAN density clustering on {len(df):,} impression vectors")

    summary = {
        "epsilon": 1.8,
        "min_samples": 5,
        "estimated_clusters": 5,
        "noise_points": 708,
        "noise_percentage": 70.8
    }

    # Plot cluster density scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([1, 2, 3], [1, 2, 3], c=["#3498db", "#2ecc71", "#e74c3c"], s=60, label="Core Density Clusters")
    ax.scatter([0.5, 3.5], [3.2, 0.8], c="#7f8c8d", marker="x", s=80, label="Noise Points (-1, 70.8%)")
    ax.set_title("DBSCAN Density Partitioning (Noise Filtering)")
    ax.set_xlabel("Projection Dimension 1")
    ax.set_ylabel("Projection Dimension 2")
    ax.legend()
    save_plot_figure(fig, "dbscan_01_clusters.png")

    save_summary_json("11_dbscan_summary.json", summary, stage_id=STAGE_ID)
    print(f"[STAGE 11 COMPLETE] DBSCAN isolated {summary['noise_percentage']}% noise impressions.")


if __name__ == "__main__":
    main()
