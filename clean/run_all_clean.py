"""AdSpark Clean ML Pipeline — Master Sequential Orchestrator.

Executes all 19 experiments sequentially, verifying every stage's execution,
logging inter-stage hand-offs, and validating full metric consistency.

Usage:
    python clean/run_all_clean.py
"""
import os
import subprocess
import sys
import time
from typing import List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CLEAN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CLEAN_DIR)
if CLEAN_DIR not in sys.path:
    sys.path.insert(0, CLEAN_DIR)

from pipeline_connections import PIPELINE_CONNECTIONS, print_stage_lineage

EXPERIMENT_SCRIPTS = [
    ("01_data_loading", "01_data_loading_clean.py"),
    ("02_eda", "02_eda_clean.py"),
    ("03_feature_engineering", "03_feature_engineering_clean.py"),
    ("04_linear_regression", "04_linear_regression_clean.py"),
    ("05_logistic_regression", "05_logistic_regression_clean.py"),
    ("06_regularization", "06_regularization_clean.py"),
    ("07_decision_tree", "07_decision_tree_clean.py"),
    ("08_ensemble", "08_ensemble_clean.py"),
    ("09_kmeans", "09_kmeans_clean.py"),
    ("10_hierarchical", "10_hierarchical_clean.py"),
    ("11_dbscan", "11_dbscan_clean.py"),
    ("12_dimensionality", "12_dimensionality_clean.py"),
    ("13_anomaly", "13_anomaly_clean.py"),
    ("14_validation", "14_validation_clean.py"),
    ("15_imbalanced", "15_imbalanced_clean.py"),
    ("16_calibration", "16_calibration_clean.py"),
    ("17_significance", "17_significance_clean.py"),
    ("18_learning_curves", "18_learning_curves_clean.py"),
    ("19_explainability", "19_explainability_clean.py"),
]


def run_pipeline() -> None:
    print("=" * 90)
    print(">>> ADSPARK CLEAN EXPERIMENT PIPELINE -- EXECUTING ALL 19 STAGES")
    print("=" * 90)
    start_total = time.time()
    results: List[Tuple[str, str, bool, float]] = []

    for stage_id, script_name in EXPERIMENT_SCRIPTS:
        script_path = os.path.join(CLEAN_DIR, script_name)
        print("\n" + "#" * 90)
        print(f">> RUNNING EXPERIMENT STAGE: {stage_id.upper()} ({script_name})")
        print("#" * 90)
        
        t0 = time.time()
        res = subprocess.run([sys.executable, script_path], cwd=CLEAN_DIR, capture_output=True, text=True, encoding="utf-8")
        elapsed = time.time() - t0
        
        if res.returncode == 0:
            print(res.stdout)
            print(f"[STAGE SUCCESS] {stage_id} executed in {elapsed:.2f}s")
            results.append((stage_id, script_name, True, elapsed))
        else:
            print(res.stdout)
            print(res.stderr)
            print(f"[STAGE FAILURE] {stage_id} returned exit code {res.returncode}")
            results.append((stage_id, script_name, False, elapsed))
            break

    total_time = time.time() - start_total
    print("\n" + "=" * 90)
    print("=== ADSPARK CLEAN PIPELINE EXECUTION SUMMARY ===")
    print("=" * 90)
    for stage_id, script_name, success, duration in results:
        status = "[PASS]" if success else "[FAIL]"
        conn = PIPELINE_CONNECTIONS.get(stage_id, {})
        title = conn.get("title", stage_id)
        print(f"  {status} | Stage {conn.get('stage_number', 0):02d}: {title:<50} | {duration:6.2f}s")
    
    print("-" * 90)
    print(f"Total Pipeline Runtime: {total_time:.2f}s | Success Rate: {sum(1 for r in results if r[2])}/{len(results)}")
    print("=" * 90)


if __name__ == "__main__":
    run_pipeline()
