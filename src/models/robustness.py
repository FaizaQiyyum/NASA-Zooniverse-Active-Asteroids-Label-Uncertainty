# NASA Zooniverse Active Asteroids
# Robustness & Multi-Seed Evaluation Pipeline
#
# Evaluates baseline and label uncertainty models across multiple random seeds
# to quantify performance variance and avoid reporting lucky single-seed results.
#
# Usage:
#   python src/models/robustness.py \
#     --metadata data/metadata/subjects.csv \
#     --seeds 42 101 2024 \
#     --output results/robustness_results.json

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.train_baseline import train_logistic


def run_multi_seed_evaluation(
    metadata_df: pd.DataFrame,
    seeds: list = [42, 101, 2024],
    use_synthetic: bool = True,
) -> dict:
    """
    Run evaluation across multiple random seeds.
    Measures standard deviation and mean for test metrics.
    """
    print(f"\n--- Multi-Seed Robustness Evaluation (Seeds: {seeds}) ---")
    
    results_by_seed = []
    
    for s in seeds:
        print(f"\nEvaluating Seed: {s}")
        # Train logistic baseline with seed
        res = train_logistic(
            metadata_df,
            use_synthetic=use_synthetic,
            seed=s,
        )
        
        test_m = res["test_metrics"]
        val_m = res["val_metrics"]
        
        results_by_seed.append({
            "seed": s,
            "val_roc_auc": val_m["roc_auc"],
            "val_pr_auc": val_m["pr_auc"],
            "test_roc_auc": test_m["roc_auc"],
            "test_pr_auc": test_m["pr_auc"],
            "test_f1": test_m["f1_at_0_5"],
            "test_recall": test_m["recall_at_0_5"],
            "test_precision": test_m["precision_at_0_5"],
        })
    
    summary_df = pd.DataFrame(results_by_seed)
    
    aggregated = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "runs": results_by_seed,
        "summary": {
            "test_roc_auc_mean": round(float(summary_df["test_roc_auc"].mean()), 4),
            "test_roc_auc_std": round(float(summary_df["test_roc_auc"].std()), 4),
            "test_pr_auc_mean": round(float(summary_df["test_pr_auc"].mean()), 4),
            "test_pr_auc_std": round(float(summary_df["test_pr_auc"].std()), 4),
            "test_f1_mean": round(float(summary_df["test_f1"].mean()), 4),
            "test_f1_std": round(float(summary_df["test_f1"].std()), 4),
        },
        "statistical_caution": (
            f"Evaluated across {len(seeds)} random seeds. With test set size n={len(metadata_df[metadata_df['split']=='test'])}, "
            "metrics exhibit finite-sample variance. Small sample size limits strong statistical significance claims."
        )
    }
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Multi-seed robustness evaluation")
    parser.add_argument("--metadata", type=str, default="data/metadata/subjects.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024])
    parser.add_argument("--output", type=str, default="results/robustness_results.json")
    parser.add_argument("--use_synthetic", action="store_true", default=True)
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent.parent
    meta_path = root / args.metadata
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(meta_path)
    res = run_multi_seed_evaluation(df, seeds=args.seeds, use_synthetic=args.use_synthetic)
    
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    
    print(f"\nRobustness results saved to {out_path}")
    print(f"Test ROC-AUC: {res['summary']['test_roc_auc_mean']:.4f} +/- {res['summary']['test_roc_auc_std']:.4f}")
    print(f"Test PR-AUC:  {res['summary']['test_pr_auc_mean']:.4f} +/- {res['summary']['test_pr_auc_std']:.4f}")


if __name__ == "__main__":
    main()
