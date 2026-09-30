# NASA Zooniverse Active Asteroids
# Error Analysis
#
# Identifies and categorizes:
#   - False positives (predicted active, actually inactive)
#   - False negatives (predicted inactive, actually active)
#   - High-confidence wrong predictions
#   - Uncertainty-stratified error patterns
#
# Usage:
#   python src/analysis/error_analysis.py \
#     --predictions results/all_predictions.csv \
#     --metadata data/metadata/subjects.csv \
#     --output results/error_analysis.csv

import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def classify_errors(
    predictions_df: pd.DataFrame,
    high_conf_threshold: float = 0.8,
    low_conf_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Classify each prediction into error categories.
    
    Error types:
      TP: True Positive  (correct active detection)
      TN: True Negative  (correct inactive classification)
      FP: False Positive (galaxy/artifact mistaken for active asteroid)
      FN: False Negative (active asteroid missed)
      HC-FP: High-confidence false positive (pred > high_conf_threshold, true=0)
      HC-FN: High-confidence false negative (pred < low_conf_threshold, true=1)
    """
    df = predictions_df.copy()
    
    # Basic error classification
    conditions = [
        (df["true_label"] == 1) & (df["pred_prob"] >= 0.5),  # TP
        (df["true_label"] == 0) & (df["pred_prob"] < 0.5),   # TN
        (df["true_label"] == 0) & (df["pred_prob"] >= 0.5),  # FP
        (df["true_label"] == 1) & (df["pred_prob"] < 0.5),   # FN
    ]
    choices = ["TP", "TN", "FP", "FN"]
    df["error_type"] = np.select(conditions, choices, default="Unknown")
    
    # High-confidence sub-categorization
    df["high_conf_error"] = False
    hc_fp_mask = (df["error_type"] == "FP") & (df["pred_prob"] > high_conf_threshold)
    hc_fn_mask = (df["error_type"] == "FN") & (df["pred_prob"] < low_conf_threshold)
    df.loc[hc_fp_mask, "high_conf_error"] = True
    df.loc[hc_fn_mask, "high_conf_error"] = True
    df.loc[hc_fp_mask, "error_type"] = "HC-FP"
    df.loc[hc_fn_mask, "error_type"] = "HC-FN"
    
    # Confidence level
    df["confidence_level"] = pd.cut(
        df["pred_prob"],
        bins=[0, 0.2, 0.35, 0.65, 0.8, 1.0],
        labels=["very_low", "low", "uncertain", "high", "very_high"],
        include_lowest=True,
    )
    
    # Possible visual causes (based on uncertainty category and error type)
    df["possible_visual_causes"] = df.apply(
        assign_possible_causes, axis=1
    )
    
    return df


def assign_possible_causes(row: pd.Series) -> str:
    """
    Assign possible visual causes to error types based on uncertainty category.
    
    IMPORTANT: These are hypothesized causes based on published literature
    about astronomical image classification errors. They are NOT confirmed
    by visual inspection unless explicitly noted.
    Reference: ZamirDev report, arXiv:2403.09768
    """
    error_type = str(row.get("error_type", ""))
    unc_cat = str(row.get("uncertainty_category", ""))
    
    if error_type in ("FP", "HC-FP"):
        # False positive causes from literature
        causes = [
            "elongated_background_galaxy",
            "ccd_artifact_or_streak",
            "image_diffraction_spike",
            "stellar_smearing",
            "cosmic_ray",
        ]
        # Catalog-matched negatives might have faint unreported activity
        if unc_cat == "unmatched_negative":
            causes.append("possible_undiscovered_active_object")
        return "; ".join(causes[:3])
    
    elif error_type in ("FN", "HC-FN"):
        # False negative causes
        causes = []
        if unc_cat == "catalog_matched_positive":
            # Object is known active but THIS image may not show activity
            causes.extend([
                "activity_not_visible_in_this_epoch",
                "faint_activity_below_SNR_threshold",
                "object_level_label_mismatch_with_image_evidence",
            ])
        elif unc_cat == "teaching_positive":
            causes.extend([
                "faint_activity",
                "model_uncertainty_low_SNR",
                "distribution_shift_from_training",
            ])
        else:
            causes.extend([
                "low_signal_to_noise",
                "ambiguous_morphology",
            ])
        return "; ".join(causes[:3])
    
    return "N/A"


def generate_error_analysis_report(
    error_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Generate comprehensive error analysis report.
    
    Reports:
    1. Overall error distribution
    2. Error rates by uncertainty category
    3. High-confidence error analysis
    4. Model confidence distribution by error type
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_predictions": len(error_df),
    }
    
    # Overall error distribution
    error_dist = error_df["error_type"].value_counts().to_dict()
    report["error_distribution"] = error_dist
    
    # Rates
    n = len(error_df)
    for et in ["TP", "TN", "FP", "FN", "HC-FP", "HC-FN"]:
        count = error_dist.get(et, 0)
        report[f"{et}_rate"] = round(count / max(n, 1), 4)
    
    # Error rates by uncertainty category
    report["error_by_uncertainty"] = {}
    for cat in error_df["uncertainty_category"].unique():
        cat_df = error_df[error_df["uncertainty_category"] == cat]
        cat_errors = cat_df["error_type"].value_counts().to_dict()
        report["error_by_uncertainty"][cat] = {
            "n_samples": len(cat_df),
            "error_counts": cat_errors,
            "mean_pred_prob": round(float(cat_df["pred_prob"].mean()), 4),
        }
    
    # High confidence errors
    hc_errors = error_df[error_df["high_conf_error"] == True]
    report["high_confidence_errors"] = {
        "total": len(hc_errors),
        "by_type": hc_errors["error_type"].value_counts().to_dict(),
        "by_uncertainty": hc_errors["uncertainty_category"].value_counts().to_dict()
        if len(hc_errors) > 0 else {},
    }
    
    # Key findings
    report["key_findings"] = []
    
    fp_count = error_dist.get("FP", 0) + error_dist.get("HC-FP", 0)
    fn_count = error_dist.get("FN", 0) + error_dist.get("HC-FN", 0)
    
    if fp_count > fn_count:
        report["key_findings"].append(
            f"Model has more false positives ({fp_count}) than false negatives ({fn_count}), "
            "consistent with high recall / low precision operating point."
        )
    else:
        report["key_findings"].append(
            f"Model has more false negatives ({fn_count}) than false positives ({fp_count}), "
            "consistent with high precision / low recall operating point."
        )
    
    # Check FN concentration in catalog-matched positives
    if "catalog_matched_positive" in report["error_by_uncertainty"]:
        cat_data = report["error_by_uncertainty"]["catalog_matched_positive"]
        fn_in_cat = cat_data["error_counts"].get("FN", 0) + cat_data["error_counts"].get("HC-FN", 0)
        n_cat = cat_data["n_samples"]
        if n_cat > 0:
            fn_rate = fn_in_cat / n_cat
            report["key_findings"].append(
                f"Catalog-matched positives: FN rate = {fn_rate:.2%} ({fn_in_cat}/{n_cat}). "
                "These represent the label uncertainty problem: object-level labels applied "
                "to images where activity may not be visible."
            )
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Error analysis for Active Asteroids predictions")
    parser.add_argument("--predictions", type=str, default="results/all_predictions.csv")
    parser.add_argument("--metadata", type=str, default="data/metadata/subjects.csv")
    parser.add_argument("--output", type=str, default="results/error_analysis.csv")
    parser.add_argument("--report", type=str, default="results/error_analysis_report.json")
    parser.add_argument("--threshold_high", type=float, default=0.8)
    parser.add_argument("--threshold_low", type=float, default=0.2)
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    predictions_path = project_root / args.predictions
    metadata_path = project_root / args.metadata
    output_path = project_root / args.output
    report_path = project_root / args.report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("NASA Zooniverse Active Asteroids — Error Analysis")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    if not predictions_path.exists():
        print(f"ERROR: Predictions not found at {predictions_path}")
        print("Run train_baseline.py first.")
        return
    
    preds_df = pd.read_csv(predictions_path)
    print(f"Loaded {len(preds_df)} predictions")
    
    # Merge with metadata
    meta_df = pd.read_csv(metadata_path)
    merge_cols = [c for c in ["designation", "label_source", "label_confidence"] 
                  if c in meta_df.columns and c not in preds_df.columns]
    if merge_cols:
        preds_df = preds_df.merge(meta_df[["subject_id"] + merge_cols], on="subject_id", how="left")
    
    # Classify errors
    error_df = classify_errors(
        preds_df,
        high_conf_threshold=args.threshold_high,
        low_conf_threshold=args.threshold_low,
    )
    
    # Save error-classified predictions
    error_df.to_csv(output_path, index=False)
    print(f"Error-classified predictions saved to {output_path}")
    
    # Generate report
    report = generate_error_analysis_report(error_df, output_path.parent)
    
    print("\n=== Error Distribution ===")
    for et, count in report["error_distribution"].items():
        print(f"  {et}: {count}")
    
    print("\n=== Error by Uncertainty Category ===")
    for cat, data in report["error_by_uncertainty"].items():
        print(f"  {cat}: n={data['n_samples']}, errors={data['error_counts']}")
    
    print("\n=== Key Findings ===")
    for finding in report["key_findings"]:
        print(f"  - {finding}")
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")
    
    return report


if __name__ == "__main__":
    main()
