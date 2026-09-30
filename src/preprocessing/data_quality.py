# NASA Zooniverse Active Asteroids
# Data Quality Checker
#
# Performs automated quality checks on the subject metadata:
#   - Duplicate detection
#   - Missing value audit
#   - Label consistency
#   - Class balance analysis
#   - Split leakage check
#
# Usage:
#   python src/preprocessing/data_quality.py \
#     --metadata data/metadata/subjects.csv \
#     --output results/data_quality_report.json

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def check_duplicates(df: pd.DataFrame) -> dict:
    """Check for duplicate subject IDs and duplicate image URLs."""
    results = {}
    
    # Duplicate subject IDs
    dup_ids = df.duplicated(subset=["subject_id"], keep=False)
    results["duplicate_subject_ids"] = int(dup_ids.sum())
    results["unique_subject_ids"] = int(df["subject_id"].nunique())
    
    # Duplicate image URLs (different subjects, same image)
    if "image_url" in df.columns:
        url_col = df["image_url"].dropna()
        url_col = url_col[url_col != ""]
        dup_urls = url_col.duplicated(keep=False)
        results["duplicate_image_urls"] = int(dup_urls.sum())
    else:
        results["duplicate_image_urls"] = "N/A (no image_url column)"
    
    # Duplicate designations per split (potential leakage)
    if "designation" in df.columns and "split" in df.columns:
        desig_splits = df.groupby("designation")["split"].nunique()
        cross_split = desig_splits[desig_splits > 1]
        results["designations_in_multiple_splits"] = int(len(cross_split))
        results["cross_split_designations"] = cross_split.index.tolist()[:10]
    
    return results


def check_missing_values(df: pd.DataFrame) -> dict:
    """Audit missing values across all columns."""
    results = {}
    
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        n_empty_str = int((df[col] == "").sum()) if df[col].dtype == object else 0
        pct = 100 * n_missing / len(df)
        results[col] = {
            "n_missing": n_missing,
            "n_empty_string": n_empty_str,
            "pct_missing": round(pct, 2),
        }
    
    return results


def check_label_consistency(df: pd.DataFrame) -> dict:
    """Check for label consistency and flag potential issues."""
    results = {}
    
    if "object_level_label" not in df.columns:
        results["error"] = "No object_level_label column found"
        return results
    
    # Label value range
    results["label_values"] = sorted(df["object_level_label"].unique().tolist())
    results["label_distribution"] = df["object_level_label"].value_counts().to_dict()
    
    # Class imbalance ratio
    n_pos = int((df["object_level_label"] == 1).sum())
    n_neg = int((df["object_level_label"] == 0).sum())
    results["n_positive"] = n_pos
    results["n_negative"] = n_neg
    results["class_imbalance_ratio"] = round(n_neg / max(n_pos, 1), 2)
    
    # Label source breakdown
    if "label_source" in df.columns:
        results["label_source_counts"] = df["label_source"].value_counts().to_dict()
    
    # Uncertainty category breakdown
    if "uncertainty_category" in df.columns:
        results["uncertainty_category_counts"] = df["uncertainty_category"].value_counts().to_dict()
    
    # Label confidence breakdown
    if "label_confidence" in df.columns:
        results["label_confidence_counts"] = df["label_confidence"].value_counts().to_dict()
    
    # Training set positives vs catalog-match positives
    if "label_source" in df.columns:
        teach_pos = df[df["label_source"] == "train_set_teaching"]
        cat_pos = df[df["label_source"] == "catalog_match"]
        no_match = df[df["label_source"] == "no_match"]
        
        results["teaching_set_positives"] = len(teach_pos)
        results["catalog_matched_positives"] = len(cat_pos)
        results["unmatched_negatives"] = len(no_match)
        
        results["label_uncertainty_note"] = (
            "Teaching set positives (high confidence): labeled by project team as showing activity. "
            "Catalog-matched positives (medium confidence): object historically active, "
            "but this specific image may not visibly show activity — this is the label uncertainty problem. "
            "Unmatched negatives (low confidence): assumed inactive, may contain undiscovered active objects."
        )
    
    return results


def check_split_integrity(df: pd.DataFrame) -> dict:
    """Check train/val/test split for leakage and balance."""
    results = {}
    
    if "split" not in df.columns:
        results["error"] = "No split column found"
        return results
    
    # Split sizes
    split_counts = df["split"].value_counts().to_dict()
    results["split_sizes"] = split_counts
    results["total"] = len(df)
    
    # Label distribution per split
    for split_name in df["split"].unique():
        split_df = df[df["split"] == split_name]
        pos = int((split_df["object_level_label"] == 1).sum())
        neg = int((split_df["object_level_label"] == 0).sum())
        results[f"{split_name}_positive"] = pos
        results[f"{split_name}_negative"] = neg
        results[f"{split_name}_positive_rate"] = round(pos / max(len(split_df), 1), 3)
    
    # Check if training set subjects appear in val/test
    if "label_source" in df.columns:
        train_ids = set(df[df["split"] == "train"]["subject_id"].tolist())
        val_ids = set(df[df["split"] == "val"]["subject_id"].tolist())
        test_ids = set(df[df["split"] == "test"]["subject_id"].tolist())
        
        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids
        
        results["train_val_id_overlap"] = len(train_val_overlap)
        results["train_test_id_overlap"] = len(train_test_overlap)
        results["val_test_id_overlap"] = len(val_test_overlap)
        results["leakage_detected"] = (
            len(train_val_overlap) > 0 or
            len(train_test_overlap) > 0 or
            len(val_test_overlap) > 0
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Data quality checker for Active Asteroids metadata"
    )
    parser.add_argument("--metadata", type=str, default="data/metadata/subjects.csv")
    parser.add_argument("--output", type=str, default="results/data_quality_report.json")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    metadata_path = project_root / args.metadata
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("NASA Zooniverse Active Asteroids — Data Quality Checker")
    print(f"Input: {metadata_path}")
    print(f"Output: {output_path}")
    print()
    
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} subjects")
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_subjects": len(df),
        "columns": list(df.columns),
    }
    
    print("\n--- Duplicate Check ---")
    report["duplicates"] = check_duplicates(df)
    print(json.dumps(report["duplicates"], indent=2, default=str))
    
    print("\n--- Missing Values ---")
    report["missing_values"] = check_missing_values(df)
    # Print condensed version
    for col, info in report["missing_values"].items():
        if info["n_missing"] > 0 or info["n_empty_string"] > 0:
            print(f"  {col}: {info['n_missing']} missing, {info['n_empty_string']} empty strings")
    
    print("\n--- Label Consistency ---")
    report["labels"] = check_label_consistency(df)
    print(json.dumps(report["labels"], indent=2, default=str))
    
    print("\n--- Split Integrity ---")
    report["splits"] = check_split_integrity(df)
    print(json.dumps(report["splits"], indent=2, default=str))
    
    # Summary flags
    flags = []
    if report["duplicates"].get("duplicate_subject_ids", 0) > 0:
        flags.append("WARNING: Duplicate subject IDs detected")
    if report["splits"].get("leakage_detected", False):
        flags.append("CRITICAL: Data leakage detected between splits")
    if report["labels"].get("class_imbalance_ratio", 1) > 10:
        flags.append(f"WARNING: Severe class imbalance ({report['labels']['class_imbalance_ratio']}:1 neg:pos)")
    if report["labels"].get("catalog_matched_positives", 0) > 0:
        flags.append(
            "NOTE: Catalog-matched positives present — these carry label uncertainty "
            "(object-level label does not guarantee image-level visible activity)"
        )
    
    report["flags"] = flags
    print("\n--- Flags ---")
    for flag in flags:
        print(f"  {flag}")
    
    if not flags:
        print("  No critical issues detected.")
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nReport saved to {output_path}")
    return report


if __name__ == "__main__":
    main()
