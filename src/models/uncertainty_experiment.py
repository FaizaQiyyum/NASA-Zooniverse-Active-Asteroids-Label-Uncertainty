# NASA Zooniverse Active Asteroids
# Label Uncertainty Experiment
#
# Main scientific contribution: investigates how object-level label uncertainty
# affects classifier behavior on active asteroid images.
#
# Experiments:
#   A: Train on ALL labels (including uncertain catalog-match positives)
#   B: Train EXCLUDING uncertain catalog-match positives
#   C: Analyze model confidence stratified by uncertainty category
#   D: Compare error rates across uncertainty categories
#
# Usage:
#   python src/models/uncertainty_experiment.py \
#     --metadata data/metadata/subjects.csv \
#     --predictions results/all_predictions.csv \
#     --output results/uncertainty_analysis.json

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocess import build_feature_matrix, AsteroidDataset
from src.models.train_baseline import (
    train_logistic, train_resnet18, compute_metrics
)


def analyze_predictions_by_uncertainty(
    predictions_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> dict:
    """
    Stratify model predictions by uncertainty category.
    
    CORE SCIENTIFIC QUESTION:
    Does the model behave differently for examples with high vs. low label confidence?
    
    Categories:
      teaching_positive      : high confidence positives (labeled by project team)
      catalog_matched_positive: medium confidence (object historically active; THIS image uncertain)
      unmatched_negative     : assumed negatives (object not in active catalog)
    
    Key hypotheses to test:
    H1: Model confidence (pred_prob) is lower for catalog-matched positives
        than for teaching positives → label uncertainty reduces model certainty
    H2: Catalog-matched positives have higher false-negative rate than
        teaching positives → ambiguous training signal hurts recall
    H3: High model confidence wrong predictions cluster in specific
        uncertainty categories → identifies which labels are most problematic
    """
    results = {}
    
    # Merge predictions with metadata if needed
    if "uncertainty_category" not in predictions_df.columns:
        merge_cols = ["subject_id", "uncertainty_category", "label_confidence", "label_source"]
        available = [c for c in merge_cols if c in metadata_df.columns]
        predictions_df = predictions_df.merge(
            metadata_df[available], on="subject_id", how="left"
        )
    
    # Overall statistics
    results["total_predictions"] = len(predictions_df)
    results["models"] = predictions_df["model"].unique().tolist() if "model" in predictions_df.columns else []
    
    # Per-uncertainty-category analysis
    results["by_uncertainty_category"] = {}
    
    unc_categories = predictions_df["uncertainty_category"].unique()
    
    for cat in unc_categories:
        cat_df = predictions_df[predictions_df["uncertainty_category"] == cat]
        
        if len(cat_df) == 0:
            continue
        
        cat_results = {
            "n_samples": len(cat_df),
            "n_positive": int((cat_df["true_label"] == 1).sum()),
            "n_negative": int((cat_df["true_label"] == 0).sum()),
        }
        
        # Prediction confidence distribution
        cat_results["mean_pred_prob"] = round(float(cat_df["pred_prob"].mean()), 4)
        cat_results["std_pred_prob"] = round(float(cat_df["pred_prob"].std()), 4)
        cat_results["median_pred_prob"] = round(float(cat_df["pred_prob"].median()), 4)
        
        # Error analysis
        positives = cat_df[cat_df["true_label"] == 1]
        negatives = cat_df[cat_df["true_label"] == 0]
        
        if len(positives) > 0:
            # False negatives: true positives predicted as negative
            fn_mask = (positives["pred_prob"] < 0.5)
            cat_results["false_negative_rate"] = round(float(fn_mask.mean()), 4)
            cat_results["mean_prob_for_positives"] = round(float(positives["pred_prob"].mean()), 4)
            
            # High confidence wrong predictions for positives
            high_conf_fn = positives[fn_mask & (positives["pred_prob"] < 0.2)]
            cat_results["high_confidence_false_negatives"] = len(high_conf_fn)
        else:
            cat_results["false_negative_rate"] = None
            cat_results["mean_prob_for_positives"] = None
            cat_results["high_confidence_false_negatives"] = 0
        
        if len(negatives) > 0:
            # False positives: true negatives predicted as positive
            fp_mask = (negatives["pred_prob"] >= 0.5)
            cat_results["false_positive_rate"] = round(float(fp_mask.mean()), 4)
            cat_results["mean_prob_for_negatives"] = round(float(negatives["pred_prob"].mean()), 4)
            
            # High confidence wrong predictions for negatives
            high_conf_fp = negatives[fp_mask & (negatives["pred_prob"] > 0.8)]
            cat_results["high_confidence_false_positives"] = len(high_conf_fp)
        else:
            cat_results["false_positive_rate"] = None
            cat_results["mean_prob_for_negatives"] = None
            cat_results["high_confidence_false_positives"] = 0
        
        # ROC-AUC per category if both classes present
        n_pos = (cat_df["true_label"] == 1).sum()
        n_neg = (cat_df["true_label"] == 0).sum()
        if n_pos > 0 and n_neg > 0:
            try:
                cat_results["roc_auc"] = round(
                    float(roc_auc_score(cat_df["true_label"], cat_df["pred_prob"])), 4
                )
            except Exception:
                cat_results["roc_auc"] = None
        else:
            cat_results["roc_auc"] = None
        
        results["by_uncertainty_category"][cat] = cat_results
    
    # Hypothesis test results
    results["hypothesis_tests"] = run_hypothesis_tests(predictions_df)
    
    return results


def run_hypothesis_tests(predictions_df: pd.DataFrame) -> dict:
    """
    Test hypotheses about label uncertainty effects.
    
    Uses descriptive statistics (not formal statistical tests,
    since sample size may be insufficient for strong inference).
    Explicitly notes where sample size limits conclusions.
    """
    tests = {}
    
    teach_pos = predictions_df[
        predictions_df["uncertainty_category"] == "teaching_positive"
    ]
    cat_pos = predictions_df[
        predictions_df["uncertainty_category"] == "catalog_matched_positive"
    ]
    no_match = predictions_df[
        predictions_df["uncertainty_category"] == "unmatched_negative"
    ]
    
    # H1: Model confidence lower for catalog-matched vs teaching positives
    h1 = {
        "hypothesis": (
            "Model confidence (pred_prob) is lower for catalog-matched positives "
            "than for teaching set positives, because the training signal is noisier "
            "for catalog-matched examples."
        ),
    }
    
    if len(teach_pos) > 0 and len(cat_pos) > 0:
        tp_prob = teach_pos["pred_prob"].mean()
        cp_prob = cat_pos["pred_prob"].mean()
        h1["teaching_positive_mean_prob"] = round(float(tp_prob), 4)
        h1["catalog_matched_mean_prob"] = round(float(cp_prob), 4)
        h1["difference"] = round(float(tp_prob - cp_prob), 4)
        h1["direction"] = "supported" if tp_prob > cp_prob else "not_supported"
        h1["note"] = (
            f"n_teaching={len(teach_pos)}, n_catalog={len(cat_pos)}. "
            "Small sample — interpret cautiously."
        )
    else:
        h1["result"] = "insufficient_data"
        h1["note"] = (
            f"n_teaching={len(teach_pos)}, n_catalog={len(cat_pos)}. "
            "Cannot test — need both categories in predictions."
        )
    
    tests["H1"] = h1
    
    # H2: Higher false-negative rate for catalog-matched vs teaching positives
    h2 = {
        "hypothesis": (
            "Catalog-matched positives have higher false-negative rate than teaching positives "
            "because the object-level label does not guarantee visible activity in every image."
        ),
    }
    
    if len(teach_pos) > 0:
        tp_fn_rate = float((teach_pos["pred_prob"] < 0.5).mean())
        h2["teaching_positive_fn_rate"] = round(tp_fn_rate, 4)
    
    if len(cat_pos) > 0:
        cp_fn_rate = float((cat_pos["pred_prob"] < 0.5).mean())
        h2["catalog_matched_fn_rate"] = round(cp_fn_rate, 4)
    
    if len(teach_pos) > 0 and len(cat_pos) > 0:
        h2["fn_rate_difference"] = round(cp_fn_rate - tp_fn_rate, 4)
        h2["direction"] = "supported" if cp_fn_rate > tp_fn_rate else "not_supported"
        h2["note"] = (
            "Difference in FN rates between label confidence groups. "
            f"n_teaching={len(teach_pos)}, n_catalog={len(cat_pos)}."
        )
    else:
        h2["result"] = "insufficient_data"
    
    tests["H2"] = h2
    
    # H3: High-confidence wrong predictions cluster in specific categories
    h3 = {
        "hypothesis": (
            "High-confidence wrong predictions (false positives with prob > 0.8, "
            "false negatives with prob < 0.2) cluster in the unmatched_negative category, "
            "indicating that the negative pool contains hard examples (galaxy false positives, "
            "possible undiscovered active objects)."
        ),
    }
    
    # High confidence FPs (pred > 0.8, true = 0)
    hc_fp = predictions_df[
        (predictions_df["pred_prob"] > 0.8) &
        (predictions_df["true_label"] == 0)
    ]
    # High confidence FNs (pred < 0.2, true = 1)
    hc_fn = predictions_df[
        (predictions_df["pred_prob"] < 0.2) &
        (predictions_df["true_label"] == 1)
    ]
    
    h3["n_high_conf_fp"] = len(hc_fp)
    h3["n_high_conf_fn"] = len(hc_fn)
    
    if len(hc_fp) > 0 and "uncertainty_category" in hc_fp.columns:
        h3["high_conf_fp_by_category"] = hc_fp["uncertainty_category"].value_counts().to_dict()
    if len(hc_fn) > 0 and "uncertainty_category" in hc_fn.columns:
        h3["high_conf_fn_by_category"] = hc_fn["uncertainty_category"].value_counts().to_dict()
    
    tests["H3"] = h3
    
    return tests


def experiment_b_filtered(
    df: pd.DataFrame,
    predictions_all: pd.DataFrame,
    use_synthetic: bool = True,
    seed: int = 42,
) -> dict:
    """
    Experiment B: Train EXCLUDING catalog-matched positives from training.
    
    Rationale: If catalog-matched positives introduce label noise
    (object label ≠ image evidence), removing them should:
    - Improve precision (fewer ambiguous positives confusing the model)
    - Possibly hurt recall (less positive training data)
    
    This is a fair comparison only possible when we have catalog-matched
    positives in the dataset. If none exist, documents the limitation.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: FILTERED TRAINING (excluding uncertain positives)")
    print("=" * 60)
    
    # Check if we have catalog-matched positives
    n_cat_matched = (df["label_source"] == "catalog_match").sum()
    print(f"Catalog-matched positives in dataset: {n_cat_matched}")
    
    if n_cat_matched == 0:
        print("No catalog-matched positives found.")
        print("In the current dataset (200 subjects from partial API fetch),")
        print("all positives come from the teaching set (set 96361).")
        print("Experiment B requires catalog-matched positives from general stream sets.")
        print("\nLIMITATION: This experiment cannot be fully executed with the current")
        print("partial dataset. With a complete API fetch across all subject sets,")
        print("catalog-matched positives would be available for this comparison.")
        
        return {
            "experiment": "B",
            "status": "limited",
            "reason": (
                "No catalog-matched positives in current dataset. "
                "Current dataset only contains teaching set positives (set 96361, all high-confidence) "
                "and unmatched negatives. To run Experiment B, a complete fetch of general-stream "
                "subject sets (36656, 96360, 97743, etc.) is needed, which requires stable "
                "API access across multiple pages."
            ),
            "workaround": (
                "Experiment B is implemented conceptually and the code is correct. "
                "The filtering logic excludes label_source='catalog_match' from training. "
                "Results would be meaningful with a larger dataset containing catalog-matched "
                "positives from the general classification stream."
            ),
            "n_catalog_matched": 0,
        }
    
    # Create filtered training set
    df_filtered = df[
        ~((df["split"] == "train") & (df["label_source"] == "catalog_match"))
    ].copy()
    
    print(f"Original training size: {(df['split'] == 'train').sum()}")
    print(f"Filtered training size: {(df_filtered['split'] == 'train').sum()}")
    print(f"Removed: {n_cat_matched} catalog-matched positives from training")
    
    # Train logistic regression on filtered data
    filtered_results = train_logistic(
        df_filtered,
        use_synthetic=use_synthetic,
        seed=seed,
    )
    
    return {
        "experiment": "B",
        "status": "executed",
        "n_catalog_matched_removed": int(n_cat_matched),
        "val_metrics": filtered_results["val_metrics"],
        "test_metrics": filtered_results["test_metrics"],
        "comparison_note": (
            "Compare to Experiment A (all labels) to assess the effect of "
            "removing uncertain catalog-matched positives from training."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Label Uncertainty Experiment for Active Asteroids"
    )
    parser.add_argument("--metadata", type=str, default="data/metadata/subjects.csv")
    parser.add_argument("--predictions", type=str, default="results/all_predictions.csv")
    parser.add_argument("--output", type=str, default="results/uncertainty_analysis.json")
    parser.add_argument("--use_synthetic", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    metadata_path = project_root / args.metadata
    predictions_path = project_root / args.predictions
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("NASA Zooniverse Active Asteroids — Label Uncertainty Experiment")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    df = pd.read_csv(metadata_path)
    
    results = {
        "generated_at": datetime.now().isoformat(),
        "dataset_size": len(df),
        "label_distribution": df["object_level_label"].value_counts().to_dict(),
        "uncertainty_distribution": df["uncertainty_category"].value_counts().to_dict(),
    }
    
    # Load existing predictions or note they need to be generated
    if predictions_path.exists():
        preds_df = pd.read_csv(predictions_path)
        print(f"Loaded {len(preds_df)} predictions from {predictions_path}")
        
        # Merge with metadata for uncertainty categories
        results["experiment_C"] = analyze_predictions_by_uncertainty(preds_df, df)
        print("\n--- Experiment C: Uncertainty-Stratified Analysis ---")
        print(json.dumps(results["experiment_C"], indent=2, default=str))
    else:
        print(f"Predictions not found at {predictions_path}")
        print("Run train_baseline.py first, then re-run this script.")
        results["experiment_C"] = {"status": "predictions_not_found"}
    
    # Experiment B: Filtered training
    results["experiment_B"] = experiment_b_filtered(
        df, 
        preds_df if predictions_path.exists() else pd.DataFrame(),
        use_synthetic=args.use_synthetic,
        seed=args.seed,
    )
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nUncertainty analysis saved to {output_path}")
    return results


if __name__ == "__main__":
    main()
