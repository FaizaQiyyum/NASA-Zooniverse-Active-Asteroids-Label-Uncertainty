# NASA Zooniverse Active Asteroids
# Figure Generation Script (Pure Matplotlib, zero external plotting dependencies)
#
# Generates scientific figures:
#   1. dataset_distribution.png: Class and uncertainty distribution across splits
#   2. training_curves.png: ResNet18 training/validation loss and PR-AUC curves
#   3. roc_pr_curves.png: ROC and Precision-Recall curves (Logistic vs. ResNet18)
#   4. confusion_matrices.png: Normalized confusion matrices for baseline and CNN
#   5. uncertainty_distribution.png: Model prediction probability stratified by uncertainty group

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_dataset_distribution(metadata_df: pd.DataFrame, output_dir: Path):
    """Plot dataset breakdown by split and uncertainty category."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # Left: Splits by Label
    split_order = ["train", "val", "test"]
    neg_counts = [len(metadata_df[(metadata_df["split"]==s) & (metadata_df["object_level_label"]==0)]) for s in split_order]
    pos_counts = [len(metadata_df[(metadata_df["split"]==s) & (metadata_df["object_level_label"]==1)]) for s in split_order]
    
    x = np.arange(len(split_order))
    width = 0.5
    
    axes[0].bar(x, neg_counts, width, label="Inactive (0)", color="#4A90E2", edgecolor="#333333")
    axes[0].bar(x, pos_counts, width, bottom=neg_counts, label="Active (1)", color="#E94E77", edgecolor="#333333")
    
    axes[0].set_title("Dataset Composition by Split", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Split Partition", fontsize=11)
    axes[0].set_ylabel("Number of Cutouts", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Train\n(n={neg_counts[0]+pos_counts[0]})", 
                             f"Val\n(n={neg_counts[1]+pos_counts[1]})", 
                             f"Test\n(n={neg_counts[2]+pos_counts[2]})"])
    axes[0].legend(frameon=True)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    
    # Annotate bars
    for i in range(len(split_order)):
        if neg_counts[i] > 0:
            axes[0].text(i, neg_counts[i] / 2, str(neg_counts[i]), ha="center", va="center", color="white", fontweight="bold")
        if pos_counts[i] > 0:
            axes[0].text(i, neg_counts[i] + pos_counts[i] / 2, str(pos_counts[i]), ha="center", va="center", color="white", fontweight="bold")
            
    # Right: Uncertainty Category Breakdown
    unc_counts = metadata_df["uncertainty_category"].value_counts()
    colors = ["#E94E77", "#4A90E2", "#F5A623"]
    labels = [
        f"Teaching Positive\n(High Conf, n={unc_counts.get('teaching_positive', 0)})",
        f"Unmatched Negative\n(Low Conf, n={unc_counts.get('unmatched_negative', 0)})",
        f"Catalog Matched\n(Medium Conf, n={unc_counts.get('catalog_matched_positive', 0)})"
    ]
    axes[1].pie(
        [unc_counts.get('teaching_positive', 0), unc_counts.get('unmatched_negative', 0), unc_counts.get('catalog_matched_positive', 0)],
        labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, explode=(0.04, 0.04, 0.08),
        wedgeprops={"edgecolor": "#333333", "linewidth": 0.8}
    )
    axes[1].set_title("Label Uncertainty Categories", fontsize=13, fontweight="bold", pad=12)
    
    plt.tight_layout()
    fig_path = output_dir / "dataset_distribution.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def plot_training_curves(training_results: dict, output_dir: Path):
    """Plot ResNet18 training loss and validation PR-AUC curves."""
    if "resnet18" not in training_results or "history" not in training_results["resnet18"]:
        return
    
    hist = training_results["resnet18"]["history"]
    epochs = hist.get("epoch", list(range(1, len(hist["train_loss"]) + 1)))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    
    # Loss curves
    axes[0].plot(epochs, hist["train_loss"], "o-", color="#D0021B", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, hist["val_loss"], "s--", color="#4A90E2", label="Val Loss", linewidth=2)
    axes[0].set_title("ResNet18 Loss Convergence", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("BCE Loss", fontsize=10)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Validation PR-AUC curve
    axes[1].plot(epochs, hist["val_pr_auc"], "d-", color="#2ECC71", label="Val PR-AUC", linewidth=2.2)
    axes[1].plot(epochs, hist["val_roc_auc"], "^:", color="#9B59B6", label="Val ROC-AUC", linewidth=1.8)
    best_ep = training_results["resnet18"].get("best_epoch", 2)
    axes[1].axvline(best_ep, color="black", linestyle=":", label=f"Best Checkpoint (Ep {best_ep})")
    axes[1].set_title("Validation Discrimination Metrics", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Metric Score", fontsize=10)
    axes[1].set_ylim([0.0, 1.05])
    axes[1].legend(frameon=True, loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    fig_path = output_dir / "training_curves.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def plot_roc_and_pr_curves(preds_df: pd.DataFrame, output_dir: Path):
    """Plot ROC and Precision-Recall curves on held-out test split."""
    test_preds = preds_df[preds_df["split"] == "test"].copy()
    if len(test_preds) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    colors = {"logistic_regression": "#F5A623", "resnet18": "#4A90E2"}
    names = {"logistic_regression": "Logistic Regression (Baseline)", "resnet18": "ResNet18 (Fine-Tuned)"}
    
    # ROC Curves
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Chance (AUC=0.50)")
    for model_name in test_preds["model"].unique():
        m_df = test_preds[test_preds["model"] == model_name]
        fpr, tpr, _ = roc_curve(m_df["true_label"], m_df["pred_prob"])
        auc_val = roc_auc_score(m_df["true_label"], m_df["pred_prob"])
        c = colors.get(model_name, "#333333")
        label_str = f"{names.get(model_name, model_name)} (AUC = {auc_val:.3f})"
        axes[0].plot(fpr, tpr, color=c, lw=2.2, label=label_str)
    
    axes[0].set_title("Test Split ROC Curves", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=10)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=10)
    axes[0].legend(frameon=True, loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Precision-Recall Curves
    baseline_pr = (test_preds["true_label"] == 1).mean()
    axes[1].axhline(baseline_pr, color="black", linestyle="--", alpha=0.4, label=f"Prevalence Baseline ({baseline_pr:.2f})")
    for model_name in test_preds["model"].unique():
        m_df = test_preds[test_preds["model"] == model_name]
        prec, rec, _ = precision_recall_curve(m_df["true_label"], m_df["pred_prob"])
        ap_val = average_precision_score(m_df["true_label"], m_df["pred_prob"])
        c = colors.get(model_name, "#333333")
        label_str = f"{names.get(model_name, model_name)} (AP = {ap_val:.3f})"
        axes[1].plot(rec, prec, color=c, lw=2.2, label=label_str)
    
    axes[1].set_title("Test Split Precision-Recall Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].legend(frameon=True, loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    fig_path = output_dir / "roc_pr_curves.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def plot_confusion_matrices(preds_df: pd.DataFrame, output_dir: Path):
    """Plot normalized confusion matrices using pure matplotlib."""
    test_preds = preds_df[preds_df["split"] == "test"].copy()
    models = test_preds["model"].unique()
    if len(models) == 0:
        return
    
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.5), dpi=300)
    if len(models) == 1:
        axes = [axes]
        
    names = {"logistic_regression": "Logistic Regression", "resnet18": "ResNet18 Fine-Tuned"}
    
    for ax, model_name in zip(axes, models):
        m_df = test_preds[test_preds["model"] == model_name]
        cm = confusion_matrix(m_df["true_label"], m_df["pred_label"])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        cax = ax.matshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{names.get(model_name, model_name)}\n(Test Normalized CM)", fontsize=11, fontweight="bold", pad=15)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Inactive", "Active"], fontsize=10)
        ax.set_yticklabels(["Inactive", "Active"], fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Object-Level Label", fontsize=10)
        
        for i in range(2):
            for j in range(2):
                val = cm_norm[i, j]
                cnt = cm[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.1%}\n(n={cnt})", ha="center", va="center", color=color, fontweight="bold")
                
    plt.tight_layout()
    fig_path = output_dir / "confusion_matrices.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def plot_uncertainty_distribution(preds_df: pd.DataFrame, output_dir: Path):
    """Plot predicted probability distributions by uncertainty category using pure matplotlib."""
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    
    plot_df = preds_df[preds_df["model"] == "resnet18"] if "resnet18" in preds_df["model"].values else preds_df
    
    cats = ["catalog_matched_positive", "unmatched_negative"]
    labels = ["Catalog Matched Active\n(Medium Conf, n=32)", "Unmatched Inactive\n(Low Conf, n=68)"]
    colors = ["#F5A623", "#4A90E2"]
    
    data = [plot_df[plot_df["uncertainty_category"] == c]["pred_prob"].values for c in cats]
    
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.45)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
        patch.set_edgecolor("#333333")
        
    for median in bp['medians']:
        median.set(color="black", linewidth=1.5)
        
    # Overlay scatter points with jitter
    np.random.seed(42)
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.plot(x, d, 'o', color="#222222", alpha=0.6, markersize=5)
        
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Default Decision Cutoff (0.5)")
    ax.set_title("ResNet18 Predicted Probabilities by Label Uncertainty Group", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Predicted Probability (Active)", fontsize=11)
    ax.set_ylim([-0.05, 1.05])
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig_path = output_dir / "uncertainty_distribution.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate scientific figures for research report")
    parser.add_argument("--metadata", type=str, default="data/metadata/subjects.csv")
    parser.add_argument("--predictions", type=str, default="results/all_predictions.csv")
    parser.add_argument("--training_results", type=str, default="results/training_results.json")
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent.parent
    metadata_path = root / args.metadata
    preds_path = root / args.predictions
    results_path = root / args.training_results
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating Active Asteroids Research Figures...")
    
    if metadata_path.exists():
        meta_df = pd.read_csv(metadata_path)
        plot_dataset_distribution(meta_df, out_dir)
    
    if results_path.exists():
        with open(results_path, "r") as f:
            t_res = json.load(f)
        plot_training_curves(t_res, out_dir)
    
    if preds_path.exists():
        preds_df = pd.read_csv(preds_path)
        plot_roc_and_pr_curves(preds_df, out_dir)
        plot_confusion_matrices(preds_df, out_dir)
        plot_uncertainty_distribution(preds_df, out_dir)
        
    print("All figures successfully generated in figures/ directory.")


if __name__ == "__main__":
    main()
