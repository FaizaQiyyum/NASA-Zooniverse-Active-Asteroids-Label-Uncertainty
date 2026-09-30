# NASA Zooniverse Active Asteroids
# Baseline Model: Logistic Regression + ResNet18 Fine-Tuning
#
# Implements two models:
#   1. Logistic regression on hand-crafted features (simple baseline)
#   2. ResNet18 with ImageNet pre-training, fine-tuned for binary classification
#
# Usage:
#   python src/models/train_baseline.py --model logistic --seed 42
#   python src/models/train_baseline.py --model resnet18 --seed 42

import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocess import AsteroidDataset, build_feature_matrix


# ─── Logistic Regression Baseline ─────────────────────────────────────────────

def train_logistic(
    df: pd.DataFrame,
    image_dir: str = None,
    use_synthetic: bool = True,
    seed: int = 42,
    C: float = 1.0,
) -> dict:
    """
    Train logistic regression on 8 hand-crafted image features.
    
    This is the simple baseline — near-random performance is expected
    because hand-crafted flux features cannot capture tail morphology.
    Documented honestly per ZamirDev's finding (ROC-AUC ~0.53).
    
    Returns dict with model, scaler, and metadata.
    """
    print("=" * 60)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 60)
    print(f"Seed: {seed}, C: {C}")
    print(f"Synthetic images: {use_synthetic}")
    
    t0 = time.time()
    
    # Build feature matrices for each split
    print("\nBuilding feature matrices...")
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    
    X_train, y_train, ids_train, unc_train = build_feature_matrix(
        train_df, image_dir=image_dir, use_synthetic=use_synthetic
    )
    X_val, y_val, ids_val, unc_val = build_feature_matrix(
        val_df, image_dir=image_dir, use_synthetic=use_synthetic
    )
    X_test, y_test, ids_test, unc_test = build_feature_matrix(
        test_df, image_dir=image_dir, use_synthetic=use_synthetic
    )
    
    print(f"Train: {X_train.shape}, positives: {y_train.sum()}")
    print(f"Val:   {X_val.shape}, positives: {y_val.sum()}")
    print(f"Test:  {X_test.shape}, positives: {y_test.sum()}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Train model with class weighting for imbalance
    np.random.seed(seed)
    model = LogisticRegression(
        max_iter=3000,
        C=C,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(X_train_s, y_train)
    
    training_time = time.time() - t0
    
    # Evaluate
    def evaluate_split(X_s, y_true, ids, unc_cats, split_name):
        y_prob = model.predict_proba(X_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        metrics = compute_metrics(y_true, y_pred, y_prob, split_name)
        
        # Predictions dataframe
        preds_df = pd.DataFrame({
            "subject_id": ids,
            "true_label": y_true,
            "pred_label": y_pred,
            "pred_prob": y_prob,
            "uncertainty_category": unc_cats,
            "split": split_name,
            "model": "logistic_regression",
        })
        
        return metrics, preds_df
    
    val_metrics, val_preds = evaluate_split(X_val_s, y_val, ids_val, unc_val, "val")
    test_metrics, test_preds = evaluate_split(X_test_s, y_test, ids_test, unc_test, "test")
    
    print(f"\nVal  ROC-AUC: {val_metrics['roc_auc']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Training time: {training_time:.1f}s")
    
    return {
        "model": model,
        "scaler": scaler,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_predictions": val_preds,
        "test_predictions": test_preds,
        "training_time": training_time,
        "config": {
            "model_type": "logistic_regression",
            "C": C,
            "seed": seed,
            "use_synthetic": use_synthetic,
            "n_features": 8,
            "class_weight": "balanced",
        },
    }


# ─── ResNet18 CNN ──────────────────────────────────────────────────────────────

class ResNet18Classifier(nn.Module):
    """
    ResNet18 with final FC replaced for binary classification.
    Pre-trained on ImageNet; fine-tuned on asteroid cutouts.
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        
        # Replace final FC layer
        in_features = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )
        self.backbone = backbone
    
    def forward(self, x):
        return self.backbone(x).squeeze(1)  # (B,)
    
    def get_features(self, x):
        """Extract penultimate layer features for uncertainty analysis."""
        # Hook into the layer before final FC
        features = []
        def hook(module, input, output):
            features.append(output.squeeze())
        
        handle = self.backbone.avgpool.register_forward_hook(hook)
        _ = self.forward(x)
        handle.remove()
        
        if features:
            return features[0].detach()
        return None


def train_resnet18(
    df: pd.DataFrame,
    image_dir: str = None,
    use_synthetic: bool = True,
    seed: int = 42,
    n_epochs: int = 15,
    lr: float = 3e-4,
    batch_size: int = 32,
    patience: int = 5,
    output_dir: str = None,
) -> dict:
    """
    Train ResNet18 on asteroid cutouts.
    
    Training strategy:
    - Pretrained ImageNet weights
    - WeightedRandomSampler for class balance
    - BCEWithLogitsLoss with pos_weight
    - CosineAnnealingLR scheduler
    - Early stopping on val PR-AUC
    
    Returns dict with model, metrics, predictions.
    """
    print("=" * 60)
    print("RESNET18 FINE-TUNING")
    print("=" * 60)
    print(f"Seed: {seed}, Epochs: {n_epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"Synthetic images: {use_synthetic}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    t0 = time.time()
    
    # Build datasets
    train_dataset = AsteroidDataset(
        df, image_dir=image_dir, split="train",
        use_synthetic=use_synthetic
    )
    val_dataset = AsteroidDataset(
        df, image_dir=image_dir, split="val",
        use_synthetic=use_synthetic
    )
    test_dataset = AsteroidDataset(
        df, image_dir=image_dir, split="test",
        use_synthetic=use_synthetic
    )
    
    # Weighted sampler for training (addresses class imbalance)
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0
    )
    
    # Model
    model = ResNet18Classifier(pretrained=True, dropout=0.3)
    model = model.to(device)
    
    # Class-weighted loss
    n_pos = (df[df["split"] == "train"]["object_level_label"] == 1).sum()
    n_neg = (df[df["split"] == "train"]["object_level_label"] == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"pos_weight: {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    
    # Training loop
    history = {
        "train_loss": [], "val_loss": [],
        "val_roc_auc": [], "val_pr_auc": [],
        "epoch": [],
    }
    
    best_val_pr_auc = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, n_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # --- Validate ---
        val_metrics, val_preds_df = eval_loader(model, val_loader, criterion, device, "val")
        
        avg_train_loss = np.mean(train_losses)
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["val_pr_auc"].append(val_metrics["pr_auc"])
        
        print(
            f"Epoch {epoch:02d}/{n_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val PR-AUC: {val_metrics['pr_auc']:.4f}"
        )
        
        # Early stopping on val PR-AUC
        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["pr_auc"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best checkpoint from epoch {best_epoch} (val PR-AUC: {best_val_pr_auc:.4f})")
    
    training_time = time.time() - t0
    
    # Final evaluation
    val_metrics_final, val_preds_final = eval_loader(model, val_loader, criterion, device, "val")
    test_metrics_final, test_preds_final = eval_loader(model, test_loader, criterion, device, "test")
    
    # Add model identifier
    for preds_df in [val_preds_final, test_preds_final]:
        preds_df["model"] = "resnet18"
    
    print(f"\nFinal Val  ROC-AUC: {val_metrics_final['roc_auc']:.4f} | PR-AUC: {val_metrics_final['pr_auc']:.4f}")
    print(f"Final Test ROC-AUC: {test_metrics_final['roc_auc']:.4f} | PR-AUC: {test_metrics_final['pr_auc']:.4f}")
    print(f"Total training time: {training_time:.1f}s")
    
    # Save model
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"resnet18_seed{seed}.pt"
        torch.save({
            "model_state_dict": best_state,
            "epoch": best_epoch,
            "val_pr_auc": best_val_pr_auc,
            "config": {
                "seed": seed, "n_epochs": n_epochs,
                "lr": lr, "batch_size": batch_size,
            },
        }, model_path)
        print(f"Model saved to {model_path}")
    
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics_final,
        "test_metrics": test_metrics_final,
        "val_predictions": val_preds_final,
        "test_predictions": test_preds_final,
        "training_time": training_time,
        "config": {
            "model_type": "resnet18",
            "seed": seed,
            "n_epochs": n_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "patience": patience,
            "use_synthetic": use_synthetic,
            "pretrained": True,
            "dropout": 0.3,
        },
    }


# ─── Evaluation Helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
) -> tuple:
    """Evaluate model on a DataLoader. Returns (metrics_dict, predictions_df)."""
    model.eval()
    all_labels, all_probs, all_ids, all_unc = [], [], [], []
    total_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1
        
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_ids.extend(batch["subject_id"])
        all_unc.extend(batch["uncertainty_category"])
    
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    
    avg_loss = total_loss / max(n_batches, 1)
    
    metrics = compute_metrics(y_true, y_pred, y_prob, split_name)
    metrics["loss"] = round(avg_loss, 6)
    
    preds_df = pd.DataFrame({
        "subject_id": all_ids,
        "true_label": y_true,
        "pred_label": y_pred,
        "pred_prob": y_prob,
        "uncertainty_category": all_unc,
        "split": split_name,
    })
    
    return metrics, preds_df


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split_name: str,
) -> dict:
    """Compute all evaluation metrics."""
    n_pos = y_true.sum()
    
    metrics = {"split": split_name}
    
    if n_pos == 0 or n_pos == len(y_true):
        # Degenerate case
        metrics.update({
            "roc_auc": 0.5, "pr_auc": float(n_pos / len(y_true)),
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "accuracy": float((y_pred == y_true).mean()),
            "n_samples": len(y_true), "n_positive": int(n_pos),
            "note": "degenerate split",
        })
        return metrics
    
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(len(y_true), 1)
    
    # Top-10% recall (what fraction of positives are in top 10% by score)
    threshold_top10 = np.percentile(y_prob, 90)
    top10_pred = (y_prob >= threshold_top10).astype(int)
    top10_recall = (top10_pred * y_true).sum() / max(y_true.sum(), 1)
    
    metrics.update({
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "precision_at_0_5": round(float(precision), 4),
        "recall_at_0_5": round(float(recall), 4),
        "f1_at_0_5": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
        "recall_at_top10pct": round(float(top10_recall), 4),
        "n_samples": int(len(y_true)),
        "n_positive": int(n_pos),
        "n_negative": int(len(y_true) - n_pos),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    })
    
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Active Asteroids classifier")
    parser.add_argument("--model", choices=["logistic", "resnet18", "both"], default="both")
    parser.add_argument("--metadata", type=str, default="data/metadata/subjects.csv")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory with downloaded images (optional)")
    parser.add_argument("--use_synthetic", action="store_true", default=True,
                        help="Use synthetic images if real images unavailable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    metadata_path = project_root / args.metadata
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nNASA Zooniverse Active Asteroids — Model Training")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Metadata: {metadata_path}")
    print(f"Output dir: {output_dir}")
    print(f"Model(s): {args.model}")
    print(f"Seed: {args.seed}")
    print()
    
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} subjects")
    
    image_dir = (project_root / args.image_dir) if args.image_dir else None
    
    all_results = {}
    all_predictions = []
    
    if args.model in ("logistic", "both"):
        lr_results = train_logistic(
            df, image_dir=image_dir,
            use_synthetic=args.use_synthetic,
            seed=args.seed,
        )
        all_results["logistic"] = {
            "val_metrics": lr_results["val_metrics"],
            "test_metrics": lr_results["test_metrics"],
            "config": lr_results["config"],
            "training_time": lr_results["training_time"],
        }
        
        # Save predictions
        lr_preds = pd.concat([
            lr_results["val_predictions"],
            lr_results["test_predictions"]
        ], ignore_index=True)
        all_predictions.append(lr_preds)
        
        # Save logistic model
        joblib.dump(lr_results["model"], output_dir / "logistic_model.pkl")
        joblib.dump(lr_results["scaler"], output_dir / "logistic_scaler.pkl")
        print(f"\nLogistic model saved to {output_dir}")
    
    if args.model in ("resnet18", "both"):
        cnn_results = train_resnet18(
            df, image_dir=image_dir,
            use_synthetic=args.use_synthetic,
            seed=args.seed,
            n_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            output_dir=str(output_dir),
        )
        all_results["resnet18"] = {
            "val_metrics": cnn_results["val_metrics"],
            "test_metrics": cnn_results["test_metrics"],
            "config": cnn_results["config"],
            "training_time": cnn_results["training_time"],
            "best_epoch": cnn_results["best_epoch"],
            "history": cnn_results["history"],
        }
        
        cnn_preds = pd.concat([
            cnn_results["val_predictions"],
            cnn_results["test_predictions"]
        ], ignore_index=True)
        all_predictions.append(cnn_preds)
    
    # Save results
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    
    if all_predictions:
        preds_path = output_dir / "all_predictions.csv"
        pd.concat(all_predictions, ignore_index=True).to_csv(preds_path, index=False)
        print(f"Predictions saved to {preds_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Split':<8} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1@0.5':<10}")
    print("-" * 70)
    for model_name, res in all_results.items():
        for split_key in ["val_metrics", "test_metrics"]:
            m = res[split_key]
            split = m.get("split", split_key.replace("_metrics", ""))
            print(
                f"{model_name:<20} {split:<8} "
                f"{m.get('roc_auc', 0):<10.4f} "
                f"{m.get('pr_auc', 0):<10.4f} "
                f"{m.get('f1_at_0_5', 0):<10.4f}"
            )
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    main()
