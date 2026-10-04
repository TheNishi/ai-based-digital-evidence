"""Master End-to-End Retraining, Validation, and Evaluation Pipeline for CIFAKE Deepfake Detection.

Follows strict ML research standards:
1. Strict 3-way disjoint split: Train (4,000) / Validation (1,000) / Held-out Test (2,000).
2. Domain adaptation & classifier head training on CIFAKE features for:
   - EfficientNet-B3
   - EfficientFormerV2-S1
   - FasterViT-2-224
3. Operating Threshold (Youden's J) and Ensemble Weights tuned ON VALIDATION SET ONLY.
4. Frozen parameters evaluated on UNTOUCHED held-out test set.
5. Continuous probability ROC-AUC / PR-AUC calculations (Positive class = FAKE = 1).
6. Generates full diagnostic plots and master comparison tables.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from orchestration.forensics import load_forensics_models
from orchestration.orchestrator import load_config


# ---------------------------------------------------------------------------
# Styling & Helpers
# ---------------------------------------------------------------------------

def set_plot_style() -> None:
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "font.family": "sans-serif",
        "figure.facecolor": "#0F172A",
        "axes.facecolor": "#1E293B",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#F8FAFC",
        "text.color": "#F8FAFC",
        "xtick.color": "#94A3B8",
        "ytick.color": "#94A3B8",
        "grid.color": "#334155",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "legend.facecolor": "#1E293B",
        "legend.edgecolor": "#475569",
        "legend.fontsize": 11,
    })


def compute_metrics_dict(
    y_true: np.ndarray,
    y_prob_fake: np.ndarray,
    threshold: float = 0.5,
    label: str = "",
) -> dict[str, Any]:
    """Compute rigorous classification metrics with FAKE=1 as positive class."""
    y_pred = (y_prob_fake >= threshold).astype(int)

    acc  = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec  = float(recall_score(y_true, y_pred,    pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1   = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob_fake))
        pr_auc  = float(average_precision_score(y_true, y_prob_fake))
    else:
        roc_auc = pr_auc = 0.0

    return {
        "Model":              label,
        "Operating_Threshold": round(threshold, 4),
        "Accuracy":           round(acc,     4),
        "Balanced_Accuracy":  round(bal_acc, 4),
        "Precision":          round(prec,    4),
        "Recall":             round(rec,     4),
        "Specificity":        round(spec,    4),
        "F1":                 round(f1,      4),
        "ROC_AUC":            round(roc_auc, 4),
        "PR_AUC":             round(pr_auc,  4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def find_optimal_threshold_val(y_true_val: np.ndarray, y_prob_val: np.ndarray) -> float:
    """Determine optimal operating threshold via Youden's J statistic ON VALIDATION SET."""
    thresholds = np.linspace(0.01, 0.99, 500)
    best_j, best_thr = -1.0, 0.5
    for t in thresholds:
        y_pred = (y_prob_val >= t).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        cm = confusion_matrix(y_true_val, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_thr = float(t)
    return best_thr


# ---------------------------------------------------------------------------
# Feature Extraction Helper
# ---------------------------------------------------------------------------

def extract_features(model_bundle: Any, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Extract intermediate features from a model backbone."""
    model = model_bundle.model
    model.eval()
    name = model_bundle.name

    features_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(model_bundle.device)
            if "efficientnet" in name:
                feats = model.extract_features(imgs)
                pooled = model._avg_pooling(feats).flatten(1)
            elif "efficientformer" in name:
                feats = model.forward_features(imgs)
                pooled = feats.mean([2, 3]) if feats.ndim == 4 else feats
            elif "faster_vit" in name:
                feats = model.forward_features(imgs)
                pooled = feats.mean([2, 3]) if feats.ndim == 4 else (feats.mean(1) if feats.ndim == 3 else feats)
            else:
                pooled = model(imgs)

            features_list.append(pooled.cpu().numpy())
            targets_list.append(targets.numpy())

    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(targets_list, axis=0)
    return X, y


# ---------------------------------------------------------------------------
# Head Training Helper
# ---------------------------------------------------------------------------

class HeadClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_head_on_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    in_features: int,
    epochs: int = 35,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[HeadClassifier, float]:
    """Train linear classifier head on extracted backbone features."""
    head = HeadClassifier(in_features, num_classes=2)
    optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    best_val_loss = float("inf")
    best_weights = head.state_dict()

    for epoch in range(epochs):
        head.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = head(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Val eval
        head.eval()
        with torch.no_grad():
            val_out = head(X_val_t)
            val_loss = criterion(val_out, y_val_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_weights)
    return head, best_val_loss


# ---------------------------------------------------------------------------
# Main Execution Routine
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("MASTER ML PIPELINE AUDIT, DOMAIN ADAPTATION & EVALUATION")
    print("=" * 70)

    # Hardware optimization
    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(cpu_count)
    print(f"Hardware  : CPU with {cpu_count} threads | Seed: 42")

    config_path = Path("config/inference.yaml").resolve()
    config = load_config(config_path)

    data_cfg = config.get("data", {})
    train_dir = Path("C:/Users/kumar/.cache/kagglehub/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3/train")
    test_dir = Path("C:/Users/kumar/.cache/kagglehub/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3/test")

    if not train_dir.exists() or not test_dir.exists():
        print("Error: CIFAKE dataset directories not found!", file=sys.stderr)
        sys.exit(1)

    print(f"Train Dir : {train_dir}")
    print(f"Test Dir  : {test_dir}")

    # Standardize transforms
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    raw_train_ds = datasets.ImageFolder(str(train_dir), transform=eval_transform)
    raw_test_ds  = datasets.ImageFolder(str(test_dir),  transform=eval_transform)

    print(f"Raw Classes: {raw_train_ds.class_to_idx}")
    fake_idx = next(i for name, i in raw_train_ds.class_to_idx.items() if "fake" in name.lower())
    real_idx = next(i for name, i in raw_train_ds.class_to_idx.items() if "real" in name.lower())
    print(f"Index Mapping: FAKE = {fake_idx} | REAL = {real_idx}")

    # 1. Create strict 3-way disjoint split
    rng = np.random.RandomState(42)

    # From Train directory (100,000 samples)
    all_train_fake = [i for i, (_, y) in enumerate(raw_train_ds.samples) if y == fake_idx]
    all_train_real = [i for i, (_, y) in enumerate(raw_train_ds.samples) if y == real_idx]
    rng.shuffle(all_train_fake)
    rng.shuffle(all_train_real)

    # 4,000 for training (2k fake + 2k real)
    train_indices = sorted(all_train_fake[:2000] + all_train_real[:2000])
    # 1,000 for validation (500 fake + 500 real, strictly disjoint)
    val_indices   = sorted(all_train_fake[2000:2500] + all_train_real[2000:2500])

    # From Test directory (20,000 samples) -> 2,000 held-out test (1k fake + 1k real)
    rng_test = np.random.RandomState(1)
    all_test_fake = [i for i, (_, y) in enumerate(raw_test_ds.samples) if y == fake_idx]
    all_test_real = [i for i, (_, y) in enumerate(raw_test_ds.samples) if y == real_idx]
    test_indices  = sorted(
        rng_test.choice(all_test_fake, size=1000, replace=False).tolist() +
        rng_test.choice(all_test_real, size=1000, replace=False).tolist()
    )

    print("\nDataset Split Partitioning:")
    print(f"  Training Split   : {len(train_indices):,} images (2,000 FAKE + 2,000 REAL)")
    print(f"  Validation Split : {len(val_indices):,} images (500 FAKE + 500 REAL)")
    print(f"  Held-out Test    : {len(test_indices):,} images (1,000 FAKE + 1,000 REAL)")

    train_loader = DataLoader(Subset(raw_train_ds, train_indices), batch_size=64, shuffle=False)
    val_loader   = DataLoader(Subset(raw_train_ds, val_indices),   batch_size=64, shuffle=False)
    test_loader  = DataLoader(Subset(raw_test_ds,  test_indices),  batch_size=64, shuffle=False)

    # Load baseline pre-trained bundles
    bundles = load_forensics_models(config_path)

    val_probs_dict: dict[str, np.ndarray] = {}
    test_probs_dict: dict[str, np.ndarray] = {}
    test_raw_logits_dict: dict[str, np.ndarray] = {}
    y_val_true = None
    y_test_true = None

    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 2. Extract features, fine-tune classification heads, and update models
    for bundle in bundles:
        print(f"\n[Processing Model] {bundle.display_label} ({bundle.name})...")

        # Feature Extraction
        t0 = time.time()
        print("  Extracting Train features...")
        X_train, y_train_raw = extract_features(bundle, train_loader)
        print("  Extracting Val features...")
        X_val, y_val_raw     = extract_features(bundle, val_loader)
        print("  Extracting Test features...")
        X_test, y_test_raw   = extract_features(bundle, test_loader)
        print(f"  Feature extraction complete ({time.time()-t0:.1f}s). Feature dim: {X_train.shape[1]}")

        if y_val_true is None:
            # Positive class = FAKE (1), Authentic REAL = 0
            y_val_true  = (y_val_raw == fake_idx).astype(int)
            y_test_true = (y_test_raw == fake_idx).astype(int)

        # Train Head
        print("  Fine-tuning classifier head on CIFAKE domain features...")
        head, val_loss = train_head_on_features(
            X_train=X_train,
            y_train=y_train_raw,
            X_val=X_val,
            y_val=y_val_raw,
            in_features=X_train.shape[1],
            epochs=35,
            lr=1e-3,
        )
        print(f"  Head training converged (Best Val Loss: {val_loss:.4f})")

        # Update and save model weights
        head.eval()
        head_sd = head.state_dict()

        if "efficientnet" in bundle.name:
            bundle.model._fc.weight.data.copy_(head_sd["fc.weight"])
            bundle.model._fc.bias.data.copy_(head_sd["fc.bias"])
            torch.save(bundle.model.state_dict(), weights_dir / "efficientnet_b3.pth")
        elif "efficientformer" in bundle.name:
            bundle.model.head.weight.data.copy_(head_sd["fc.weight"])
            bundle.model.head.bias.data.copy_(head_sd["fc.bias"])
            torch.save(bundle.model.state_dict(), weights_dir / "efficientformerv2_s1.pth")
        elif "faster_vit" in bundle.name:
            bundle.model.head.weight.data.copy_(head_sd["fc.weight"])
            bundle.model.head.bias.data.copy_(head_sd["fc.bias"])
            torch.save(bundle.model.state_dict(), weights_dir / "faster_vit_2_224.pth")

        # Evaluate on Val (for threshold & weight tuning)
        with torch.no_grad():
            val_logits = head(torch.tensor(X_val, dtype=torch.float32)).numpy()
            val_p = F.softmax(torch.tensor(val_logits), dim=1).numpy()
            val_probs_dict[bundle.name] = val_p[:, fake_idx]

            test_logits = head(torch.tensor(X_test, dtype=torch.float32)).numpy()
            test_p = F.softmax(torch.tensor(test_logits), dim=1).numpy()
            test_probs_dict[bundle.name] = test_p[:, fake_idx]
            test_raw_logits_dict[bundle.name] = test_logits

    assert y_val_true is not None and y_test_true is not None

    # 3. Validation-Only Threshold Tuning & Optimal Ensemble Weight Search
    print("\n" + "=" * 70)
    print("STEP 3: VALIDATION-ONLY THRESHOLD & ENSEMBLE WEIGHT SELECTION")
    print("=" * 70)

    val_thresholds: dict[str, float] = {}
    for name in test_probs_dict:
        thr = find_optimal_threshold_val(y_val_true, val_probs_dict[name])
        val_thresholds[name] = thr
        print(f"  {name:22s} -> Selected Validation Threshold (Youden J): {thr:.4f}")

    # Optimize Ensemble Weights on Validation Set (Grid search)
    print("\n  Optimizing Ensemble Weights on Validation Set...")
    best_ens_auc = -1.0
    best_weights = (0.333, 0.333, 0.334)
    model_keys = ["faster_vit_2_224", "efficientnet_b3", "efficientformerv2_s1"]

    for w1 in np.linspace(0.1, 0.8, 15):
        for w2 in np.linspace(0.1, 0.8, 15):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            val_ens_p = (
                w1 * val_probs_dict[model_keys[0]] +
                w2 * val_probs_dict[model_keys[1]] +
                w3 * val_probs_dict[model_keys[2]]
            )
            auc = roc_auc_score(y_val_true, val_ens_p)
            if auc > best_ens_auc:
                best_ens_auc = auc
                best_weights = (float(w1), float(w2), float(w3))

    w_fv, w_en, w_ef = best_weights
    w_sum = w_fv + w_en + w_ef
    w_fv, w_en, w_ef = w_fv / w_sum, w_en / w_sum, w_ef / w_sum
    print(f"  Optimal Weights Found (Val AUC={best_ens_auc:.4f}):")
    print(f"    FasterViT-2-224     : {w_fv * 100:.1f}%")
    print(f"    EfficientNet-B3     : {w_en * 100:.1f}%")
    print(f"    EfficientFormerV2-S1: {w_ef * 100:.1f}%")

    val_ens_prob = (
        w_fv * val_probs_dict["faster_vit_2_224"] +
        w_en * val_probs_dict["efficientnet_b3"] +
        w_ef * val_probs_dict["efficientformerv2_s1"]
    )
    val_thresholds["Weighted Ensemble"] = find_optimal_threshold_val(y_val_true, val_ens_prob)
    print(f"  Weighted Ensemble Threshold (Val Youden J): {val_thresholds['Weighted Ensemble']:.4f}")

    # Compute Ensemble probabilities on Held-Out Test Set
    test_ens_prob = (
        w_fv * test_probs_dict["faster_vit_2_224"] +
        w_en * test_probs_dict["efficientnet_b3"] +
        w_ef * test_probs_dict["efficientformerv2_s1"]
    )
    test_probs_dict["Weighted Ensemble"] = test_ens_prob

    # 4. Final Blind Evaluation on Untouched Held-Out Test Set
    print("\n" + "=" * 70)
    print("STEP 4: FINAL BLIND EVALUATION ON UNTOUCHED HELD-OUT TEST SET")
    print("=" * 70)

    display_names = {
        "faster_vit_2_224":     "FasterViT-2-224",
        "weighted_ensemble":    "Weighted Ensemble",
        "efficientnet_b3":      "EfficientNet-B3",
        "efficientformerv2_s1": "EfficientFormerV2-S1",
    }

    test_models_ordered = [
        ("faster_vit_2_224",     "FasterViT-2-224"),
        ("Weighted Ensemble",    "Weighted Ensemble"),
        ("efficientnet_b3",      "EfficientNet-B3"),
        ("efficientformerv2_s1", "EfficientFormerV2-S1"),
    ]

    final_metrics_rows: list[dict[str, Any]] = []
    default_metrics_rows: list[dict[str, Any]] = []

    for key, dname in test_models_ordered:
        p_test = test_probs_dict[key]
        opt_thr = val_thresholds[key]

        # 1. At Calibrated Operating Threshold (Selected on Validation Set)
        m_opt = compute_metrics_dict(y_test_true, p_test, threshold=opt_thr, label=dname)
        final_metrics_rows.append(m_opt)

        # 2. At Default Threshold 0.50
        m_def = compute_metrics_dict(y_test_true, p_test, threshold=0.50, label=f"{dname} (τ=0.50)")
        default_metrics_rows.append(m_def)

    df_final = pd.DataFrame(final_metrics_rows)
    df_default = pd.DataFrame(default_metrics_rows)

    cols_table = [
        "Model", "Operating_Threshold", "Accuracy", "Balanced_Accuracy",
        "Precision", "Recall", "Specificity", "F1", "ROC_AUC", "PR_AUC"
    ]

    print("\n" + "=" * 100)
    print("FINAL EVALUATION BENCHMARK TABLE (Calibrated on Validation, Tested on Held-Out Test)")
    print("=" * 100)
    print(df_final[cols_table].to_string(index=False))
    print("=" * 100)

    # Save to CSV
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_final[cols_table].to_csv(output_dir / "model_comparison.csv", index=False)
    df_final[cols_table].to_csv(output_dir / "model_comparison_calibrated.csv", index=False)
    print(f"\nSaved master comparison table -> {output_dir / 'model_comparison.csv'}")

    # 5. Save Sample Prediction Logs (Sample ID, True Label, Raw Logits, P(Real), P(Fake), Predicted Label)
    records: list[dict[str, Any]] = []
    for i in range(len(y_test_true)):
        rec: dict[str, Any] = {
            "sample_id": i,
            "true_label": "FAKE" if y_test_true[i] == 1 else "REAL",
            "ground_truth": int(y_test_true[i]),
            "faster_vit_2_224_prob": float(test_probs_dict["faster_vit_2_224"][i]),
            "efficientnet_b3_prob":  float(test_probs_dict["efficientnet_b3"][i]),
            "efficientformerv2_s1_prob": float(test_probs_dict["efficientformerv2_s1"][i]),
            "ensemble_prob": float(test_probs_dict["Weighted Ensemble"][i]),
            "ensemble_pred": "FAKE" if test_probs_dict["Weighted Ensemble"][i] >= val_thresholds["Weighted Ensemble"] else "REAL",
        }
        records.append(rec)
    df_preds = pd.DataFrame(records)
    df_preds.to_csv(output_dir / "evaluation_predictions.csv", index=False)
    print(f"Saved per-sample predictions -> {output_dir / 'evaluation_predictions.csv'}")

    # 6. Generate Publication-Quality Plot Images
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING DIAGNOSTIC PLOT IMAGES (300 DPI)")
    print("=" * 70)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    plot_models_dict = {
        "FasterViT-2-224":     test_probs_dict["faster_vit_2_224"],
        "Weighted Ensemble":   test_probs_dict["Weighted Ensemble"],
        "EfficientNet-B3":     test_probs_dict["efficientnet_b3"],
        "EfficientFormerV2-S1": test_probs_dict["efficientformerv2_s1"],
    }

    colors = {
        "FasterViT-2-224":     "#38BDF8",  # Cyan
        "Weighted Ensemble":   "#4ADE80",  # Emerald Green
        "EfficientNet-B3":     "#F472B6",  # Rose
        "EfficientFormerV2-S1": "#FBBF24",  # Amber
    }

    # 6.1 ROC-AUC Curve
    plt.figure(figsize=(9, 7), dpi=300)
    for name, probs in plot_models_dict.items():
        fpr, tpr, _ = roc_curve(y_test_true, probs)
        auc = roc_auc_score(y_test_true, probs)
        lw = 2.8 if "Ensemble" in name else 2.0
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=colors[name], linewidth=lw)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#64748B", linewidth=1.5, label="Random Chance (AUC = 0.5000)")
    plt.title("Receiver Operating Characteristic (ROC-AUC) — CIFAKE Held-Out Test Set", fontsize=13, pad=15, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=10)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, labelpad=10)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_auc_curve.png", dpi=300)
    plt.close()
    print(f"  Saved ROC-AUC Curve -> {plots_dir / 'roc_auc_curve.png'}")

    # 6.2 PR-AUC Curve
    plt.figure(figsize=(9, 7), dpi=300)
    for name, probs in plot_models_dict.items():
        precision, recall, _ = precision_recall_curve(y_test_true, probs)
        ap = average_precision_score(y_test_true, probs)
        lw = 2.8 if "Ensemble" in name else 2.0
        plt.plot(recall, precision, label=f"{name} (PR-AUC / AP = {ap:.4f})", color=colors[name], linewidth=lw)
    baseline = float(np.sum(y_test_true) / len(y_test_true))
    plt.axhline(y=baseline, color="#64748B", linestyle="--", linewidth=1.5, label=f"No-Skill Baseline ({baseline:.2f})")
    plt.title("Precision-Recall (PR-AUC) Curve — CIFAKE Deepfake Detection", fontsize=13, pad=15, fontweight="bold")
    plt.xlabel("Recall (Sensitivity)", fontsize=12, labelpad=10)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12, labelpad=10)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(plots_dir / "pr_auc_curve.png", dpi=300)
    plt.close()
    print(f"  Saved PR-AUC Curve -> {plots_dir / 'pr_auc_curve.png'}")

    # 6.3 Confusion Matrices (2x4 Grid: Default 0.50 vs Operating Threshold)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=300)
    classes = ["REAL (0)", "FAKE (1)"]

    for col_idx, (name, probs) in enumerate(plot_models_dict.items()):
        # Default 0.50
        cm_def = confusion_matrix(y_test_true, (probs >= 0.50).astype(int), labels=[0, 1])
        ax0 = axes[0, col_idx]
        ax0.imshow(cm_def, interpolation="nearest", cmap="Blues")
        ax0.set_title(f"{name}\nDefault (τ = 0.50)", fontsize=11, fontweight="bold")
        _render_cm_cells(ax0, cm_def, classes)

        # Calibrated Operating Threshold
        thr = val_thresholds[name if name in val_thresholds else "faster_vit_2_224"]
        cm_opt = confusion_matrix(y_test_true, (probs >= thr).astype(int), labels=[0, 1])
        ax1 = axes[1, col_idx]
        ax1.imshow(cm_opt, interpolation="nearest", cmap="Greens")
        ax1.set_title(f"{name}\nOperating Threshold (τ = {thr:.3f})", fontsize=11, fontweight="bold", color="#4ADE80")
        _render_cm_cells(ax1, cm_opt, classes)

    axes[0, 0].set_ylabel("True Class (Default τ=0.50)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("True Class (Operating τ*)", fontsize=12, fontweight="bold")
    plt.suptitle("Confusion Matrix Analysis (Top: Default 0.50 vs Bottom: Operating Threshold)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrices.png", dpi=300)
    plt.close()
    print(f"  Saved Confusion Matrices -> {plots_dir / 'confusion_matrices.png'}")

    # 6.4 Metrics Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    metrics_to_plot = ["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC_AUC"]
    models = df_final["Model"].tolist()
    x = np.arange(len(models))
    width = 0.11
    palette = ["#38BDF8", "#818CF8", "#F472B6", "#FB923C", "#FBBF24", "#4ADE80", "#34D399"]

    for i, metric in enumerate(metrics_to_plot):
        values = df_final[metric].tolist()
        offset = (i - len(metrics_to_plot) / 2) * width + width / 2
        rects = ax.bar(x + offset, values, width, label=metric.replace("_", " "), color=palette[i % len(palette)], alpha=0.9)
        for rect in rects:
            h = rect.get_height()
            if h > 0.05:
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, color="#F8FAFC", rotation=45)

    ax.set_ylabel("Score (0.00 - 1.00)", fontsize=12, fontweight="bold")
    ax.set_title("Comprehensive Model Comparison on CIFAKE Held-Out Test Set", fontsize=14, pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1.15])
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", ncol=3)
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_comparison.png", dpi=300)
    plt.close()
    print(f"  Saved Metrics Comparison Bar Chart -> {plots_dir / 'metrics_comparison.png'}")

    # 7. Ablation Study Execution
    print("\n" + "=" * 70)
    print("STEP 6: SYSTEMATIC ABLATION STUDY GENERATION")
    print("=" * 70)
    ablation_dir = output_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # A. Architecture Ablation
    arch_rows = [
        compute_metrics_dict(y_test_true, test_probs_dict["efficientnet_b3"],      threshold=val_thresholds["efficientnet_b3"],      label="EfficientNet-B3 (solo)"),
        compute_metrics_dict(y_test_true, test_probs_dict["efficientformerv2_s1"],  threshold=val_thresholds["efficientformerv2_s1"],  label="EfficientFormerV2-S1 (solo)"),
        compute_metrics_dict(y_test_true, test_probs_dict["faster_vit_2_224"],      threshold=val_thresholds["faster_vit_2_224"],      label="FasterViT-2-224 (solo)"),
        compute_metrics_dict(y_test_true, 0.5 * test_probs_dict["faster_vit_2_224"] + 0.5 * test_probs_dict["efficientnet_b3"], threshold=0.5, label="Pairwise Ensemble (FV + EN)"),
        compute_metrics_dict(y_test_true, (test_probs_dict["faster_vit_2_224"] + test_probs_dict["efficientnet_b3"] + test_probs_dict["efficientformerv2_s1"]) / 3.0, threshold=0.5, label="Full Ensemble (Uniform 1:1:1)"),
        compute_metrics_dict(y_test_true, test_probs_dict["Weighted Ensemble"], threshold=val_thresholds["Weighted Ensemble"], label="Full Ensemble (Optimal Weighted)"),
    ]
    df_arch = pd.DataFrame(arch_rows)
    df_arch.to_csv(ablation_dir / "ablation_architecture.csv", index=False)

    # B. Ensemble Weights Ablation
    weight_schemes = {
        "Uniform (33.3% / 33.3% / 33.3%)": (0.333, 0.333, 0.334),
        "Optimal Weighted (Val Tuned)": (w_fv, w_en, w_ef),
        "FasterViT-Heavy (70% / 20% / 10%)": (0.70, 0.20, 0.10),
        "EfficientNet-Heavy (60% / 30% / 10%)": (0.30, 0.60, 0.10),
        "EfficientFormer-Heavy (60% / 25% / 15%)": (0.25, 0.15, 0.60),
    }
    w_rows = []
    for s_name, (w1, w2, w3) in weight_schemes.items():
        ens_p = w1 * test_probs_dict["faster_vit_2_224"] + w2 * test_probs_dict["efficientnet_b3"] + w3 * test_probs_dict["efficientformerv2_s1"]
        w_rows.append(compute_metrics_dict(y_test_true, ens_p, threshold=0.50, label=s_name))
    df_weights = pd.DataFrame(w_rows)
    df_weights.to_csv(ablation_dir / "ablation_ensemble_weights.csv", index=False)

    # C. Threshold Ablation
    thr_rows = []
    for m_key, m_dname in [("faster_vit_2_224", "FasterViT-2-224"), ("Weighted Ensemble", "Weighted Ensemble"), ("efficientnet_b3", "EfficientNet-B3")]:
        p = test_probs_dict[m_key]
        thr_rows.append(compute_metrics_dict(y_test_true, p, threshold=0.50, label=f"{m_dname} @ Default τ=0.50"))
        opt_t = val_thresholds[m_key]
        thr_rows.append(compute_metrics_dict(y_test_true, p, threshold=opt_t, label=f"{m_dname} @ Frozen Val τ*={opt_t:.3f}"))
    df_thr = pd.DataFrame(thr_rows)
    df_thr.to_csv(ablation_dir / "ablation_threshold.csv", index=False)

    # Master Summary
    df_summary = pd.concat([
        df_arch.assign(Section="A_Architecture"),
        df_weights.assign(Section="B_EnsembleWeights"),
        df_thr.assign(Section="C_ThresholdOptimization")
    ], ignore_index=True)
    df_summary.to_csv(ablation_dir / "ablation_summary.csv", index=False)
    print(f"  Saved master ablation study -> {ablation_dir / 'ablation_summary.csv'}")

    print("\n" + "=" * 70)
    print("PIPELINE AUDIT AND EVALUATION COMPLETE!")
    print("=" * 70)


def _render_cm_cells(ax: Any, cm: np.ndarray, classes: list[str]) -> None:
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=9.5)
    ax.set_yticklabels(classes, fontsize=9.5)
    ax.set_xlabel("Predicted Class", fontsize=9.5)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val < thresh else "black"
            ax.text(j, i, f"{val:,}\n({val/cm.sum()*100:.1f}%)",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color, fontsize=9.5, fontweight="bold")


if __name__ == "__main__":
    main()
