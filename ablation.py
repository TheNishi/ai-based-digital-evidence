"""Ablation Study for Deepfake Detection on CIFAKE Dataset.

This script systematically evaluates:
  A) Architecture Ablation   - each backbone individually vs pairwise vs full ensemble
  B) Augmentation Ablation   - with/without center-crop vs resize-only
  C) Ensemble Weight Ablation - uniform vs weighted ensemble strategies
  D) Threshold Ablation       - fixed 0.5 vs optimal threshold (Youden's J statistic)

Outputs (to results/ablation/):
  - ablation_architecture.csv
  - ablation_augmentation.csv
  - ablation_ensemble_weights.csv
  - ablation_threshold.csv
  - ablation_summary.csv  (master table)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from orchestration.forensics import load_forensics_models
from orchestration.orchestrator import load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".ppm", ".pgm", ".tif"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prob_fake: np.ndarray,
    threshold: float = 0.5,
    label: str = "",
) -> dict[str, Any]:
    """Compute classification metrics with FAKE=1 as positive class."""
    y_pred = (y_prob_fake >= threshold).astype(int)

    acc  = float(accuracy_score(y_true, y_pred))
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

    row: dict[str, Any] = {
        "Experiment":  label,
        "Accuracy":    round(acc,     4),
        "Precision":   round(prec,    4),
        "Recall":      round(rec,     4),
        "Specificity": round(spec,    4),
        "F1":          round(f1,      4),
        "ROC_AUC":     round(roc_auc, 4),
        "PR_AUC":      round(pr_auc,  4),
        "Threshold":   round(threshold, 4),
    }
    return row


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Youden's J statistic to pick optimal threshold."""
    thresholds = np.linspace(0.0, 1.0, 501)
    best_j, best_t = -1.0, 0.5
    for t in thresholds:
        y_p = (y_prob >= t).astype(int)
        if len(np.unique(y_p)) < 2:
            continue
        cm  = confusion_matrix(y_true, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def load_predictions_from_csv(csv_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load existing model predictions from evaluate.py output."""
    df = pd.read_csv(csv_path)
    y_true = df["ground_truth"].to_numpy(dtype=int)
    probs: dict[str, np.ndarray] = {}
    for col in df.columns:
        if col.endswith("_prob") and col != "ensemble_prob":
            model_name = col[:-5]
            probs[model_name] = df[col].to_numpy(dtype=float)
    return y_true, probs


def collect_raw_predictions_batched(
    config_path: Path,
    test_dir: Path,
    max_samples: int | None = None,
    seed: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Run all models over test_dir using DataLoader batched inference and return:
      y_true  : (N,) ground-truth labels (1=FAKE, 0=REAL)
      probs   : dict model_name -> (N,) P(FAKE)
    """
    bundles = load_forensics_models(config_path)
    if not bundles:
        print("Error: No models loaded.", file=sys.stderr)
        sys.exit(1)

    raw_dataset = datasets.ImageFolder(str(test_dir))
    fake_idx = next(
        (idx for name, idx in raw_dataset.class_to_idx.items() if "fake" in name.lower()), 0
    )

    indices = list(range(len(raw_dataset)))
    if max_samples is not None and max_samples < len(raw_dataset):
        rng = np.random.RandomState(seed)
        fake_indices = [i for i, (_, target) in enumerate(raw_dataset.samples) if target == fake_idx]
        real_indices = [i for i, (_, target) in enumerate(raw_dataset.samples) if target != fake_idx]
        n_half = max_samples // 2
        sel_fake = rng.choice(fake_indices, size=min(n_half, len(fake_indices)), replace=False)
        sel_real = rng.choice(real_indices, size=min(n_half, len(real_indices)), replace=False)
        indices = sorted(np.concatenate([sel_fake, sel_real]).tolist())

    per_model_probs: dict[str, np.ndarray] = {}
    all_targets: np.ndarray | None = None
    batch_size = 64

    for bundle in bundles:
        dataset = datasets.ImageFolder(str(test_dir), transform=bundle.transform)
        if len(indices) < len(raw_dataset):
            dataset = Subset(dataset, indices)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        probs_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []

        print(f"\n[Inference] {bundle.name} ...")
        for i, (images, targets) in enumerate(loader):
            images = images.to(bundle.device)
            with torch.inference_mode():
                logits = bundle.model(images)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs[:, fake_idx])
            targets_list.append(targets.numpy())

        per_model_probs[bundle.name] = np.concatenate(probs_list)
        if all_targets is None:
            all_targets = np.concatenate(targets_list)

    assert all_targets is not None
    y_true = (all_targets == fake_idx).astype(int)
    return y_true, per_model_probs


# ---------------------------------------------------------------------------
# Ablation A: Architecture
# ---------------------------------------------------------------------------

def ablation_architecture(
    y_true: np.ndarray, probs: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Test each backbone alone and every pairwise + full ensemble."""
    weights_full = {
        "faster_vit_2_224": 0.50,
        "efficientnet_b3": 0.35,
        "efficientformerv2_s1": 0.15,
    }

    rows: list[dict[str, Any]] = []

    # Individual models
    display = {
        "efficientnet_b3":      "EfficientNet-B3 (solo)",
        "efficientformerv2_s1": "EfficientFormerV2-S1 (solo)",
        "faster_vit_2_224":     "FasterViT-2-224 (solo)",
    }
    for name, label in display.items():
        if name in probs:
            rows.append(compute_metrics(y_true, probs[name], label=label))

    # Pairwise ensembles (uniform)
    model_names = list(probs.keys())
    from itertools import combinations
    for m1, m2 in combinations(model_names, 2):
        avg = (probs[m1] + probs[m2]) / 2.0
        label = f"Ensemble ({m1.replace('_2_224','').replace('_b3','').replace('formerv2_s1','former')[:7].upper()} + {m2.replace('_2_224','').replace('_b3','').replace('formerv2_s1','former')[:7].upper()}, uniform)"
        rows.append(compute_metrics(y_true, avg, label=label))

    # Full ensemble – uniform
    all_p = np.stack(list(probs.values()), axis=1).mean(axis=1)
    rows.append(compute_metrics(y_true, all_p, label="Full Ensemble (uniform)"))

    # Full ensemble – weighted (FasterViT 50%, EffNet 35%, EffFormer 15%)
    w_sum  = sum(weights_full.get(n, 1.0) for n in probs)
    w_avg  = sum(weights_full.get(n, 1.0) * probs[n] for n in probs) / w_sum
    rows.append(compute_metrics(y_true, w_avg, label="Full Ensemble (weighted)"))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ablation B: Augmentation (resize-only vs center-crop)
# ---------------------------------------------------------------------------

def ablation_augmentation(
    config_path: Path,
    test_dir:    Path,
    y_true:      np.ndarray,
    probs_full:  dict[str, np.ndarray],
    max_samples: int | None = None,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Compare two preprocessing pipelines:
      1. Standard: Resize(224) + CenterCrop(224) + ToTensor + Normalize  [already in probs_full]
      2. Minimal:  Resize(224)                   + ToTensor + Normalize  [re-run]
    """
    rows: list[dict[str, Any]] = []

    # Standard pipeline – reuse collected probs
    display = {
        "efficientnet_b3":      "EfficientNet-B3 (std aug)",
        "efficientformerv2_s1": "EfficientFormerV2-S1 (std aug)",
        "faster_vit_2_224":     "FasterViT-2-224 (std aug)",
    }
    for name, label in display.items():
        if name in probs_full:
            rows.append(compute_metrics(y_true, probs_full[name], label=label))

    # Minimal pipeline – no center crop
    minimal_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    bundles = load_forensics_models(config_path)
    raw_dataset = datasets.ImageFolder(str(test_dir))
    fake_idx = next(
        (idx for n, idx in raw_dataset.class_to_idx.items() if "fake" in n.lower()), 0
    )

    indices = list(range(len(raw_dataset)))
    if max_samples is not None and max_samples < len(raw_dataset):
        rng = np.random.RandomState(seed)
        fake_indices = [i for i, (_, target) in enumerate(raw_dataset.samples) if target == fake_idx]
        real_indices = [i for i, (_, target) in enumerate(raw_dataset.samples) if target != fake_idx]
        n_half = max_samples // 2
        sel_fake = rng.choice(fake_indices, size=min(n_half, len(fake_indices)), replace=False)
        sel_real = rng.choice(real_indices, size=min(n_half, len(real_indices)), replace=False)
        indices = sorted(np.concatenate([sel_fake, sel_real]).tolist())

    min_probs: dict[str, np.ndarray] = {}
    for bundle in bundles:
        dataset = datasets.ImageFolder(str(test_dir), transform=minimal_tf)
        if len(indices) < len(raw_dataset):
            dataset = Subset(dataset, indices)

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        probs_list: list[np.ndarray] = []
        for images, _ in loader:
            images = images.to(bundle.device)
            with torch.inference_mode():
                logits = bundle.model(images)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs[:, fake_idx])
        min_probs[bundle.name] = np.concatenate(probs_list)

    display_min = {
        "efficientnet_b3":      "EfficientNet-B3 (direct resize)",
        "efficientformerv2_s1": "EfficientFormerV2-S1 (direct resize)",
        "faster_vit_2_224":     "FasterViT-2-224 (direct resize)",
    }
    for name, label in display_min.items():
        if name in min_probs and len(min_probs[name]) == len(y_true):
            rows.append(compute_metrics(y_true, min_probs[name], label=label))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ablation C: Ensemble weight strategies
# ---------------------------------------------------------------------------

def ablation_ensemble_weights(
    y_true: np.ndarray, probs: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Test multiple ensemble weighting schemes."""
    strategies: dict[str, dict[str, float]] = {
        "Uniform (1:1:1)":            {n: 1.0 for n in probs},
        "Optimal Weighted (FV=50%, EN=35%, EF=15%)": {
            "faster_vit_2_224": 0.50,
            "efficientnet_b3":  0.35,
            "efficientformerv2_s1": 0.15,
        },
        "FasterViT-heavy (FV=70%, EN=20%, EF=10%)": {
            "faster_vit_2_224": 0.70,
            "efficientnet_b3":  0.20,
            "efficientformerv2_s1": 0.10,
        },
        "EfficientNet-heavy (EN=60%, FV=30%, EF=10%)": {
            "efficientnet_b3":  0.60,
            "faster_vit_2_224": 0.30,
            "efficientformerv2_s1": 0.10,
        },
        "EfficientFormer-heavy (EF=60%, FV=25%, EN=15%)": {
            "efficientformerv2_s1": 0.60,
            "faster_vit_2_224": 0.25,
            "efficientnet_b3": 0.15,
        },
    }

    rows: list[dict[str, Any]] = []
    for label, ws in strategies.items():
        w_sum = sum(ws.get(n, 1.0) for n in probs)
        ens   = sum(ws.get(n, 1.0) * probs[n] for n in probs) / w_sum
        rows.append(compute_metrics(y_true, ens, label=label))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ablation D: Threshold
# ---------------------------------------------------------------------------

def ablation_threshold(
    y_true: np.ndarray, probs: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Compare fixed 0.5 vs Youden-optimal threshold for each model + ensemble."""
    weights_full = {
        "faster_vit_2_224": 0.50,
        "efficientnet_b3":  0.35,
        "efficientformerv2_s1": 0.15,
    }
    w_sum = sum(weights_full.get(n, 1.0) for n in probs)
    ens   = sum(weights_full.get(n, 1.0) * probs[n] for n in probs) / w_sum
    all_probs = {**probs, "Ensemble (weighted)": ens}

    display = {
        "efficientnet_b3":      "EfficientNet-B3",
        "efficientformerv2_s1": "EfficientFormerV2-S1",
        "faster_vit_2_224":     "FasterViT-2-224",
        "Ensemble (weighted)":  "Ensemble (weighted)",
    }

    rows: list[dict[str, Any]] = []
    for name, label in display.items():
        if name not in all_probs:
            continue
        p = all_probs[name]
        rows.append(compute_metrics(y_true, p, threshold=0.5, label=f"{label} @ thr=0.5"))
        opt = best_threshold(y_true, p)
        rows.append(compute_metrics(y_true, p, threshold=opt,  label=f"{label} @ thr={opt:.3f} (Youden Optimal)"))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study for deepfake detection on CIFAKE")
    parser.add_argument("--config",          type=Path, default=Path("config/ablation.yaml"))
    parser.add_argument("--predictions-csv", type=Path, default=None,
                        help="Path to evaluation_predictions.csv to compute ablations instantly without re-inference")
    parser.add_argument("--output-dir",      type=Path, default=Path("results/ablation"))
    parser.add_argument("--max-samples",     type=int,  default=None)
    parser.add_argument("--seed",            type=int,  default=1)
    parser.add_argument(
        "--skip-augmentation", action="store_true",
        help="Skip the augmentation ablation (slower – reruns inference with minimal transforms)",
    )
    args = parser.parse_args()

    # Hardware optimization
    try:
        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(cpu_count)
    except Exception:
        pass

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ABLATION STUDY - CIFAKE Deepfake Detection")
    print("=" * 60)

    # Check if predictions CSV is provided or already exists
    pred_path = args.predictions_csv
    if pred_path is None:
        default_pred = Path("results/evaluation_predictions.csv")
        if default_pred.exists():
            pred_path = default_pred

    if pred_path is not None and pred_path.exists():
        print(f"[Step 1/5] Loading pre-computed predictions from: {pred_path}")
        y_true, probs = load_predictions_from_csv(pred_path)
        print(f"  Loaded {len(y_true)} samples | FAKE={y_true.sum()} | REAL={(1-y_true).sum()}")
    else:
        config_path = args.config.resolve()
        if not config_path.exists():
            print(f"Config not found: {config_path}", file=sys.stderr)
            sys.exit(1)

        config = load_config(config_path)
        data_cfg = config.get("data", {})
        test_dir_str = data_cfg.get("test_dir") or (
            str(Path(data_cfg.get("root", "data")).expanduser() / data_cfg.get("test_split", "test"))
        )
        test_dir = Path(test_dir_str).expanduser()
        if not test_dir.is_absolute():
            test_dir = (Path.cwd() / test_dir).resolve()

        if not test_dir.exists():
            print(f"Test directory not found: {test_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Config    : {config_path}")
        print(f"Test dir  : {test_dir}")
        print(f"[Step 1/5] Collecting raw per-model predictions (batched DataLoader)...")
        y_true, probs = collect_raw_predictions_batched(
            config_path=config_path,
            test_dir=test_dir,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        print(f"  Collected {len(y_true)} samples | FAKE={y_true.sum()} | REAL={(1-y_true).sum()}")

    # Save raw predictions for reproducibility
    records: list[dict[str, Any]] = []
    for i in range(len(y_true)):
        rec: dict[str, Any] = {"ground_truth": int(y_true[i])}
        for name, p in probs.items():
            rec[f"{name}_prob"] = float(p[i])
        records.append(rec)
    pd.DataFrame(records).to_csv(output_dir / "raw_predictions.csv", index=False)
    print(f"  Saved raw predictions -> {output_dir / 'raw_predictions.csv'}")

    # --- Step 2: Architecture Ablation ---
    print("\n[Step 2/5] Architecture ablation...")
    df_arch = ablation_architecture(y_true, probs)
    df_arch.to_csv(output_dir / "ablation_architecture.csv", index=False)
    print(df_arch.to_string(index=False))

    # --- Step 3: Ensemble weight ablation ---
    print("\n[Step 3/5] Ensemble weight ablation...")
    df_ens = ablation_ensemble_weights(y_true, probs)
    df_ens.to_csv(output_dir / "ablation_ensemble_weights.csv", index=False)
    print(df_ens.to_string(index=False))

    # --- Step 4: Threshold ablation ---
    print("\n[Step 4/5] Threshold ablation...")
    df_thr = ablation_threshold(y_true, probs)
    df_thr.to_csv(output_dir / "ablation_threshold.csv", index=False)
    print(df_thr.to_string(index=False))

    # --- Step 5: Augmentation ablation (optional – reruns inference) ---
    if not args.skip_augmentation:
        try:
            config_path = args.config.resolve()
            config = load_config(config_path)
            data_cfg = config.get("data", {})
            test_dir_str = data_cfg.get("test_dir") or (
                str(Path(data_cfg.get("root", "data")).expanduser() / data_cfg.get("test_split", "test"))
            )
            test_dir = Path(test_dir_str).expanduser()
            if not test_dir.is_absolute():
                test_dir = (Path.cwd() / test_dir).resolve()
            print("\n[Step 5/5] Augmentation ablation (minimal transforms)...")
            df_aug = ablation_augmentation(
                config_path=config_path,
                test_dir=test_dir,
                y_true=y_true,
                probs_full=probs,
                max_samples=args.max_samples,
                seed=args.seed,
            )
            df_aug.to_csv(output_dir / "ablation_augmentation.csv", index=False)
            print(df_aug.to_string(index=False))
        except Exception as e:
            print(f"Warning: Augmentation ablation failed ({e}), skipping.")
            df_aug = pd.DataFrame()
    else:
        print("\n[Step 5/5] Augmentation ablation skipped (--skip-augmentation).")
        df_aug = pd.DataFrame()

    # --- Master summary ---
    print("\n[Summary] Compiling master ablation table...")
    dfs: list[pd.DataFrame] = []
    dfs.append(df_arch.assign(Section="A_Architecture"))
    dfs.append(df_ens.assign(Section="C_EnsembleWeights"))
    dfs.append(df_thr.assign(Section="D_Threshold"))
    if not df_aug.empty:
        dfs.append(df_aug.assign(Section="B_Augmentation"))
    df_summary = pd.concat(dfs, ignore_index=True)
    cols = ["Section", "Experiment", "Accuracy", "Precision", "Recall",
            "Specificity", "F1", "ROC_AUC", "PR_AUC", "Threshold"]
    df_summary = df_summary[[c for c in cols if c in df_summary.columns]]
    df_summary.to_csv(output_dir / "ablation_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)
    print(f"  results/ablation/ablation_architecture.csv")
    print(f"  results/ablation/ablation_ensemble_weights.csv")
    print(f"  results/ablation/ablation_threshold.csv")
    if not df_aug.empty:
        print(f"  results/ablation/ablation_augmentation.csv")
    print(f"  results/ablation/ablation_summary.csv")
    print()


if __name__ == "__main__":
    main()
