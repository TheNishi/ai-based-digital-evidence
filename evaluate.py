"""Evaluation script for Deepfake Detection models and Ensemble.

Evaluates EfficientNet, FasterViT, EfficientFormerV2-S1, and their Ensemble
on the fixed held-out CIFAKE test set using FAKE as the positive class.

Produces:
- results/model_comparison.csv
- results/evaluation_predictions.csv

NOTE: Uses DataLoader batched inference. Pass --max-samples N to evaluate
      on a stratified subset (e.g. 2000 = 1000 FAKE + 1000 REAL) for speed
      on CPU. Full 20K evaluation is recommended when GPU is available.
"""

from __future__ import annotations

import argparse
import random
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from orchestration.forensics import load_forensics_models
from orchestration.orchestrator import load_config


def compute_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute optimal threshold via Youden's J statistic."""
    thresholds = np.linspace(0.01, 0.99, 500)
    best_j, best_thr = -1.0, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_thr = float(t)
    return best_thr


def compute_metrics(
    y_true: np.ndarray, y_prob_fake: np.ndarray, threshold: float | None = None
) -> dict[str, Any]:
    """Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC (FAKE=positive)."""
    if threshold is None:
        threshold = compute_optimal_threshold(y_true, y_prob_fake)

    y_pred = (y_prob_fake >= threshold).astype(int)
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec  = float(recall_score(y_true, y_pred,    pos_label=1, zero_division=0))
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1   = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob_fake))
        pr_auc  = float(average_precision_score(y_true, y_prob_fake))
    else:
        roc_auc = pr_auc = 0.0
    return {
        "Threshold":   round(threshold, 4),
        "Accuracy":    round(acc,     4),
        "Precision":   round(prec,    4),
        "Recall":      round(rec,     4),
        "Specificity": round(spec,    4),
        "F1":          round(f1,      4),
        "ROC_AUC":     round(roc_auc, 4),
        "PR_AUC":      round(pr_auc,  4),
    }


def _get_fake_class_idx(class_to_idx: dict[str, int]) -> int:
    fake_keywords = {"fake", "deepfake", "synthetic"}
    for cls_name, idx in class_to_idx.items():
        if cls_name.lower() in fake_keywords:
            return idx
    return 0


def stratified_indices(
    samples: list[tuple[str, int]],
    fake_idx: int,
    max_samples: int,
    seed: int = 1,
) -> list[int]:
    """Return balanced stratified indices: max_samples // 2 per class."""
    rng      = random.Random(seed)
    fake_ids = [i for i, (_, lbl) in enumerate(samples) if lbl == fake_idx]
    real_ids = [i for i, (_, lbl) in enumerate(samples) if lbl != fake_idx]
    per_cls  = max_samples // 2
    fake_ids = rng.sample(fake_ids, min(per_cls, len(fake_ids)))
    real_ids = rng.sample(real_ids, min(per_cls, len(real_ids)))
    return sorted(fake_ids + real_ids)


def run_evaluation(
    config_path: Path = Path("config/inference.yaml"),
    test_dir_override: Path | None = None,
    output_dir: Path = Path("results"),
    max_samples: int | None = None,
    seed: int = 1,
) -> bool:
    config_path = config_path.resolve()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return False

    config   = load_config(config_path)
    data_cfg = config.get("data", {})

    dataset_name  = data_cfg.get("dataset_name", "CIFAKE")
    data_root_str = data_cfg.get("dataset_root", data_cfg.get("root", "data"))
    test_split    = data_cfg.get("test_split", "test")

    if test_dir_override:
        test_dir = test_dir_override.resolve()
    else:
        test_dir_str = data_cfg.get("test_dir")
        if test_dir_str:
            test_dir = Path(test_dir_str).expanduser()
        else:
            test_dir = Path(data_root_str).expanduser() / test_split
        if not test_dir.is_absolute():
            test_dir = (Path.cwd() / test_dir).resolve()

    print("==================================================")
    print("DEEPFAKE MODEL EVALUATION PIPELINE")
    print("==================================================")
    print(f"Dataset Name  : {dataset_name}")
    print(f"Test Directory: {test_dir}")

    if not test_dir.exists():
        print(f"\nERROR: Test directory not found: {test_dir}")
        return False

    device_str = config.get("device", "cuda")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    # Use all CPU threads for faster inference
    if device.type == "cpu":
        import os
        torch.set_num_threads(os.cpu_count() or 4)
    print(f"Device        : {device}  (threads={torch.get_num_threads()})")

    print("\nLoading models...")
    bundles = load_forensics_models(config_path)
    if not bundles:
        print("Error: No models loaded.", file=sys.stderr)
        return False

    img_size = int(data_cfg.get("img_size", 224))

    def _ensure_rgb(img: Image.Image) -> Image.Image:
        return img.convert("RGB") if img.mode != "RGB" else img

    probe_tf = transforms.Compose([
        transforms.Lambda(_ensure_rgb),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    probe_ds     = datasets.ImageFolder(str(test_dir), transform=probe_tf)
    class_to_idx = probe_ds.class_to_idx
    fake_idx     = _get_fake_class_idx(class_to_idx)
    all_samples  = probe_ds.samples

    # Stratified subset if requested
    if max_samples is not None and max_samples < len(all_samples):
        indices = stratified_indices(all_samples, fake_idx, max_samples, seed=seed)
        print(f"\nUsing stratified subset: {len(indices)} / {len(all_samples)} images (seed={seed})")
    else:
        indices = list(range(len(all_samples)))
        print(f"\nUsing full test set: {len(indices)} images")

    n_fake       = sum(1 for i in indices if all_samples[i][1] == fake_idx)
    n_real       = len(indices) - n_fake
    total_images = len(indices)

    print(f"\nDetected classes  : {class_to_idx}")
    print(f"FAKE class index  : {fake_idx}")
    print("\n--------------------------------------------------")
    print(f"Total test images : {total_images}")
    print(f"Real images       : {n_real}")
    print(f"Fake images       : {n_fake}")
    print("--------------------------------------------------")

    if n_real == 0 or n_fake == 0:
        print("\n[ERROR] Need images in both REAL and FAKE classes.")
        return False

    num_workers      = 0
    all_targets: np.ndarray | None = None
    per_model_probs: dict[str, np.ndarray] = {}

    for bundle in bundles:
        print(f"\n[Inference] {bundle.name} ...")
        model_ds  = datasets.ImageFolder(str(test_dir), transform=bundle.transform)
        subset_ds = Subset(model_ds, indices)
        loader    = DataLoader(
            subset_ds, batch_size=64, shuffle=False,
            num_workers=num_workers, pin_memory=False,
        )
        probs_list:   list[np.ndarray] = []
        targets_list: list[np.ndarray] = []
        bundle.model.eval()
        with torch.inference_mode():
            for i, (images, targets) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                logits = bundle.model(images)
                probs  = F.softmax(logits, dim=1)
                p_fake = probs[:, fake_idx].cpu().numpy() if fake_idx < probs.shape[1] else (1.0 - probs[:, 1]).cpu().numpy()
                probs_list.append(p_fake)
                targets_list.append(targets.numpy())
                done = min((i + 1) * 64, total_images)
                print(f"  [{done:>5}/{total_images}]", end="\r", flush=True)
        print(f"  [{total_images}/{total_images}] done          ")
        per_model_probs[bundle.name] = np.concatenate(probs_list)
        model_targets = np.concatenate(targets_list)
        if all_targets is None:
            all_targets = model_targets

    assert all_targets is not None
    y_true = (all_targets == fake_idx).astype(int)

    # Weighted ensemble
    model_weights = {
        "faster_vit_2_224":     0.50,
        "efficientnet_b3":      0.35,
        "efficientformerv2_s1": 0.15,
    }
    w_sum      = sum(model_weights.get(n, 1.0) for n in per_model_probs)
    p_ensemble = sum(model_weights.get(n, 1.0) * p for n, p in per_model_probs.items()) / w_sum

    # Save per-image predictions
    records: list[dict[str, Any]] = []
    for i in range(len(y_true)):
        rec: dict[str, Any] = {"ground_truth": int(y_true[i])}
        for name, arr in per_model_probs.items():
            rec[f"{name}_prob"] = float(arr[i])
        rec["ensemble_prob"] = float(p_ensemble[i])
        records.append(rec)

    output_dir.mkdir(parents=True, exist_ok=True)
    df_preds = pd.DataFrame(records)
    predictions_csv = output_dir / "evaluation_predictions.csv"
    df_preds.to_csv(predictions_csv, index=False)
    print(f"\nSaved per-image predictions -> {predictions_csv}")

    # Compute and display metrics
    evaluation_targets = [
        ("efficientnet_b3",      "EfficientNet-B3"),
        ("faster_vit_2_224",     "FasterViT-2-224"),
        ("efficientformerv2_s1", "EfficientFormerV2-S1"),
        ("Ensemble",             "Ensemble"),
    ]
    rows: list[dict[str, Any]] = []
    for key, display_name in evaluation_targets:
        y_prob = p_ensemble if key == "Ensemble" else per_model_probs.get(key)
        if y_prob is None:
            continue
        metrics          = compute_metrics(y_true, y_prob)
        metrics["Model"] = display_name
        rows.append(metrics)

    cols          = ["Model", "Threshold", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC_AUC", "PR_AUC"]
    df_comparison = pd.DataFrame(rows)[cols]
    comparison_csv = output_dir / "model_comparison.csv"
    df_comparison.to_csv(comparison_csv, index=False)

    print("\n==================================================")
    print("FINAL MODEL COMPARISON TABLE")
    print(f"Dataset : {dataset_name} | Evaluated on: {total_images} images")
    print("==================================================")
    print(df_comparison.to_string(index=False))
    print("==================================================")
    print(f"Saved comparison table -> {comparison_csv}\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detection models & ensemble")
    parser.add_argument("--config",      type=Path, default=Path("config/inference.yaml"))
    parser.add_argument("--test-dir",    type=Path, help="Override path to test dataset directory")
    parser.add_argument("--output-dir",  type=Path, default=Path("results"))
    parser.add_argument("--max-samples", type=int,  default=None,
                        help="Stratified subset size (half FAKE, half REAL). Use for fast CPU eval.")
    parser.add_argument("--seed",        type=int,  default=1)
    args = parser.parse_args()
    success = run_evaluation(
        config_path=args.config,
        test_dir_override=args.test_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()