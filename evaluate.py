"""Evaluation script for Deepfake Detection models and Ensemble.

Evaluates EfficientNet, FasterViT, EfficientFormerV2-S1, and their Ensemble
on the fixed held-out test set using FAKE as the positive class.

Produces:
- results/model_comparison.csv
- results/evaluation_predictions.csv
"""

from __future__ import annotations

import argparse
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

from orchestration.forensics import load_forensics_models
from orchestration.orchestrator import load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".ppm", ".pgm", ".tif"}


def compute_metrics(
    y_true: np.ndarray, y_prob_fake: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Calculate Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC with FAKE as positive class (1)."""
    y_pred = (y_prob_fake >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob_fake))
        pr_auc = float(average_precision_score(y_true, y_prob_fake))
    else:
        roc_auc = 0.0
        pr_auc = 0.0

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Specificity": round(spec, 4),
        "F1": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC": round(pr_auc, 4),
    }


def find_test_samples(test_dir: Path) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    """Recursively scan test_dir for images in class subfolders (e.g. real, fake)."""
    samples: list[tuple[Path, int]] = []
    class_to_idx: dict[str, int] = {}

    subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return samples, class_to_idx

    # Class mapping: fake=0, real=1 (or dynamic based on names)
    # Sort names for consistency
    dir_names = sorted([d.name for d in subdirs])

    # Assign class indices prioritizing fake=0, real=1
    fake_names = {"fake", "deepfake", "synthetic"}
    real_names = {"real", "authentic", "original"}

    for name in dir_names:
        n_lower = name.lower()
        if n_lower in fake_names and "fake" not in class_to_idx:
            class_to_idx[name] = 0
        elif n_lower in real_names and "real" not in class_to_idx:
            class_to_idx[name] = 1

    # Fill remaining
    current_idx = 0
    for name in dir_names:
        if name not in class_to_idx:
            while current_idx in class_to_idx.values():
                current_idx += 1
            class_to_idx[name] = current_idx

    # Scan files
    for subdir in subdirs:
        c_idx = class_to_idx[subdir.name]
        for f in subdir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((f, c_idx))

    return samples, class_to_idx


def run_evaluation(
    config_path: Path = Path("config/inference.yaml"),
    test_dir_override: Path | None = None,
    output_dir: Path = Path("results"),
) -> bool:
    """Run model evaluation if dataset is available."""
    config_path = config_path.resolve()
    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found.", file=sys.stderr)
        return False

    config = load_config(config_path)
    data_cfg = config.get("data", {})

    dataset_name = data_cfg.get("dataset_name", "mini_deepfake_dataset")
    data_root_str = data_cfg.get("dataset_root", data_cfg.get("root", "data/mini_deepfake_dataset"))
    test_split = data_cfg.get("test_split", "test")

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
    print(f"Dataset Name : {dataset_name}")
    print(f"Test Directory: {test_dir}")

    # Check dataset existence
    if not test_dir.exists():
        print("\nMISSING: dataset information")
        print(f"The specified test directory does not exist: '{test_dir}'")
        print("\nPlease provide the evaluation dataset at the configured path or set test_dir in config/inference.yaml.")
        print("Required ImageFolder format:")
        print(f"  {test_dir}/real/*.jpg")
        print(f"  {test_dir}/fake/*.jpg")
        print("==================================================")
        return False

    samples, class_to_idx = find_test_samples(test_dir)

    if not samples:
        print("\nMISSING: dataset information")
        print(f"No valid images found in subdirectories of test folder: '{test_dir}'")
        print("\nNote: Standard evaluation requires test images in both class folders:")
        print(f"  - Real images in: {test_dir}/real/")
        print(f"  - Fake images in: {test_dir}/fake/")
        print("==================================================")
        return False

    # Determine FAKE and REAL indices
    fake_idx = None
    real_idx = None
    for cls_name, idx in class_to_idx.items():
        name_lower = cls_name.lower()
        if "fake" in name_lower or "synthetic" in name_lower or "deepfake" in name_lower:
            fake_idx = idx
        elif "real" in name_lower or "authentic" in name_lower or "original" in name_lower:
            real_idx = idx

    if fake_idx is None:
        fake_idx = 0
        real_idx = 1 if len(class_to_idx) > 1 else 0

    print(f"Detected subdirectories: {list(class_to_idx.keys())}")
    print(f"Mapped FAKE class index -> {fake_idx}, REAL class index -> {real_idx}")

    # Count dataset statistics
    total_images = len(samples)
    fake_images = sum(1 for _, target in samples if target == fake_idx)
    real_images = sum(1 for _, target in samples if target == real_idx)

    print("\n--------------------------------------------------")
    print(f"Total test images: {total_images}")
    print(f"Real images      : {real_images}")
    print(f"Fake images      : {fake_images}")
    print("--------------------------------------------------")

    if real_images == 0 or fake_images == 0:
        print("\n[Notice] Dataset evaluation requires images in BOTH 'real' and 'fake' subfolders.")
        if fake_images == 0:
            print(" -> Currently missing images in: 'data/mini_deepfake_dataset/test/fake/'")
        if real_images == 0:
            print(" -> Currently missing images in: 'data/mini_deepfake_dataset/test/real/'")

    # Load models
    print("\nLoading models for evaluation...")
    bundles = load_forensics_models(config_path)
    if not bundles:
        print("Error: No models loaded for evaluation.", file=sys.stderr)
        return False

    # Collect predictions for all test images
    records: list[dict[str, Any]] = []

    print("\nRunning inference across all models on the test set...")
    for img_path, target in samples:
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to open image '{img_path}': {e}")
            continue

        # Ground truth: 1 for FAKE, 0 for REAL
        gt = 1 if target == fake_idx else 0

        probs_fake: dict[str, float] = {}

        for bundle in bundles:
            tensor = bundle.transform(img_rgb)
            batch = tensor.unsqueeze(0).to(bundle.device)
            with torch.inference_mode():
                logits = bundle.model(batch)
                probs = F.softmax(logits, dim=1)
                p_fake = float(probs[0, fake_idx]) if fake_idx < probs.shape[1] else float(1.0 - probs[0, 1])
                probs_fake[bundle.name] = p_fake

        # Ensemble calculation (matching forensics.py weights)
        model_weights = {
            "faster_vit_2_224": 0.50,
            "efficientnet_b3": 0.35,
            "efficientformerv2_s1": 0.15,
        }
        total_w = 0.0
        weighted_p = 0.0
        for name, p_val in probs_fake.items():
            w = model_weights.get(name, 1.0)
            weighted_p += p_val * w
            total_w += w

        p_ensemble = weighted_p / total_w if total_w > 0 else float(np.mean(list(probs_fake.values())))
        probs_fake["Ensemble"] = p_ensemble

        record = {
            "image_path": str(img_path),
            "ground_truth": gt,
            "efficientnet_b3_prob": probs_fake.get("efficientnet_b3", 0.0),
            "faster_vit_2_224_prob": probs_fake.get("faster_vit_2_224", 0.0),
            "efficientformerv2_s1_prob": probs_fake.get("efficientformerv2_s1", 0.0),
            "ensemble_prob": p_ensemble,
        }
        records.append(record)

    if not records:
        print("Error: No images were successfully evaluated.", file=sys.stderr)
        return False

    df_preds = pd.DataFrame(records)

    # Output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = output_dir / "evaluation_predictions.csv"
    df_preds.to_csv(predictions_csv, index=False)
    print(f"\nSaved per-image predictions to: {predictions_csv}")

    # Compute metrics for each model and ensemble
    y_true = df_preds["ground_truth"].to_numpy(dtype=int)

    rows: list[dict[str, Any]] = []

    evaluation_targets = [
        ("efficientnet_b3", "EfficientNet"),
        ("faster_vit_2_224", "FasterViT"),
        ("efficientformerv2_s1", "EfficientFormerV2-S1"),
        ("Ensemble", "Ensemble"),
    ]

    for key, display_name in evaluation_targets:
        if key == "Ensemble":
            y_prob = df_preds["ensemble_prob"].to_numpy(dtype=float)
        else:
            col_name = f"{key}_prob"
            if col_name in df_preds.columns:
                y_prob = df_preds[col_name].to_numpy(dtype=float)
            else:
                continue

        metrics = compute_metrics(y_true, y_prob)
        metrics["Model"] = display_name
        rows.append(metrics)

    # Reorder columns exactly as required: Model,Accuracy,Precision,Recall,Specificity,F1,ROC_AUC,PR_AUC
    cols = ["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC_AUC", "PR_AUC"]
    df_comparison = pd.DataFrame(rows)[cols]

    comparison_csv = output_dir / "model_comparison.csv"
    df_comparison.to_csv(comparison_csv, index=False)

    print("\n==================================================")
    print("FINAL MODEL COMPARISON TABLE")
    print("==================================================")
    print(df_comparison.to_string(index=False))
    print("==================================================")
    print(f"Saved comparison table to: {comparison_csv}\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detection models & ensemble")
    parser.add_argument("--config", type=Path, default=Path("config/inference.yaml"), help="Path to inference.yaml")
    parser.add_argument("--test-dir", type=Path, help="Override path to test dataset directory")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory for CSV files")
    args = parser.parse_args()

    success = run_evaluation(
        config_path=args.config,
        test_dir_override=args.test_dir,
        output_dir=args.output_dir,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
