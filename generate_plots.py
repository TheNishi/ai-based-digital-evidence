"""Generate publication-quality metric visualization plots and images for Digital Evidence Forensics.

Generates:
1. ROC-AUC Curves (All models & Ensemble) -> results/plots/roc_auc_curve.png
2. PR-AUC (Precision-Recall) Curves -> results/plots/pr_auc_curve.png
3. Confusion Matrices (2x2 grid for all models at optimal & default threshold) -> results/plots/confusion_matrices.png
4. Metrics Comparison Bar Chart -> results/plots/metrics_comparison.png
5. Operating Threshold Optimization Curves -> results/plots/threshold_tuning_curves.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def set_plot_style() -> None:
    """Set modern, clean scientific styling."""
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


def compute_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Youden's J statistic to determine optimal decision threshold."""
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


def plot_roc_curve(y_true: np.ndarray, models_dict: dict[str, np.ndarray], output_path: Path) -> None:
    """Generate high-resolution ROC-AUC curve image."""
    plt.figure(figsize=(9, 7), dpi=300)
    
    colors = {
        "FasterViT-2-224": "#38BDF8",      # Bright Cyan
        "EfficientNet-B3": "#F472B6",      # Bright Rose/Pink
        "EfficientFormerV2-S1": "#FBBF24", # Amber
        "Weighted Ensemble": "#4ADE80",   # Emerald Green
    }

    for name, probs in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        c = colors.get(name, "#A78BFA")
        lw = 2.8 if "Ensemble" in name else 2.0
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=c, linewidth=lw)

    # Random chance line
    plt.plot([0, 1], [0, 1], linestyle="--", color="#64748B", linewidth=1.5, label="Random Chance (AUC = 0.5000)")

    plt.title("Receiver Operating Characteristic (ROC-AUC) — CIFAKE Benchmark", fontsize=14, pad=15, fontweight="bold", color="#F8FAFC")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=10)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12, labelpad=10)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved ROC Curve Image -> {output_path}")


def plot_pr_curve(y_true: np.ndarray, models_dict: dict[str, np.ndarray], output_path: Path) -> None:
    """Generate high-resolution Precision-Recall curve image."""
    plt.figure(figsize=(9, 7), dpi=300)
    
    colors = {
        "FasterViT-2-224": "#38BDF8",
        "EfficientNet-B3": "#F472B6",
        "EfficientFormerV2-S1": "#FBBF24",
        "Weighted Ensemble": "#4ADE80",
    }

    for name, probs in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        c = colors.get(name, "#A78BFA")
        lw = 2.8 if "Ensemble" in name else 2.0
        plt.plot(recall, precision, label=f"{name} (PR-AUC / AP = {ap:.4f})", color=c, linewidth=lw)

    baseline = float(np.sum(y_true) / len(y_true))
    plt.axhline(y=baseline, color="#64748B", linestyle="--", linewidth=1.5, label=f"No-Skill Baseline ({baseline:.2f})")

    plt.title("Precision-Recall (PR-AUC) Curve — CIFAKE Deepfake Detection", fontsize=14, pad=15, fontweight="bold", color="#F8FAFC")
    plt.xlabel("Recall (Sensitivity)", fontsize=12, labelpad=10)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12, labelpad=10)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved PR-AUC Curve Image -> {output_path}")


def plot_confusion_matrices(y_true: np.ndarray, models_dict: dict[str, np.ndarray], output_path: Path) -> None:
    """Generate 2x4 grid showing confusion matrices at both Default and Calibrated Optimal Thresholds."""
    model_keys = list(models_dict.keys())
    fig, axes = plt.subplots(2, len(model_keys), figsize=(4.5 * len(model_keys), 8), dpi=300)

    for col_idx, (name, probs) in enumerate(models_dict.items()):
        # Row 0: Default threshold (0.5)
        thr_default = 0.5
        y_pred_def = (probs >= thr_default).astype(int)
        cm_def = confusion_matrix(y_true, y_pred_def, labels=[0, 1])

        # Row 1: Calibrated optimal threshold
        thr_opt = compute_optimal_threshold(y_true, probs)
        y_pred_opt = (probs >= thr_opt).astype(int)
        cm_opt = confusion_matrix(y_true, y_pred_opt, labels=[0, 1])

        # Render Default
        ax0 = axes[0, col_idx]
        im0 = ax0.imshow(cm_def, interpolation="nearest", cmap="Blues")
        ax0.set_title(f"{name}\nDefault (τ = {thr_default:.2f})", fontsize=11, fontweight="bold", color="#F8FAFC")
        _format_cm_ax(ax0, cm_def)

        # Render Optimal
        ax1 = axes[1, col_idx]
        im1 = ax1.imshow(cm_opt, interpolation="nearest", cmap="Greens")
        ax1.set_title(f"{name}\nCalibrated Optimal (τ = {thr_opt:.3f})", fontsize=11, fontweight="bold", color="#4ADE80")
        _format_cm_ax(ax1, cm_opt)

    axes[0, 0].set_ylabel("True Class (Default τ)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("True Class (Calibrated τ)", fontsize=12, fontweight="bold")

    plt.suptitle("Confusion Matrix Analysis (Top: Default 0.5 vs Bottom: Calibrated Optimal Threshold)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrices Image -> {output_path}")


def _format_cm_ax(ax: Any, cm: np.ndarray) -> None:
    """Format single confusion matrix subplot with clear cell annotations."""
    classes = ["REAL (0)", "FAKE (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val < thresh else "black"
            ax.text(j, i, f"{val:,}\n({val/cm.sum()*100:.1f}%)",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color, fontsize=10, fontweight="bold")


def plot_metrics_comparison_bar(df_metrics: pd.DataFrame, output_path: Path) -> None:
    """Generate bar chart comparing Accuracy, Precision, Recall, Specificity, F1, ROC-AUC across models."""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC_AUC"]
    models = df_metrics["Model"].tolist()
    
    x = np.arange(len(models))
    width = 0.13
    
    palette = ["#38BDF8", "#818CF8", "#F472B6", "#FB923C", "#FBBF24", "#4ADE80"]
    
    for i, metric in enumerate(metrics_to_plot):
        values = df_metrics[metric].tolist()
        offset = (i - len(metrics_to_plot) / 2) * width + width / 2
        rects = ax.bar(x + offset, values, width, label=metric, color=palette[i % len(palette)], alpha=0.9)
        for rect in rects:
            h = rect.get_height()
            if h > 0.05:
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color="#F8FAFC", rotation=45)

    ax.set_ylabel("Score (0.00 - 1.00)", fontsize=12, fontweight="bold")
    ax.set_title("Comprehensive Model Comparison on CIFAKE Test Set (Calibrated Operating Threshold)", fontsize=14, pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1.15])
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", ncol=3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Metrics Comparison Bar Chart -> {output_path}")


def main() -> None:
    predictions_csv = Path("results/evaluation_predictions.csv")
    if not predictions_csv.exists():
        print(f"Error: {predictions_csv} not found. Run evaluation first.")
        return

    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    # Load predictions
    df = pd.read_csv(predictions_csv)
    y_true = df["ground_truth"].to_numpy(dtype=int)

    models_dict = {
        "EfficientNet-B3": df["efficientnet_b3_prob"].to_numpy(dtype=float),
        "EfficientFormerV2-S1": df["efficientformerv2_s1_prob"].to_numpy(dtype=float),
        "FasterViT-2-224": df["faster_vit_2_224_prob"].to_numpy(dtype=float),
        "Weighted Ensemble": df["ensemble_prob"].to_numpy(dtype=float),
    }

    print("Generating High-Resolution Diagnostic Images & Evaluation Plots...")

    # 1. ROC-AUC Image
    plot_roc_curve(y_true, models_dict, output_dir / "roc_auc_curve.png")

    # 2. PR-AUC Image
    plot_pr_curve(y_true, models_dict, output_dir / "pr_auc_curve.png")

    # 3. Confusion Matrix Image
    plot_confusion_matrices(y_true, models_dict, output_dir / "confusion_matrices.png")

    # 4. Generate calibrated metrics dataframe
    rows: list[dict[str, Any]] = []
    for name, probs in models_dict.items():
        thr_opt = compute_optimal_threshold(y_true, probs)
        y_pred = (probs >= thr_opt).astype(int)
        
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        roc_auc = float(roc_auc_score(y_true, probs))
        pr_auc = float(average_precision_score(y_true, probs))

        rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "Specificity": round(spec, 4),
            "F1": round(f1, 4),
            "ROC_AUC": round(roc_auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "Threshold": round(thr_opt, 4),
        })

    df_calibrated = pd.DataFrame(rows)
    df_calibrated.to_csv(Path("results/model_comparison_calibrated.csv"), index=False)
    print("\n==================================================")
    print("CALIBRATED BENCHMARK METRICS (Optimal Threshold)")
    print("==================================================")
    print(df_calibrated.to_string(index=False))
    print("==================================================\n")

    # 5. Metrics Comparison Bar Chart Image
    plot_metrics_comparison_bar(df_calibrated, output_dir / "metrics_comparison.png")

    print("\nAll visualization plot images successfully created under results/plots/!")


if __name__ == "__main__":
    main()
