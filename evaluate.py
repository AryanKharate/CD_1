"""
============================================================
EVALUATION — LULC U-Net Semantic Segmentation
============================================================
Computes segmentation quality metrics:
  - Overall Pixel Accuracy
  - Per-class IoU (Intersection over Union)
  - Mean IoU
  - Confusion Matrix with heatmap visualization
============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from config import (
    NUM_CLASSES, CLASS_NAMES, PLOTS_DIR,
    VALIDATION_SPLIT, RANDOM_SEED, BATCH_SIZE, MODEL_DIR
)
from preprocessing import load_all_patches
from utils import print_banner


def pixel_accuracy(y_true, y_pred):
    """
    Compute overall pixel accuracy.
    
    Pixel Accuracy = (correctly classified pixels) / (total pixels)
    
    Args:
        y_true: Ground truth labels, shape (N, H, W)
        y_pred: Predicted labels, shape (N, H, W)
    
    Returns:
        Float accuracy value in [0, 1]
    """
    correct = np.sum(y_true == y_pred)
    total = y_true.size
    return correct / total


def compute_iou(y_true, y_pred, num_classes=NUM_CLASSES):
    """
    Compute per-class Intersection over Union (IoU / Jaccard Index).
    
    IoU_c = TP_c / (TP_c + FP_c + FN_c)
    
    This is the standard metric for semantic segmentation quality.
    IoU = 1.0 means perfect segmentation, IoU = 0.0 means no overlap.
    
    Args:
        y_true:      Ground truth labels, flat array
        y_pred:      Predicted labels, flat array
        num_classes: Number of classes
    
    Returns:
        per_class_iou: Dictionary of class_id → IoU value
        mean_iou:      Average IoU across all classes
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    per_class_iou = {}
    
    for c in range(num_classes):
        # True positives: correctly predicted as class c
        tp = np.sum((y_pred_flat == c) & (y_true_flat == c))
        
        # False positives: incorrectly predicted as class c
        fp = np.sum((y_pred_flat == c) & (y_true_flat != c))
        
        # False negatives: class c pixels missed by prediction
        fn = np.sum((y_pred_flat != c) & (y_true_flat == c))
        
        # IoU = TP / (TP + FP + FN)
        denominator = tp + fp + fn
        if denominator == 0:
            iou = 0.0  # Class not present in ground truth
        else:
            iou = tp / denominator
        
        per_class_iou[c] = iou
    
    # Mean IoU (average across all classes, including zero)
    mean_iou = np.mean(list(per_class_iou.values()))
    
    return per_class_iou, mean_iou


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Generate and plot a confusion matrix heatmap.
    
    Args:
        y_true:    Ground truth labels, flat array
        y_pred:    Predicted labels, flat array
        save_path: If provided, save the plot to this path
    """
    cm = confusion_matrix(
        y_true.flatten(), y_pred.flatten(),
        labels=list(range(NUM_CLASSES))
    )
    
    # Normalize by row (true class) for better visualization
    cm_normalized = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax1
    )
    ax1.set_title("Confusion Matrix (Pixel Counts)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Class")
    ax1.set_ylabel("True Class")
    
    # Normalized (percentage)
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax2
    )
    ax2.set_title("Confusion Matrix (Normalized)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Predicted Class")
    ax2.set_ylabel("True Class")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved confusion matrix to: {save_path}")
    
    plt.close(fig)


def evaluate_model():
    """
    Full evaluation pipeline:
      1. Load model and validation data
      2. Run predictions
      3. Compute pixel accuracy
      4. Compute per-class and mean IoU
      5. Generate confusion matrix
      6. Print formatted report
    
    Returns:
        Dictionary with all metrics
    """
    print_banner("STEP 6: Model Evaluation")
    
    # Load model
    import tensorflow as tf
    best_path = os.path.join(MODEL_DIR, "unet_lulc_best.keras")
    final_path = os.path.join(MODEL_DIR, "unet_lulc_final.keras")
    
    model_path = best_path if os.path.exists(best_path) else final_path
    print(f"  → Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Load data and get validation split (same as training)
    images, labels = load_all_patches()
    _, X_val, _, y_val = train_test_split(
        images, labels,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    print(f"  → Evaluating on {len(X_val)} validation patches...")
    
    # Run predictions
    probabilities = model.predict(X_val, batch_size=BATCH_SIZE, verbose=1)
    predictions = np.argmax(probabilities, axis=-1)
    
    # ── Pixel Accuracy ────────────────────────────────────────
    acc = pixel_accuracy(y_val, predictions)
    
    # ── IoU ───────────────────────────────────────────────────
    per_class_iou, mean_iou = compute_iou(y_val, predictions)
    
    # ── Print Report ──────────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════════════╗")
    print("  ║       LULC Segmentation — Evaluation Report      ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  Overall Pixel Accuracy:  {acc*100:>6.2f}%               ║")
    print(f"  ║  Mean IoU:                {mean_iou*100:>6.2f}%               ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print("  ║  Per-Class IoU:                                  ║")
    
    for i, name in enumerate(CLASS_NAMES):
        iou = per_class_iou[i]
        bar = "█" * int(iou * 20) + "░" * (20 - int(iou * 20))
        print(f"  ║    {name:>15s}: {iou*100:>6.2f}%  {bar}  ║")
    
    print("  ╚══════════════════════════════════════════════════╝")
    
    # ── Confusion Matrix ──────────────────────────────────────
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_val, predictions, save_path=cm_path)
    
    # ── Per-class precision and recall ────────────────────────
    print("\n  Per-class breakdown:")
    print(f"  {'Class':>15s} | {'IoU':>7s} | {'Pixels (GT)':>12s} | {'Pixels (Pred)':>13s}")
    print(f"  {'-'*15}-+-{'-'*7}-+-{'-'*12}-+-{'-'*13}")
    
    for i, name in enumerate(CLASS_NAMES):
        gt_count = np.sum(y_val == i)
        pred_count = np.sum(predictions == i)
        iou = per_class_iou[i]
        print(f"  {name:>15s} | {iou*100:>6.2f}% | {gt_count:>11,d} | {pred_count:>12,d}")
    
    # Save metrics
    metrics = {
        'pixel_accuracy': acc,
        'mean_iou': mean_iou,
        'per_class_iou': per_class_iou,
        'num_val_samples': len(X_val)
    }
    
    print(f"\n  ✓ Evaluation complete!")
    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()
