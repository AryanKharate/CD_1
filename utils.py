"""
============================================================
UTILITY FUNCTIONS — LULC U-Net Semantic Segmentation
============================================================
Helper functions used across the pipeline:
- NDVI / NDWI / NDBI computation
- Color mask generation
- Visualization helpers
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import CLASS_NAMES, CLASS_COLORS, CLASS_COLORS_NORM, NUM_CLASSES


def compute_ndvi(nir, red):
    """
    Compute Normalized Difference Vegetation Index.
    
    NDVI = (NIR - Red) / (NIR + Red + epsilon)
    
    Args:
        nir: NIR band array (float)
        red: Red band array (float)
    
    Returns:
        NDVI normalized to [0, 1] range
    """
    # Add small epsilon to avoid division by zero
    ndvi = (nir - red) / (nir + red + 1e-6)
    
    # Clip to valid NDVI range [-1, 1], then normalize to [0, 1]
    ndvi = np.clip(ndvi, -1.0, 1.0)
    ndvi_normalized = (ndvi + 1.0) / 2.0  # Maps [-1,1] → [0,1]
    
    return ndvi_normalized


def compute_ndwi(green, nir):
    """
    Compute Normalized Difference Water Index.
    
    NDWI = (Green - NIR) / (Green + NIR + epsilon)
    Highlights water bodies (positive NDWI = water).
    
    Args:
        green: Green band array (float)
        nir:   NIR band array (float)
    
    Returns:
        NDWI normalized to [0, 1] range
    """
    ndwi = (green - nir) / (green + nir + 1e-6)
    ndwi = np.clip(ndwi, -1.0, 1.0)
    return (ndwi + 1.0) / 2.0


def compute_ndbi(swir, nir):
    """
    Compute Normalized Difference Built-up Index.
    
    NDBI = (SWIR - NIR) / (SWIR + NIR + epsilon)
    Highlights built-up/urban areas (positive NDBI = built-up).
    
    Args:
        swir: SWIR band array (float)
        nir:  NIR band array (float)
    
    Returns:
        NDBI normalized to [0, 1] range
    """
    ndbi = (swir - nir) / (swir + nir + 1e-6)
    ndbi = np.clip(ndbi, -1.0, 1.0)
    return (ndbi + 1.0) / 2.0


def class_to_color_mask(class_map):
    """
    Convert a 2D integer class map to an RGB color image.
    
    Args:
        class_map: 2D numpy array of shape (H, W) with integer class labels
    
    Returns:
        RGB image of shape (H, W, 3) with uint8 values
    """
    h, w = class_map.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
        mask = class_map == class_id
        color_mask[mask] = color
    
    return color_mask


def create_legend_patches():
    """
    Create matplotlib legend patches for the LULC classes.
    
    Returns:
        List of matplotlib Patch objects
    """
    patches = []
    for i, name in enumerate(CLASS_NAMES):
        color = CLASS_COLORS_NORM[i]
        patches.append(mpatches.Patch(color=color, label=name))
    return patches


def plot_sample(image_rgb, true_mask, pred_mask=None, title="Sample", save_path=None):
    """
    Plot original image alongside ground truth and optionally predicted mask.
    
    Args:
        image_rgb:  RGB image array, shape (H, W, 3), values in [0,1]
        true_mask:  Ground truth class map, shape (H, W), integer labels
        pred_mask:  Predicted class map (optional), shape (H, W), integer labels
        title:      Plot title string
        save_path:  If provided, save the figure to this path
    """
    ncols = 3 if pred_mask is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    
    # Original RGB image
    axes[0].imshow(np.clip(image_rgb, 0, 1))
    axes[0].set_title("Sentinel-2 RGB")
    axes[0].axis("off")
    
    # Ground truth mask
    true_color = class_to_color_mask(true_mask)
    axes[1].imshow(true_color)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    # Predicted mask (if provided)
    if pred_mask is not None:
        pred_color = class_to_color_mask(pred_mask)
        axes[2].imshow(pred_color)
        axes[2].set_title("Prediction")
        axes[2].axis("off")
    
    # Add legend
    legend_patches = create_legend_patches()
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=NUM_CLASSES,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02)
    )
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved plot to: {save_path}")
    
    plt.close(fig)


def plot_training_history(history_dict, save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history_dict: Dictionary with 'loss', 'val_loss', 'accuracy', 'val_accuracy'
        save_path:    If provided, save the figure to this path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(history_dict['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history_dict['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title("Loss Over Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    # Handle different key names
    acc_key = 'accuracy' if 'accuracy' in history_dict else 'sparse_categorical_accuracy'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history_dict else 'val_sparse_categorical_accuracy'
    
    if acc_key in history_dict:
        ax2.plot(history_dict[acc_key], label='Train Accuracy', linewidth=2)
        ax2.plot(history_dict[val_acc_key], label='Val Accuracy', linewidth=2)
        ax2.set_title("Accuracy Over Epochs", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved training curves to: {save_path}")
    
    plt.close(fig)


def print_banner(text):
    """Print a formatted section banner."""
    width = max(len(text) + 4, 60)
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)
