"""
============================================================
MAIN ORCHESTRATOR — LULC U-Net Semantic Segmentation
============================================================
Runs the complete pipeline step-by-step:

  1. Download ESA WorldCover labels (via Google Earth Engine)
  2. Preprocess Sentinel-2 images → 256×256 patches
  3. Build tf.data datasets (train/val split)
  4. Train U-Net model
  5. Predict on validation samples
  6. Evaluate with IoU and accuracy metrics

Usage:
  python main.py                    # Run full pipeline
  python main.py --download         # Only download data
  python main.py --preprocess       # Only preprocess
  python main.py --train            # Only train
  python main.py --predict          # Only predict
  python main.py --evaluate         # Only evaluate
  python main.py --visualize-data   # Visualize data before training
============================================================
"""

import os
import sys
import argparse
import numpy as np

# Suppress TensorFlow info logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import (
    SENTINEL_2020, SENTINEL_2025, WORLDCOVER_FILE,
    PATCHES_DIR, MODEL_DIR, PREDICTIONS_DIR, PLOTS_DIR,
    CLASS_NAMES, NUM_CLASSES
)
from utils import print_banner


def visualize_data_samples():
    """
    Visualize a few data samples BEFORE training to verify:
      - Images look correct (RGB composite)
      - Labels are properly mapped
      - NDVI makes sense
      - No alignment issues
    
    This is a CRITICAL sanity check!
    """
    print_banner("DATA INSPECTION — Visual Sanity Check")
    
    from preprocessing import load_all_patches
    from utils import plot_sample, class_to_color_mask
    import matplotlib.pyplot as plt
    
    images, labels = load_all_patches()
    
    # Show 4 random samples
    rng = np.random.RandomState(42)
    indices = rng.choice(len(images), min(4, len(images)), replace=False)
    
    for i, idx in enumerate(indices):
        img = images[idx]
        lbl = labels[idx]
        
        print(f"\n  Sample {i+1} (patch #{idx}):")
        print(f"    Image shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"    Label shape: {lbl.shape}, unique classes: {np.unique(lbl)}")
        
        # Show class distribution in this patch
        for c in range(NUM_CLASSES):
            pct = np.sum(lbl == c) / lbl.size * 100
            if pct > 0:
                print(f"      {CLASS_NAMES[c]}: {pct:.1f}%")
        
        # Save plot
        rgb = img[:, :, :3][:, :, ::-1]  # BGR to RGB from the 6-channel input
        save_path = os.path.join(PLOTS_DIR, f"data_sample_{i}.png")
        plot_sample(rgb, lbl, title=f"Data Sample {i+1} (patch #{idx})",
                   save_path=save_path)
    
    # NDVI channel visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, idx in enumerate(indices):
        # RGB
        rgb_disp = images[idx][:, :, :3][:, :, ::-1]
        axes[0, i].imshow(np.clip(rgb_disp, 0, 1))
        axes[0, i].set_title(f"RGB #{idx}", fontsize=9)
        axes[0, i].axis("off")
        
        # NDVI channel (Index 5)
        axes[1, i].imshow(images[idx][:, :, 5], cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, i].set_title(f"NDVI #{idx}", fontsize=9)
        axes[1, i].axis("off")
    
    fig.suptitle("Data Verification — RGB vs NDVI", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ndvi_path = os.path.join(PLOTS_DIR, "data_ndvi_check.png")
    fig.savefig(ndvi_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n  ✓ Saved data inspection plots to: {PLOTS_DIR}")
    print("  ✓ VISUAL CHECK: Review plots before training!")


def run_full_pipeline():
    """Run the complete end-to-end pipeline."""
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   LULC Classification — U-Net Semantic Segmentation     ║")
    print("║   Sentinel-2 Imagery | ESA WorldCover Labels            ║")
    print("║   Hyderabad, India | 5 Classes                          ║")
    print("╚" + "═" * 58 + "╝")
    
    # ── Step 1: Download ESA WorldCover ────────────────────────
    from data_download import run_download
    run_download()
    
    # ── Step 2: Preprocess & create patches ────────────────────
    from preprocessing import run_preprocessing
    run_preprocessing()
    
    # ── Step 2.5: Visual inspection ────────────────────────────
    visualize_data_samples()
    
    # ── Step 3+4: Train model ──────────────────────────────────
    from train import train_model
    model, history = train_model()
    
    # ── Step 5: Predict & visualize ────────────────────────────
    from predict import visualize_predictions, predict_full_image
    visualize_predictions(model, num_samples=8)
    
    # Predict on full 2020 image
    predict_full_image(model, SENTINEL_2020)
    
    # Predict on full 2025 image if available
    if os.path.exists(SENTINEL_2025):
        predict_full_image(model, SENTINEL_2025)
        
        # ── Step 5.5: Change Detection ─────────────────────────────
        from change_detection import generate_change_map
        y1_pred = os.path.join(PREDICTIONS_DIR, f"predicted_lulc_{os.path.basename(SENTINEL_2020)}")
        y2_pred = os.path.join(PREDICTIONS_DIR, f"predicted_lulc_{os.path.basename(SENTINEL_2025)}")
        if os.path.exists(y1_pred) and os.path.exists(y2_pred):
            generate_change_map(y1_pred, y2_pred)
    
    # ── Step 6: Evaluate ───────────────────────────────────────
    from evaluate import evaluate_model
    metrics = evaluate_model()
    
    # ── Final Summary ──────────────────────────────────────────
    print_banner("PIPELINE COMPLETE!")
    print(f"\n  Results saved to:")
    print(f"    Models:      {MODEL_DIR}")
    print(f"    Plots:       {PLOTS_DIR}")
    print(f"    Predictions: {PREDICTIONS_DIR}")
    print(f"\n  Final Metrics:")
    print(f"    Pixel Accuracy: {metrics['pixel_accuracy']*100:.2f}%")
    print(f"    Mean IoU:       {metrics['mean_iou']*100:.2f}%")
    print()


def main():
    """Parse command-line arguments and run selected steps."""
    parser = argparse.ArgumentParser(
        description="LULC Classification Pipeline — U-Net Semantic Segmentation"
    )
    parser.add_argument('--download', action='store_true',
                       help='Download ESA WorldCover labels only')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run preprocessing only')
    parser.add_argument('--train', action='store_true',
                       help='Train the model only')
    parser.add_argument('--predict', action='store_true',
                       help='Run predictions only')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation only')
    parser.add_argument('--change-detection', action='store_true',
                       help='Run change detection only')
    parser.add_argument('--visualize-data', action='store_true',
                       help='Visualize data samples before training')
    
    args = parser.parse_args()
    
    # If no specific step is selected, run full pipeline
    if not any(vars(args).values()):
        run_full_pipeline()
        return
    
    # Run selected steps
    if args.download:
        from data_download import run_download
        run_download()
    
    if args.preprocess:
        from preprocessing import run_preprocessing
        run_preprocessing()
    
    if args.visualize_data:
        visualize_data_samples()
    
    if args.train:
        from train import train_model
        train_model()
    
    if args.predict:
        from predict import load_trained_model, visualize_predictions, predict_full_image
        from config import SENTINEL_2020, SENTINEL_2025
        model = load_trained_model()
        visualize_predictions(model, num_samples=8)
        if os.path.exists(SENTINEL_2020):
            predict_full_image(model, SENTINEL_2020)
        if os.path.exists(SENTINEL_2025):
            predict_full_image(model, SENTINEL_2025)
    
    if args.evaluate:
        from evaluate import evaluate_model
        evaluate_model()

    if args.change_detection:
        from change_detection import generate_change_map
        from config import SENTINEL_2020, SENTINEL_2025
        y1_pred = os.path.join(PREDICTIONS_DIR, f"predicted_lulc_{os.path.basename(SENTINEL_2020)}")
        y2_pred = os.path.join(PREDICTIONS_DIR, f"predicted_lulc_{os.path.basename(SENTINEL_2025)}")
        if os.path.exists(y1_pred) and os.path.exists(y2_pred):
            generate_change_map(y1_pred, y2_pred)
        else:
            print("  ⚠ Predictions not found. Run --predict first.")


if __name__ == "__main__":
    main()
