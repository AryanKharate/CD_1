"""
============================================================
PREDICTION & VISUALIZATION — LULC U-Net Segmentation
============================================================
Loads the trained model and generates predictions with:
  - Argmax class extraction from softmax outputs
  - Color-coded segmentation maps
  - Side-by-side comparison plots
  - Batch prediction on validation set
============================================================
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from config import (
    MODEL_DIR, PREDICTIONS_DIR, PLOTS_DIR,
    PATCH_SIZE, INPUT_CHANNELS, NUM_CLASSES, CLASS_NAMES,
    VALIDATION_SPLIT, RANDOM_SEED, BATCH_SIZE
)
from preprocessing import load_all_patches
from utils import (
    print_banner, class_to_color_mask,
    create_legend_patches, plot_sample
)
from sklearn.model_selection import train_test_split


def load_trained_model():
    """
    Load the best trained U-Net model from disk.
    
    Returns:
        Loaded Keras model
    """
    # Try loading best model first, then final
    best_path = os.path.join(MODEL_DIR, "unet_lulc_best.keras")
    final_path = os.path.join(MODEL_DIR, "unet_lulc_final.keras")
    
    if os.path.exists(best_path):
        print(f"  → Loading best model: {best_path}")
        # Need custom objects for combined loss
        model = tf.keras.models.load_model(best_path, compile=False)
    elif os.path.exists(final_path):
        print(f"  → Loading final model: {final_path}")
        model = tf.keras.models.load_model(final_path, compile=False)
    else:
        raise FileNotFoundError(
            f"No trained model found! Expected at:\n"
            f"  {best_path}\n  {final_path}\n"
            f"Run train.py first."
        )
    
    print(f"  ✓ Model loaded successfully")
    return model


def predict_batch(model, images):
    """
    Run model prediction on a batch of images.
    
    Args:
        model:  Trained Keras model
        images: numpy array, shape (N, 256, 256, 4)
    
    Returns:
        predictions: numpy array, shape (N, 256, 256), integer class labels
        probabilities: numpy array, shape (N, 256, 256, 5), class probabilities
    """
    # model.predict() returns softmax probabilities
    probabilities = model.predict(images, batch_size=BATCH_SIZE, verbose=0)
    
    # Convert probabilities to class labels via argmax
    predictions = np.argmax(probabilities, axis=-1)
    
    return predictions, probabilities


def visualize_predictions(model, num_samples=8):
    """
    Generate and visualize predictions on validation set samples.
    
    Creates side-by-side plots:
      [Sentinel-2 RGB] | [Ground Truth] | [Prediction]
    
    Args:
        model:       Trained Keras model
        num_samples: Number of samples to visualize
    """
    print_banner("STEP 5: Prediction & Visualization")
    
    # Load data and split (same split as training)
    images, labels = load_all_patches()
    _, X_val, _, y_val = train_test_split(
        images, labels,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    # Limit samples to available data
    num_samples = min(num_samples, len(X_val))
    
    # Select random samples
    rng = np.random.RandomState(RANDOM_SEED)
    indices = rng.choice(len(X_val), num_samples, replace=False)
    
    print(f"  → Predicting on {num_samples} validation samples...")
    
    # Run predictions
    sample_images = X_val[indices]
    sample_labels = y_val[indices]
    predictions, probabilities = predict_batch(model, sample_images)
    
    # Generate individual comparison plots
    for i, idx in enumerate(indices):
        # Extract RGB channels from the 4-channel input [R, G, B, NDVI]
        rgb = sample_images[i][:, :, :3]  # R, G, B channels
        
        # Create side-by-side plot
        save_path = os.path.join(PREDICTIONS_DIR, f"prediction_{i:03d}.png")
        plot_sample(
            image_rgb=rgb,
            true_mask=sample_labels[i],
            pred_mask=predictions[i],
            title=f"Validation Sample {i+1}",
            save_path=save_path
        )
    
    # Create a grid overview of all predictions
    _create_prediction_grid(
        sample_images, sample_labels, predictions, num_samples
    )
    
    print(f"\n  ✓ Saved {num_samples} prediction plots to: {PREDICTIONS_DIR}")


def _create_prediction_grid(images, labels, predictions, num_samples):
    """
    Create a grid showing multiple predictions at once.
    
    Args:
        images:      Input images, shape (N, H, W, 4)
        labels:      Ground truth, shape (N, H, W)
        predictions: Predicted classes, shape (N, H, W)
        num_samples: Number of samples to show
    """
    rows = min(num_samples, 8)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    
    if rows == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(rows):
        # BGR to RGB image
        rgb = images[i][:, :, :3][:, :, ::-1]
        axes[i, 0].imshow(np.clip(rgb, 0, 1))
        axes[i, 0].set_title(f"Sentinel-2 RGB #{i+1}", fontsize=10)
        axes[i, 0].axis("off")
        
        # Ground truth
        true_color = class_to_color_mask(labels[i])
        axes[i, 1].imshow(true_color)
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        axes[i, 1].axis("off")
        
        # Prediction
        pred_color = class_to_color_mask(predictions[i])
        axes[i, 2].imshow(pred_color)
        axes[i, 2].set_title("Prediction", fontsize=10)
        axes[i, 2].axis("off")
    
    # Add legend at the bottom
    legend_patches = create_legend_patches()
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=NUM_CLASSES,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01)
    )
    
    fig.suptitle("LULC Predictions — U-Net Semantic Segmentation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    grid_path = os.path.join(PREDICTIONS_DIR, "prediction_grid.png")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved prediction grid to: {grid_path}")


def recover_bridges_spectrally(full_prediction, input_stack):
    """
    Post-processing rule to rescue bridges misclassified as water.
    Uses SWIR, NIR, and Green bands to detect impervious materials.
    
    Args:
        full_prediction: 2D array of class labels (0-4)
        input_stack: 3D array of bands [B, G, R, NIR, SWIR, NDVI]
    
    Returns:
        Refined full_prediction
    """
    # [0:B, 1:G, 2:R, 3:NIR, 4:SWIR, 5:NDVI]
    green = input_stack[..., 1]
    nir   = input_stack[..., 3]
    swir  = input_stack[..., 4]

    # NDBI = (SWIR - NIR) / (SWIR + NIR) -> Higher for concrete/asphalt
    ndbi = (swir - nir) / (swir + nir + 1e-6)
    
    # NDWI = (Green - NIR) / (Green + NIR) -> Higher for deep water
    ndwi = (green - nir) / (green + nir + 1e-6)

    # RESCUE LOGIC:
    # 1. Pixel is currently classified as Water (4)
    # 2. SWIR is high (> 0.05) -> Water is normally < 0.04
    # 3. NDBI is positive -> Spectral signature of a building/bridge
    # 4. NDWI is not extremely high (< 0.3) -> Not pure deep water
    bridge_mask = (
        (full_prediction == 4) &
        (swir > 0.05) &
        (ndbi > 0.0) &
        (ndwi < 0.3)
    )

    n_recovered = np.sum(bridge_mask)
    if n_recovered > 0:
        print(f"  → Spectrally recovered {n_recovered:,d} bridge/built-up pixels from water!")
        full_prediction[bridge_mask] = 2 # Set to Built-up (Red)
        
    return full_prediction


def predict_full_image(model, sentinel_filepath):
    """
    Run prediction on a full Sentinel-2 image using sliding window.
    
    Processes the full image patch-by-patch and stitches the
    predictions back together into a complete segmentation map.
    
    Args:
        model:             Trained Keras model
        sentinel_filepath: Path to Sentinel-2 GeoTIFF
    
    Returns:
        full_prediction: 2D array, shape (H, W), integer class labels
    """
    from preprocessing import load_sentinel_bands, build_input_stack
    
    print(f"  → Predicting full image: {os.path.basename(sentinel_filepath)}")
    
    # Load and preprocess
    bands = load_sentinel_bands(sentinel_filepath)
    input_stack = build_input_stack(bands)
    
    import scipy.ndimage

    h, w = input_stack.shape[:2]
    stride = PATCH_SIZE // 2
    full_probs = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    # Slide across the image in patches via overlapping window
    for y in range(0, h - PATCH_SIZE + 1, stride):
        for x in range(0, w - PATCH_SIZE + 1, stride):
            patch = input_stack[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]
            
            # Skip mostly-empty patches
            if np.sum(patch[:, :, 0] > 0) < (PATCH_SIZE * PATCH_SIZE * 0.5):
                continue
            
            # Predict
            patch_input = np.expand_dims(patch, axis=0)
            probs = model.predict(patch_input, verbose=0)
            
            # Accumulate probabilities to smooth boundaries
            full_probs[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :] += probs[0]
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1.0
    
    print(f"    Predicted area: {np.sum(count_map > 0):,d} / {h*w:,d} pixels")
    
    # Normalize accumulated probabilities to [0, 1] for thresholding
    probs_sum = np.sum(full_probs, axis=-1, keepdims=True)
    probs_sum[probs_sum == 0] = 1.0
    normalized_probs = full_probs / probs_sum
    
    # Take argmax
    full_prediction = np.argmax(normalized_probs, axis=-1).astype(np.uint8)
    
    # ── SPECTRAL BRIDGE RECOVERY ──
    # Rescue bridges/roads misclassified as water using physical light properties
    full_prediction = recover_bridges_spectrally(full_prediction, input_stack)
    
    # Mask out areas where no predictions occurred
    full_prediction[count_map == 0] = 0
    
    print("  → Bypassing median filter to preserve thin linear structures (bridges) perfectly!")
    # full_prediction = scipy.ndimage.median_filter(full_prediction, size=3)
    
    # Visualize full prediction
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # BGR to RGB composite
    rgb = np.stack([input_stack[:,:,2], input_stack[:,:,1], input_stack[:,:,0]], axis=-1)
    axes[0].imshow(np.clip(rgb, 0, 1))
    axes[0].set_title(f"Sentinel-2 RGB — {os.path.basename(sentinel_filepath)}")
    axes[0].axis("off")
    
    # Prediction
    pred_color = class_to_color_mask(full_prediction)
    axes[1].imshow(pred_color)
    axes[1].set_title("LULC Prediction")
    axes[1].axis("off")
    
    legend_patches = create_legend_patches()
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=NUM_CLASSES, fontsize=11)
    
    plt.tight_layout()
    save_path = os.path.join(PREDICTIONS_DIR,
                              f"full_{os.path.basename(sentinel_filepath).replace('.tif', '.png')}")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved full prediction to: {save_path}")
    import rasterio
    tif_save_path = os.path.join(PREDICTIONS_DIR, f"predicted_lulc_{os.path.basename(sentinel_filepath)}")
    
    with rasterio.open(sentinel_filepath) as src:
        prof = src.profile
        prof.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=255,
            compress='lzw'
        )
        with rasterio.open(tif_save_path, 'w', **prof) as dst:
            dst.write(full_prediction, 1)
            from config import CLASS_COLORS
            dst.write_colormap(1, CLASS_COLORS)
            
    print(f"  → Saved GeoTIFF prediction to: {tif_save_path}")
    
    return full_prediction


if __name__ == "__main__":
    model = load_trained_model()
    visualize_predictions(model, num_samples=8)
