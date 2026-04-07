"""
============================================================
DATASET PIPELINE — LULC U-Net Semantic Segmentation
============================================================
Creates TensorFlow tf.data.Dataset pipelines for training
and validation with:
  - 80/20 train/val split
  - Data augmentation (random flips, rotations)
  - Efficient batching and prefetching
  - Class weight computation for imbalanced data
============================================================
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import (
    BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED,
    PATCH_SIZE, INPUT_CHANNELS, NUM_CLASSES
)
from preprocessing import load_all_patches
from utils import print_banner


def compute_class_weights(labels):
    """
    Compute inverse-frequency class weights to handle class imbalance.
    
    Classes with fewer pixels get higher weights so the model
    pays more attention to rare classes (e.g., water, barren).
    
    Args:
        labels: numpy array of integer labels, shape (N, H, W)
    
    Returns:
        Dictionary mapping class_id → weight (float)
    """
    from config import CLASS_NAMES
    
    # Count pixels per class
    flat_labels = labels.flatten()
    total_pixels = len(flat_labels)
    
    class_counts = {}
    for i in range(NUM_CLASSES):
        count = np.sum(flat_labels == i)
        class_counts[i] = count
    
    # Compute weights: inverse frequency, normalized
    # weight_i = total_pixels / (num_classes * count_i)
    weights = {}
    for i in range(NUM_CLASSES):
        if class_counts[i] > 0:
            weights[i] = total_pixels / (NUM_CLASSES * class_counts[i])
        else:
            weights[i] = 1.0  # Default weight for missing classes
    
    # Cap extreme weights to avoid instability
    max_weight = 10.0
    for i in weights:
        weights[i] = min(weights[i], max_weight)
    
    print("  Class weights (inverse frequency):")
    for i, name in enumerate(CLASS_NAMES):
        count = class_counts[i]
        pct = count / total_pixels * 100
        print(f"    {i} ({name:>15s}): {count:>10,d} px ({pct:>5.1f}%) → weight = {weights[i]:.3f}")
    
    return weights


def augment(image, label):
    """
    Apply random data augmentation to image-label pairs.
    
    Augmentations applied:
      - Random horizontal flip
      - Random vertical flip
      - Random 90° rotation (0, 1, 2, or 3 times)
    
    IMPORTANT: The same transform must be applied to BOTH
    the image and label to maintain pixel alignment!
    
    Args:
        image: Tensor of shape (H, W, 4)
        label: Tensor of shape (H, W), integer labels
    
    Returns:
        Augmented (image, label) tensors
    """
    # Add channel dim to label for consistent operations
    label = tf.expand_dims(label, axis=-1)  # (H, W, 1)
    
    # Concatenate for synchronized transforms
    combined = tf.concat([image, tf.cast(label, tf.float32)], axis=-1)  # (H, W, 5)
    
    # Random horizontal flip
    combined = tf.image.random_flip_left_right(combined)
    
    # Random vertical flip
    combined = tf.image.random_flip_up_down(combined)
    
    # Random 90° rotation (k times)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    combined = tf.image.rot90(combined, k)
    
    # Split back into image and label
    image = combined[:, :, :INPUT_CHANNELS]
    label = tf.cast(combined[:, :, INPUT_CHANNELS], tf.int32)
    
    return image, label


def create_datasets():
    """
    Create train and validation tf.data.Dataset pipelines.
    
    Returns:
        train_ds:      tf.data.Dataset for training (augmented)
        val_ds:        tf.data.Dataset for validation (no augmentation)
        class_weights: Dictionary of class weights
        n_train:       Number of training samples
        n_val:         Number of validation samples
    """
    print_banner("STEP 3: Building Dataset Pipeline")
    
    # Load all patches
    images, labels = load_all_patches()
    
    # Verify shapes
    assert images.shape[1:] == (PATCH_SIZE, PATCH_SIZE, INPUT_CHANNELS), \
        f"Expected image shape (*, {PATCH_SIZE}, {PATCH_SIZE}, {INPUT_CHANNELS}), got {images.shape}"
    assert labels.shape[1:] == (PATCH_SIZE, PATCH_SIZE), \
        f"Expected label shape (*, {PATCH_SIZE}, {PATCH_SIZE}), got {labels.shape}"
    
    # Verify label range
    assert labels.max() < NUM_CLASSES, \
        f"Label values must be 0-{NUM_CLASSES-1}, found max={labels.max()}"
    
    # Compute class weights before splitting
    class_weights = compute_class_weights(labels)
    
    # 80/20 train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    n_train = len(X_train)
    n_val = len(X_val)
    print(f"\n  Train: {n_train} patches")
    print(f"  Val:   {n_val} patches")
    
    # ── Training Dataset ──────────────────────────────────────
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train.astype(np.float32), y_train.astype(np.int32))
    )
    train_ds = (
        train_ds
        .shuffle(buffer_size=min(1000, n_train), seed=RANDOM_SEED)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # ── Validation Dataset (NO augmentation) ──────────────────
    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val.astype(np.float32), y_val.astype(np.int32))
    )
    val_ds = (
        val_ds
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Train batches per epoch: {n_train // BATCH_SIZE}")
    print(f"  Val batches per epoch: {n_val // BATCH_SIZE}")
    print("  ✓ Dataset pipeline ready!")
    
    return train_ds, val_ds, class_weights, n_train, n_val


if __name__ == "__main__":
    train_ds, val_ds, weights, n_train, n_val = create_datasets()
    
    # Quick sanity check: inspect one batch
    for images, labels in train_ds.take(1):
        print(f"\n  Sample batch:")
        print(f"    Images: {images.shape}, dtype={images.dtype}")
        print(f"    Labels: {labels.shape}, dtype={labels.dtype}")
        print(f"    Label range: [{labels.numpy().min()}, {labels.numpy().max()}]")
