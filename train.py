"""
============================================================
TRAINING — LULC U-Net Semantic Segmentation
============================================================
Handles model training with:
  - Combined Sparse CrossEntropy + Dice Loss
  - Class weights for imbalanced data
  - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLR
  - Training history logging and visualization
============================================================
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from config import (
    EPOCHS, LEARNING_RATE, BATCH_SIZE,
    EARLY_STOP_PATIENCE, LR_REDUCE_PATIENCE, LR_REDUCE_FACTOR,
    MODEL_DIR, PLOTS_DIR, NUM_CLASSES,
    USE_DICE_LOSS, DICE_LOSS_WEIGHT, CE_LOSS_WEIGHT,
    USE_CLASS_WEIGHTS
)
from model import build_unet
from dataset import create_datasets
from utils import print_banner, plot_training_history


# ──────────────────────────────────────────────────────────────
# CUSTOM LOSS FUNCTIONS
# ──────────────────────────────────────────────────────────────

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Compute Dice Loss for multi-class segmentation.
    
    Dice Loss = 1 - (2 * intersection + smooth) / (union + smooth)
    
    This loss directly optimizes the IoU-like metric and helps
    when classes are imbalanced (unlike cross-entropy alone).
    
    Args:
        y_true: Ground truth, shape (batch, H, W), integer labels
        y_pred: Predictions, shape (batch, H, W, num_classes), probabilities
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Scalar dice loss value
    """
    # One-hot encode the ground truth
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), NUM_CLASSES)
    # y_true_onehot shape: (batch, H, W, num_classes)
    
    # Flatten spatial dimensions for computation
    y_true_flat = tf.reshape(y_true_onehot, [-1, NUM_CLASSES])
    y_pred_flat = tf.reshape(y_pred, [-1, NUM_CLASSES])
    
    # Compute dice coefficient per class
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
    union = tf.reduce_sum(y_true_flat, axis=0) + tf.reduce_sum(y_pred_flat, axis=0)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    # Average across classes
    return 1.0 - tf.reduce_mean(dice)


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Categorical Focal Loss function for handling highly imbalanced datasets.
    Down-weights easy examples and focuses training on hard negatives.
    """
    def loss_fn(y_true, y_pred):
        # Convert y_true to one-hot for dot product matching
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), NUM_CLASSES)
        
        # Clip y_pred to prevent log(0) NaN errors
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        # Cross Entropy term
        cross_entropy = -y_true_onehot * tf.math.log(y_pred)
        
        # Focal multiplier
        loss = alpha * tf.math.pow(1.0 - y_pred, gamma) * cross_entropy
        
        # Mean across everything
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    
    return loss_fn


def combined_loss(class_weights_dict=None):
    """
    Create a combined loss function: Focal Loss + Dice Loss.
    """
    # Create Focal Loss
    focal_fn = categorical_focal_loss()
    
    def loss_fn(y_true, y_pred):
        # Focal Loss component
        focal = focal_fn(y_true, y_pred)
        
        if USE_DICE_LOSS:
            # Dice loss component
            dl = dice_loss(y_true, y_pred)
            # Weighted combination
            total = CE_LOSS_WEIGHT * focal + DICE_LOSS_WEIGHT * dl
        else:
            total = focal
        
        return total
    
    return loss_fn


def create_sample_weight_map(y_true, class_weights_dict):
    """
    Create per-pixel sample weights based on class weights.
    
    This gives more importance to underrepresented classes
    during training (e.g., water gets higher weight).
    
    Args:
        y_true:             Label array, shape (N, H, W)
        class_weights_dict: Dictionary of class_id → weight
    
    Returns:
        Sample weight array, shape (N, H, W)
    """
    weight_map = np.ones_like(y_true, dtype=np.float32)
    for class_id, weight in class_weights_dict.items():
        weight_map[y_true == class_id] = weight
    return weight_map


def train_model():
    """
    Main training function.
    
    Steps:
      1. Build dataset (train + val)
      2. Build U-Net model
      3. Configure loss, optimizer, callbacks
      4. Train the model
      5. Save model and training plots
    
    Returns:
        model:   Trained Keras model
        history: Training history dictionary
    """
    print_banner("STEP 4: Model Training")
    
    # ── 1. Build datasets ─────────────────────────────────────
    train_ds, val_ds, class_weights, n_train, n_val = create_datasets()
    
    # ── 2. Build model ────────────────────────────────────────
    print("\n  Building U-Net model...")
    model = build_unet()
    model.summary(line_length=90, print_fn=lambda x: print(f"    {x}"))
    
    total_params = model.count_params()
    print(f"\n  Total parameters: {total_params:,d}")
    print(f"  Estimated memory: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # ── 3. Compile model ──────────────────────────────────────
    print("\n  Compiling model...")
    print(f"    Loss: {'CE + Dice' if USE_DICE_LOSS else 'Sparse CE'}")
    print(f"    Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"    Class weights: {'Enabled' if USE_CLASS_WEIGHTS else 'Disabled'}")
    
    loss_fn = combined_loss(class_weights if USE_CLASS_WEIGHTS else None)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    # ── 4. Setup callbacks ────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "unet_lulc_best.keras")
    csv_path = os.path.join(MODEL_DIR, "training_log.csv")
    
    callbacks = [
        # Save best model based on validation loss
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Stop training if no improvement for N epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Log metrics to CSV
        tf.keras.callbacks.CSVLogger(csv_path, append=False),
    ]
    
    print(f"\n  Callbacks:")
    print(f"    → ModelCheckpoint: {model_path}")
    print(f"    → EarlyStopping: patience={EARLY_STOP_PATIENCE}")
    print(f"    → ReduceLROnPlateau: patience={LR_REDUCE_PATIENCE}, factor={LR_REDUCE_FACTOR}")
    print(f"    → CSVLogger: {csv_path}")
    
    # ── 5. Train ──────────────────────────────────────────────
    print(f"\n  Starting training...")
    print(f"    Epochs: {EPOCHS} (max, with early stopping)")
    print(f"    Batch size: {BATCH_SIZE}")
    print(f"    Train samples: {n_train}")
    print(f"    Val samples: {n_val}")
    print()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # ── 6. Save results ───────────────────────────────────────
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, "unet_lulc_final.keras")
    model.save(final_model_path)
    print(f"\n  ✓ Final model saved to: {final_model_path}")
    print(f"  ✓ Best model saved to: {model_path}")
    
    # Plot training curves
    history_dict = history.history
    plot_path = os.path.join(PLOTS_DIR, "training_curves.png")
    plot_training_history(history_dict, save_path=plot_path)
    
    # Print final metrics
    final_train_loss = history_dict['loss'][-1]
    final_val_loss = history_dict['val_loss'][-1]
    final_train_acc = history_dict['accuracy'][-1]
    final_val_acc = history_dict['val_accuracy'][-1]
    
    print(f"\n  ──── Final Metrics ────────────────────")
    print(f"    Train Loss:     {final_train_loss:.4f}")
    print(f"    Val Loss:       {final_val_loss:.4f}")
    print(f"    Train Accuracy: {final_train_acc:.4f}")
    print(f"    Val Accuracy:   {final_val_acc:.4f}")
    print(f"    Epochs trained: {len(history_dict['loss'])}")
    
    return model, history_dict


if __name__ == "__main__":
    model, history = train_model()
