"""
============================================================
RESIDUAL U-NET MODEL — LULC Semantic Segmentation
============================================================
Implements a High-Accuracy ResUNet architecture for 5-class
land use / land cover classification.

Architecture:
  Encoder (4 blocks) → Bottleneck → Decoder (4 blocks)
  with skip connections between encoder and decoder.
  Each block is a Residual Block (x + F(x)) to preserve
  fine localization details and solve vanishing gradients.

Designed to run on 8-16GB RAM systems with batch_size=8.
============================================================
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from config import INPUT_SHAPE, NUM_CLASSES, UNET_FILTERS, DROPOUT_RATE
from utils import print_banner


def residual_block(x, filters, name_prefix):
    """
    Residual block replacing standard double convolution.
    
    Structure: Conv → BN → ReLU → Conv → BN → ADD(shortcut) → ReLU
    
    Args:
        x:           Input tensor
        filters:     Number of convolution filters
        name_prefix: String prefix for layer names
    
    Returns:
        Output tensor after residual operation
    """
    # 1x1 conv shortcut to match channel dimensions if needed
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, (1, 1), padding='same',
            name=f'{name_prefix}_shortcut'
        )(x)
        shortcut = layers.BatchNormalization(name=f'{name_prefix}_shortcut_bn')(shortcut)
        
    # First convolution
    x = layers.Conv2D(
        filters, (3, 3), padding='same',
        kernel_initializer='he_normal',
        name=f'{name_prefix}_conv1'
    )(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Activation('relu', name=f'{name_prefix}_relu1')(x)
    
    # Second convolution
    x = layers.Conv2D(
        filters, (3, 3), padding='same',
        kernel_initializer='he_normal',
        name=f'{name_prefix}_conv2'
    )(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    
    # Add shortcut
    x = layers.Add(name=f'{name_prefix}_add')([shortcut, x])
    x = layers.Activation('relu', name=f'{name_prefix}_relu2')(x)
    
    return x


def encoder_block(x, filters, name_prefix):
    """
    Encoder block: residual_block → MaxPool.
    
    Returns both the conv output (for skip connection)
    and the downsampled output (for next encoder level).
    """
    skip = residual_block(x, filters, name_prefix)
    pooled = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_pool')(skip)
    return skip, pooled


def decoder_block(x, skip, filters, name_prefix):
    """
    Decoder block: UpConv → Concatenate(skip) → residual_block.
    """
    # Upsample using transpose convolution
    x = layers.Conv2DTranspose(
        filters, (2, 2), strides=(2, 2), padding='same',
        kernel_initializer='he_normal',
        name=f'{name_prefix}_upconv'
    )(x)
    
    # Concatenate with skip connection
    x = layers.Concatenate(name=f'{name_prefix}_concat')([x, skip])
    
    # Apply bottleneck dropout dynamically in deeper layers
    x = layers.Dropout(DROPOUT_RATE / 2.0, name=f'{name_prefix}_dropout')(x)
    
    # Residual convolution
    x = residual_block(x, filters, name_prefix)
    
    return x


def build_unet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Build the complete Residual U-Net model.
    
    Architecture:
    Input: (256, 256, 6) — [Blue, Green, Red, NIR, SWIR, NDVI]
    
    ENCODER:
      Block 1: 32 filters  → skip1, pool → (128, 128)
      Block 2: 64 filters  → skip2, pool → (64, 64)
      Block 3: 128 filters → skip3, pool → (32, 32)
      Block 4: 256 filters → skip4, pool → (16, 16)
    
    BOTTLENECK:
      512 filters + Dropout → (16, 16)
    
    DECODER:
      Block 5: 256 filters + skip4 → (32, 32)
      Block 6: 128 filters + skip3 → (64, 64)
      Block 7: 64 filters  + skip2 → (128, 128)
      Block 8: 32 filters  + skip1 → (256, 256)
    
    OUTPUT:
      Conv(5, 1×1, softmax) → (256, 256, 5)
    """
    filters = UNET_FILTERS  # [32, 64, 128, 256, 512]
    
    # ── Input ─────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # ── Encoder Path ──────────────────────────────────────────
    skip1, x = encoder_block(inputs, filters[0], 'enc1')  # 256→128
    skip2, x = encoder_block(x,      filters[1], 'enc2')  # 128→64
    skip3, x = encoder_block(x,      filters[2], 'enc3')  # 64→32
    skip4, x = encoder_block(x,      filters[3], 'enc4')  # 32→16
    
    # ── Bottleneck ────────────────────────────────────────────
    x = residual_block(x, filters[4], 'bottleneck')       # 16×16×512
    x = layers.Dropout(DROPOUT_RATE, name='bottleneck_dropout')(x)
    
    # ── Decoder Path ──────────────────────────────────────────
    x = decoder_block(x, skip4, filters[3], 'dec4')       # 16→32
    x = decoder_block(x, skip3, filters[2], 'dec3')       # 32→64
    x = decoder_block(x, skip2, filters[1], 'dec2')       # 64→128
    x = decoder_block(x, skip1, filters[0], 'dec1')       # 128→256
    
    # ── Output Layer ──────────────────────────────────────────
    outputs = layers.Conv2D(
        num_classes, (1, 1),
        activation='softmax',
        name='output_segmentation'
    )(x)
    
    # ── Build Model ───────────────────────────────────────────
    model = Model(inputs=inputs, outputs=outputs, name='ResUNet_LULC')
    
    return model


def get_model_summary():
    """Build model and print summary."""
    print_banner("Residual U-Net Model Architecture")
    
    model = build_unet()
    model.summary(line_length=100)
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    )
    
    print(f"\n  Total parameters:     {total_params:>12,d}")
    print(f"  Trainable parameters: {trainable_params:>12,d}")
    print(f"  Estimated size:       {total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    return model


if __name__ == "__main__":
    model = get_model_summary()
