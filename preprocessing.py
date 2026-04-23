"""
============================================================
PREPROCESSING — LULC U-Net Semantic Segmentation
============================================================
Handles all data preprocessing:
  1. Band extraction from Sentinel-2 GeoTIFFs
  2. NDVI / NDWI / NDBI computation
  3. Input stacking (R, G, B, NDVI)
  4. ESA WorldCover → 5-class label mapping
  5. Patching into 256×256 tiles
  6. Invalid pixel handling
============================================================
"""

import os
import numpy as np
import rasterio
from tqdm import tqdm
from config import (
    SENTINEL_2020, SENTINEL_2025, WORLDCOVER_FILE,
    PATCHES_DIR, PATCH_SIZE, PATCH_STRIDE,
    BAND_B2_IDX, BAND_B3_IDX, BAND_B4_IDX, BAND_B8_IDX, BAND_B11_IDX,
    SENTINEL_SCALE, ESA_TO_LULC, NUM_CLASSES
)
from utils import compute_ndvi, compute_ndwi, compute_ndbi, print_banner


def load_sentinel_bands(filepath):
    """
    Load and extract required bands from a Sentinel-2 GeoTIFF.
    
    Extracts: Blue (B2), Green (B3), Red (B4), NIR (B8), SWIR (B11)
    Normalizes reflectance values from [0, 10000] to [0, 1].
    
    Args:
        filepath: Path to Sentinel-2 GeoTIFF
    
    Returns:
        Dictionary with band arrays: {'blue', 'green', 'red', 'nir', 'swir'}
        All arrays have shape (H, W) with values in [0, 1]
    """
    print(f"  → Loading bands from: {os.path.basename(filepath)}")
    
    with rasterio.open(filepath) as src:
        # Read required bands (1-indexed)
        blue  = src.read(BAND_B2_IDX).astype(np.float32)
        green = src.read(BAND_B3_IDX).astype(np.float32)
        red   = src.read(BAND_B4_IDX).astype(np.float32)
        nir   = src.read(BAND_B8_IDX).astype(np.float32)
        swir  = src.read(BAND_B11_IDX).astype(np.float32)
    
    # Replace NaN with 0 before normalization
    for arr in [blue, green, red, nir, swir]:
        arr[np.isnan(arr)] = 0.0
    
    # Normalize from Sentinel-2 scale (0-10000) to (0-1)
    blue  = np.clip(blue  / SENTINEL_SCALE, 0, 1)
    green = np.clip(green / SENTINEL_SCALE, 0, 1)
    red   = np.clip(red   / SENTINEL_SCALE, 0, 1)
    nir   = np.clip(nir   / SENTINEL_SCALE, 0, 1)
    swir  = np.clip(swir  / SENTINEL_SCALE, 0, 1)
    
    print(f"    Blue  (B2):  min={blue.min():.4f}, max={blue.max():.4f}")
    print(f"    Green (B3):  min={green.min():.4f}, max={green.max():.4f}")
    print(f"    Red   (B4):  min={red.min():.4f}, max={red.max():.4f}")
    print(f"    NIR   (B8):  min={nir.min():.4f}, max={nir.max():.4f}")
    print(f"    SWIR  (B11): min={swir.min():.4f}, max={swir.max():.4f}")
    
    return {
        'blue': blue,
        'green': green,
        'red': red,
        'nir': nir,
        'swir': swir
    }


def build_input_stack(bands):
    """
    Build the 6-channel input stack: [Blue, Green, Red, NIR, SWIR, NDVI].
    
    This deep stack allows the ResUNet to decouple complex classes.
    NDVI is computed from NIR and Red bands and normalized to [0, 1].
    
    Args:
        bands: Dictionary from load_sentinel_bands()
    
    Returns:
        3D array of shape (H, W, 6) with channels [B, G, R, NIR, SWIR, NDVI]
    """
    # Compute NDVI
    ndvi = compute_ndvi(bands['nir'], bands['red'])
    print(f"    NDVI: min={ndvi.min():.4f}, max={ndvi.max():.4f}, mean={ndvi.mean():.4f}")
    
    # Optional: compute NDWI and NDBI for reference (not in main input)
    ndwi = compute_ndwi(bands['green'], bands['nir'])
    ndbi = compute_ndbi(bands['swir'], bands['nir'])
    print(f"    NDWI: min={ndwi.min():.4f}, max={ndwi.max():.4f} (for water detection)")
    print(f"    NDBI: min={ndbi.min():.4f}, max={ndbi.max():.4f} (for built-up detection)")
    
    # Stack channels: [Blue, Green, Red, NIR, SWIR, NDVI]
    # Shape: (H, W, 6)
    input_stack = np.stack([
        bands['blue'],
        bands['green'],
        bands['red'],
        bands['nir'],
        bands['swir'],
        ndvi
    ], axis=-1)
    
    print(f"    Input stack shape: {input_stack.shape}")
    return input_stack


def load_worldcover_labels():
    """
    Load ESA WorldCover and map to our 5-class scheme.
    
    ESA WorldCover classes → Our classes:
      10 (Tree Cover)      → 1 (Dense Canopy)
      20 (Shrubland)       → 0 (Vegetation)
      30 (Grassland)       → 0 (Vegetation)
      40 (Cropland)        → 0 (Vegetation)
      50 (Built-up)        → 2 (Built-up)
      60 (Bare/Sparse)     → 4 (Barren Land)
      70 (Snow/Ice)        → 4 (Barren Land)
      80 (Water)           → 3 (Water)
      90 (Wetland)         → 3 (Water)
      100 (Moss/Lichen)    → 4 (Barren Land)
    
    Returns:
        2D numpy array of shape (H, W) with integer class labels [0-4]
    """
    print(f"  → Loading WorldCover labels from: {os.path.basename(WORLDCOVER_FILE)}")
    
    with rasterio.open(WORLDCOVER_FILE) as src:
        wc_data = src.read(1)
    
    # Print ESA class distribution before mapping
    unique, counts = np.unique(wc_data, return_counts=True)
    print("    ESA WorldCover class distribution:")
    esa_names = {
        0: "No Data", 10: "Tree Cover", 20: "Shrubland", 30: "Grassland",
        40: "Cropland", 50: "Built-up", 60: "Bare/Sparse", 70: "Snow/Ice",
        80: "Water", 90: "Wetland", 95: "Mangroves", 100: "Moss/Lichen"
    }
    for val, cnt in zip(unique, counts):
        name = esa_names.get(val, f"Unknown({val})")
        pct = cnt / wc_data.size * 100
        print(f"      {val:>3d} ({name:>15s}): {cnt:>10,d} pixels ({pct:>5.1f}%)")
    
    # Map ESA classes to our 5-class scheme
    labels = np.full_like(wc_data, fill_value=255, dtype=np.uint8)  # 255 = unmapped
    for esa_class, our_class in ESA_TO_LULC.items():
        labels[wc_data == esa_class] = our_class
    
    # Handle unmapped pixels (e.g., nodata=0) — assign to nearest valid class
    # For simplicity, we'll mark them and exclude during training
    unmapped_count = np.sum(labels == 255)
    if unmapped_count > 0:
        unmapped_pct = unmapped_count / labels.size * 100
        print(f"    ⚠ {unmapped_count:,d} unmapped pixels ({unmapped_pct:.1f}%) — will be excluded")
    
    # Print our class distribution
    print("    Mapped LULC class distribution:")
    from config import CLASS_NAMES
    for i, name in enumerate(CLASS_NAMES):
        cnt = np.sum(labels == i)
        pct = cnt / labels.size * 100
        print(f"      {i} ({name:>15s}): {cnt:>10,d} pixels ({pct:>5.1f}%)")
    
    return labels


def create_validity_mask(bands):
    """
    Create a boolean mask indicating valid (non-NaN, non-zero) pixels.
    
    The Sentinel-2 images have ~44% NaN values at the edges.
    We only want to create patches from valid data regions.
    
    Args:
        bands: Dictionary from load_sentinel_bands()
    
    Returns:
        Boolean mask, shape (H, W), True = valid pixel
    """
    # A pixel is valid if ALL bands have non-zero values
    valid = np.ones_like(bands['red'], dtype=bool)
    for band_name in ['red', 'green', 'blue', 'nir']:
        valid &= (bands[band_name] > 0)
    
    valid_pct = valid.sum() / valid.size * 100
    print(f"    Valid pixels: {valid.sum():,d} / {valid.size:,d} ({valid_pct:.1f}%)")
    
    return valid


def extract_patches(image, labels, valid_mask, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    """
    Extract non-overlapping 256x256 patches from the full image and labels.
    
    Only patches where >90% of pixels are valid (non-NaN) are kept.
    This avoids training on mostly-empty patches from image edges.
    
    Args:
        image:      4-channel input, shape (H, W, 4)
        labels:     Integer labels, shape (H, W)
        valid_mask: Boolean mask, shape (H, W)
        patch_size: Size of each patch (default: 256)
        stride:     Step between patches (default: 256, non-overlapping)
    
    Returns:
        Tuple of (image_patches, label_patches):
          - image_patches: list of arrays, each (patch_size, patch_size, 4)
          - label_patches: list of arrays, each (patch_size, patch_size)
    """
    h, w = image.shape[:2]
    image_patches = []
    label_patches = []
    skipped = 0
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch regions
            valid_patch = valid_mask[y:y+patch_size, x:x+patch_size]
            
            # Skip patches with too many invalid pixels (< 90% valid)
            valid_ratio = valid_patch.sum() / (patch_size * patch_size)
            if valid_ratio < 0.90:
                skipped += 1
                continue
            
            # Extract image and label patch
            img_patch = image[y:y+patch_size, x:x+patch_size, :]
            lbl_patch = labels[y:y+patch_size, x:x+patch_size]
            
            # Skip patches with unmapped labels (255)
            if np.any(lbl_patch == 255):
                # Fill unmapped pixels with the most common class in the patch
                valid_labels = lbl_patch[lbl_patch != 255]
                if len(valid_labels) == 0:
                    skipped += 1
                    continue
                mode_class = np.bincount(valid_labels).argmax()
                lbl_patch[lbl_patch == 255] = mode_class
            
            # Fill any remaining zero pixels in image with patch mean
            for c in range(img_patch.shape[-1]):
                channel = img_patch[:, :, c]
                mask = channel == 0
                if mask.any() and (~mask).any():
                    channel[mask] = channel[~mask].mean()
            
            image_patches.append(img_patch.astype(np.float32))
            label_patches.append(lbl_patch.astype(np.uint8))
    
    print(f"    Extracted {len(image_patches)} patches, skipped {skipped} (insufficient valid pixels)")
    return image_patches, label_patches


def save_patches(image_patches, label_patches, prefix="2020"):
    """
    Save image and label patches as .npy files for efficient loading.
    
    Args:
        image_patches: List of image arrays
        label_patches: List of label arrays
        prefix:        Filename prefix (e.g., '2020' or '2025')
    """
    save_dir = os.path.join(PATCHES_DIR, prefix)
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (img, lbl) in enumerate(zip(image_patches, label_patches)):
        np.save(os.path.join(save_dir, f"image_{i:04d}.npy"), img)
        np.save(os.path.join(save_dir, f"label_{i:04d}.npy"), lbl)
    
    print(f"    Saved {len(image_patches)} patches to: {save_dir}")


def load_all_patches():
    """
    Load all saved patches from disk.
    
    Returns:
        images: numpy array, shape (N, 256, 256, 4)
        labels: numpy array, shape (N, 256, 256)
    """
    all_images = []
    all_labels = []
    
    for prefix in ["2020", "2025"]:
        patch_dir = os.path.join(PATCHES_DIR, prefix)
        if not os.path.exists(patch_dir):
            continue
        
        # Find all image patches
        img_files = sorted([
            f for f in os.listdir(patch_dir)
            if f.startswith("image_") and f.endswith(".npy")
        ])
        
        for img_file in img_files:
            lbl_file = img_file.replace("image_", "label_")
            img_path = os.path.join(patch_dir, img_file)
            lbl_path = os.path.join(patch_dir, lbl_file)
            
            if os.path.exists(lbl_path):
                all_images.append(np.load(img_path))
                all_labels.append(np.load(lbl_path))
    
    if len(all_images) == 0:
        raise RuntimeError("No patches found! Run preprocessing first.")
    
    images = np.array(all_images, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.uint8)
    
    print(f"  → Loaded {len(images)} patches total")
    print(f"    Images shape: {images.shape}")
    print(f"    Labels shape: {labels.shape}")
    
    return images, labels


def preprocess_image(filepath, labels, prefix):
    """
    Full preprocessing pipeline for one Sentinel-2 image:
      1. Load bands
      2. Build input stack [R, G, B, NDVI]
      3. Create validity mask
      4. Extract 256×256 patches
      5. Save patches to disk
    
    Args:
        filepath: Path to Sentinel-2 GeoTIFF
        labels:   Full label array (H, W)
        prefix:   Name prefix for saving ('2020' or '2025')
    
    Returns:
        Number of patches extracted
    """
    print(f"\n  Processing: {os.path.basename(filepath)}")
    
    # Step 1: Load bands
    bands = load_sentinel_bands(filepath)
    
    # Step 2: Build 4-channel input [R, G, B, NDVI]
    print("  → Building input stack [R, G, B, NDVI]...")
    input_stack = build_input_stack(bands)
    
    # Step 3: Create validity mask
    print("  → Creating validity mask...")
    valid_mask = create_validity_mask(bands)
    
    # Step 4: Extract patches
    print(f"  → Extracting {PATCH_SIZE}×{PATCH_SIZE} patches...")
    img_patches, lbl_patches = extract_patches(input_stack, labels, valid_mask)
    
    # Step 5: Save patches
    print("  → Saving patches to disk...")
    save_patches(img_patches, lbl_patches, prefix=prefix)
    
    return len(img_patches)


def run_preprocessing():
    """Main entry point for preprocessing."""
    print_banner("STEP 2: Preprocessing & Patch Extraction")
    
    # Check if patches already exist
    existing = 0
    for prefix in ["2020", "2025"]:
        patch_dir = os.path.join(PATCHES_DIR, prefix)
        if os.path.exists(patch_dir):
            n = len([f for f in os.listdir(patch_dir) if f.startswith("image_")])
            existing += n
    
    if existing > 0:
        print(f"  ✓ Found {existing} existing patches. Skipping preprocessing.")
        print(f"    (Delete {PATCHES_DIR} to re-run preprocessing)")
        return
    
    # Load WorldCover labels
    labels = load_worldcover_labels()
    
    # Process 2020 image
    n_2020 = preprocess_image(SENTINEL_2020, labels, prefix="2020")
    
    # Process 2025 image
    sentinel_2025 = SENTINEL_2025 if os.path.exists(SENTINEL_2025) else None
    n_2025 = 0
    if sentinel_2025:
        n_2025 = preprocess_image(sentinel_2025, labels, prefix="2025")
    
    total = n_2020 + n_2025
    print(f"\n  ✓ Preprocessing complete!")
    print(f"    Total patches: {total} ({n_2020} from 2020, {n_2025} from 2025)")
    print(f"    Patch size: {PATCH_SIZE}×{PATCH_SIZE}×4 channels")


if __name__ == "__main__":
    run_preprocessing()
