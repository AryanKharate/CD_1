"""
============================================================
CONFIGURATION FILE — LULC U-Net Semantic Segmentation
============================================================
Centralizes all settings: paths, hyperparameters, class maps,
color palettes, and AOI coordinates for the Hyderabad region.
============================================================
"""

import os

# ──────────────────────────────────────────────────────────────
# 1. PROJECT PATHS
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PATCHES_DIR = os.path.join(DATA_DIR, "patches")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

# Create directories if they don't exist
for d in [OUTPUT_DIR, PATCHES_DIR, MODEL_DIR, PLOTS_DIR, PREDICTIONS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 2. INPUT DATA FILES
# ──────────────────────────────────────────────────────────────
# Your Sentinel-2 satellite images
SENTINEL_2020 = os.path.join(DATA_DIR, "palgarh_2021-04-27_pre.tif")
SENTINEL_2025 = os.path.join(DATA_DIR, "palgarh_2025-12-27_post.tif")

# ESA WorldCover label file (will be downloaded)
WORLDCOVER_FILE = os.path.join(DATA_DIR, "worldcover_mumbai.tif")

# ──────────────────────────────────────────────────────────────
# 3. AREA OF INTEREST — Automatic detection
# ──────────────────────────────────────────────────────────────
def get_aoi_bounds(filepath):
    """
    Automatically calculate WGS84 (Lat/Lon) bounding box from a GeoTIFF file.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    
    if not os.path.exists(filepath):
        return None
        
    try:
        with rasterio.open(filepath) as src:
            bounds = src.bounds
            # If not already WGS84, transform
            if src.crs.to_string() != 'EPSG:4326':
                w, s, e, n = transform_bounds(src.crs, 'EPSG:4326', *bounds)
                return [round(w, 4), round(s, 4), round(e, 4), round(n, 4)]
            return [round(bounds.left, 4), round(bounds.bottom, 4), 
                    round(bounds.right, 4), round(bounds.top, 4)]
    except Exception as e:
        print(f"  ⚠ Could not extract bounds from {os.path.basename(filepath)}: {e}")
        return None

# Bounding box: [west, south, east, north]
# Try to detect from file, otherwise fallback to Mumbai coordinates
AOI_BOUNDS = get_aoi_bounds(SENTINEL_2020) or [72.740, 18.839, 73.285, 19.496]

if os.path.exists(SENTINEL_2020):
    print(f"  → Detected AOI for {os.path.basename(SENTINEL_2020)}: {AOI_BOUNDS}")

# ──────────────────────────────────────────────────────────────
# 4. SENTINEL-2 BAND CONFIGURATION
# ──────────────────────────────────────────────────────────────
# Band names as stored in the GeoTIFF files
# Based on inspection: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
BAND_NAMES = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# Band indices (1-based) for the bands we need
BAND_B2_IDX = 1   # Blue
BAND_B3_IDX = 2   # Green
BAND_B4_IDX = 3   # Red
BAND_B8_IDX = 7   # NIR
BAND_B11_IDX = 9  # SWIR1 (for NDBI)

# Sentinel-2 scale factor (values are 0-10000, represent reflectance * 10000)
SENTINEL_SCALE = 10000.0

# ──────────────────────────────────────────────────────────────
# 5. ESA WORLDCOVER CLASS MAPPING
# ──────────────────────────────────────────────────────────────
# ESA WorldCover 2021 class codes → Our 5-class scheme
#
# Target classes:
#   0 = Vegetation    (Grassland + Shrubland + Cropland)
#   1 = Dense Canopy  (Tree Cover + Mangroves)
#   2 = Built-up      (Built-up areas)
#   3 = Water         (Water bodies + Wetlands)
#   4 = Barren Land   (Bare/Sparse vegetation + Snow/Ice + Moss/Lichen)

ESA_TO_LULC = {
    10:  1,   # Tree Cover        → Dense Canopy
    20:  0,   # Shrubland         → Vegetation
    30:  0,   # Grassland         → Vegetation
    40:  0,   # Cropland          → Vegetation
    50:  2,   # Built-up          → Built-up
    60:  4,   # Bare/Sparse Veg   → Barren Land
    70:  4,   # Snow and Ice      → Barren Land
    80:  3,   # Permanent Water   → Water
    90:  3,   # Herbaceous Wetland→ Water
    95:  1,   # Mangroves         → Dense Canopy
    100: 4,   # Moss and Lichen   → Barren Land
}

# Number of output classes
NUM_CLASSES = 5

# Class names for display
CLASS_NAMES = [
    "Vegetation",      # 0
    "Dense Canopy",    # 1
    "Built-up",        # 2
    "Water",           # 3
    "Barren Land",     # 4
]

# ──────────────────────────────────────────────────────────────
# 6. VISUALIZATION COLORS (RGB 0-255)
# ──────────────────────────────────────────────────────────────
CLASS_COLORS = {
    0: (0, 255, 0),       # Vegetation    → Green
    1: (0, 100, 0),       # Dense Canopy  → Dark Green
    2: (255, 0, 0),       # Built-up      → Red
    3: (0, 0, 255),       # Water         → Blue
    4: (139, 69, 19),     # Barren Land   → Brown
}

# Normalized colors for matplotlib (0-1 range)
CLASS_COLORS_NORM = {
    k: (r / 255.0, g / 255.0, b / 255.0)
    for k, (r, g, b) in CLASS_COLORS.items()
}

# ──────────────────────────────────────────────────────────────
# 7. IMAGE DIMENSIONS & PATCHING
# ──────────────────────────────────────────────────────────────
PATCH_SIZE = 256          # Output patch dimensions (H, W)
PATCH_STRIDE = 256        # Non-overlapping patches (set < PATCH_SIZE for overlap)
INPUT_CHANNELS = 6        # B, G, R, NIR, SWIR, NDVI
INPUT_SHAPE = (PATCH_SIZE, PATCH_SIZE, INPUT_CHANNELS)

# ──────────────────────────────────────────────────────────────
# 8. TRAINING HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────
BATCH_SIZE = 8            # Optimized for 8GB RAM
EPOCHS = 30               # Max epochs (early stopping may end earlier)
LEARNING_RATE = 1e-4      # Adam optimizer learning rate
VALIDATION_SPLIT = 0.2    # 20% validation
EARLY_STOP_PATIENCE = 10  # Stop if no improvement for 10 epochs
LR_REDUCE_PATIENCE = 5   # Reduce LR if no improvement for 5 epochs
LR_REDUCE_FACTOR = 0.5   # Halve the learning rate
DROPOUT_RATE = 0.3        # Dropout in bottleneck

# ──────────────────────────────────────────────────────────────
# 9. LOSS FUNCTION SETTINGS
# ──────────────────────────────────────────────────────────────
USE_DICE_LOSS = True      # Combine CE + Dice loss
DICE_LOSS_WEIGHT = 0.5    # Weight for dice loss component
CE_LOSS_WEIGHT = 0.5      # Weight for cross-entropy component
USE_CLASS_WEIGHTS = True  # Compute weights from label distribution

# ──────────────────────────────────────────────────────────────
# 10. U-NET MODEL SETTINGS
# ──────────────────────────────────────────────────────────────
# Lightweight filter sizes to fit in 8GB RAM
UNET_FILTERS = [32, 64, 128, 256, 512]

# ──────────────────────────────────────────────────────────────
# 11. RANDOM SEED
# ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
