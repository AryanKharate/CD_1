"""
============================================================
DATA DOWNLOAD — ESA WorldCover Labels
============================================================
Downloads ESA WorldCover 2021 ground truth labels for the
Hyderabad AOI, aligned to match the existing Sentinel-2 images.

Supports two methods:
  1. Google Earth Engine (requires authenticated GEE + cloud project)
  2. Direct download from ESA WorldCover S3 bucket (no auth needed)
============================================================
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
from config import (
    AOI_BOUNDS, WORLDCOVER_FILE, SENTINEL_2020, DATA_DIR
)
from utils import print_banner


# ──────────────────────────────────────────────────────────────
# METHOD 1: Direct download from ESA (NO GEE required)
# ──────────────────────────────────────────────────────────────

def download_worldcover_direct():
    """
    Download ESA WorldCover 2021 tiles directly from the
    ESA S3 bucket. This does NOT require GEE authentication.
    
    The WorldCover tiles are named by their grid coordinates.
    For Hyderabad (around 78°E, 17°N), we need tiles covering
    that area.
    """
    import urllib.request
    
    if os.path.exists(WORLDCOVER_FILE):
        print(f"  ✓ WorldCover file already exists: {WORLDCOVER_FILE}")
        return True
    
    print("  → Downloading ESA WorldCover 2021 (direct method)...")
    
    # ESA WorldCover 2021 v200 tiles are available at:
    # https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/
    # Tile naming: ESA_WorldCover_10m_2021_v200_N{lat}_E{lon}_Map.tif
    #
    # For Hyderabad area (78°E, 17°N), we need the N15E75 tile
    # (tiles are in 3° x 3° blocks)
    
    # Calculate which tile(s) we need based on AOI
    west, south, east, north = AOI_BOUNDS
    
    # Tiles are 3×3 degree blocks, named by bottom-left corner
    # rounded to nearest multiple of 3
    lat_tiles = set()
    lon_tiles = set()
    
    for lat in range(int(np.floor(south / 3) * 3), int(np.ceil(north / 3) * 3), 3):
        for lon in range(int(np.floor(west / 3) * 3), int(np.ceil(east / 3) * 3), 3):
            lat_tiles.add(lat)
            lon_tiles.add(lon)
    
    tile_files = []
    
    for lat in sorted(lat_tiles):
        for lon in sorted(lon_tiles):
            lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
            
            tile_id = f"{lat_str}{lon_str}"
            filename = f"ESA_WorldCover_10m_2021_v200_{tile_id}_Map.tif"
            url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{filename}"
            
            local_path = os.path.join(DATA_DIR, filename)
            tile_files.append(local_path)
            
            if os.path.exists(local_path):
                print(f"  ✓ Tile already downloaded: {filename}")
                continue
            
            print(f"  → Downloading tile: {filename}")
            print(f"    URL: {url}")
            
            try:
                urllib.request.urlretrieve(url, local_path)
                print(f"  ✓ Downloaded: {filename}")
            except Exception as e:
                print(f"  ⚠ Failed to download {filename}: {e}")
                print("  → Will try GEE method instead.")
                return False
    
    # Crop and align the tile(s) to match Sentinel-2
    _crop_and_align_worldcover(tile_files)
    return True


def _crop_and_align_worldcover(tile_files):
    """
    Crop the downloaded WorldCover tile(s) to the AOI and
    resample to match the Sentinel-2 grid exactly.
    
    Uses rasterio's reproject to ensure pixel-perfect alignment.
    """
    from rasterio.merge import merge
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box
    
    print("  → Aligning WorldCover to Sentinel-2 grid...")
    
    # Read Sentinel-2 reference file for target grid
    with rasterio.open(SENTINEL_2020) as ref:
        target_transform = ref.transform
        target_width = ref.width
        target_height = ref.height
        target_crs = ref.crs
        target_bounds = ref.bounds
    
    # Open all tiles
    existing_tiles = [f for f in tile_files if os.path.exists(f)]
    if not existing_tiles:
        print("  ⚠ No WorldCover tiles found!")
        return
    
    # If single tile, process directly
    if len(existing_tiles) == 1:
        src_file = existing_tiles[0]
    else:
        # Merge multiple tiles
        datasets = [rasterio.open(f) for f in existing_tiles]
        merged, merged_transform = merge(datasets)
        for ds in datasets:
            ds.close()
        
        # Save merged temporarily
        src_file = os.path.join(DATA_DIR, "worldcover_merged_temp.tif")
        with rasterio.open(
            src_file, 'w', driver='GTiff',
            height=merged.shape[1], width=merged.shape[2],
            count=1, dtype=merged.dtype,
            crs=datasets[0].crs, transform=merged_transform
        ) as dst:
            dst.write(merged)
    
    # Reproject and resample to match Sentinel-2 grid
    with rasterio.open(src_file) as wc:
        # Create aligned output array
        aligned = np.zeros((target_height, target_width), dtype=np.uint8)
        
        reproject(
            source=rasterio.band(wc, 1),
            destination=aligned,
            src_transform=wc.transform,
            src_crs=wc.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest  # MUST be nearest for categorical data!
        )
    
    # Save aligned WorldCover
    with rasterio.open(
        WORLDCOVER_FILE, 'w',
        driver='GTiff',
        height=target_height,
        width=target_width,
        count=1,
        dtype='uint8',
        crs=target_crs,
        transform=target_transform
    ) as dst:
        dst.write(aligned, 1)
    
    # Verify
    unique_vals = np.unique(aligned)
    print(f"  ✓ Aligned WorldCover saved: {target_width}x{target_height}")
    print(f"    Unique class values: {unique_vals}")
    
    # Clean up temp files
    temp = os.path.join(DATA_DIR, "worldcover_merged_temp.tif")
    if os.path.exists(temp):
        os.remove(temp)


# ──────────────────────────────────────────────────────────────
# METHOD 2: Google Earth Engine
# ──────────────────────────────────────────────────────────────

def authenticate_gee():
    """
    Authenticate and initialize Google Earth Engine.
    
    IMPORTANT: Requires a GCP Cloud Project with Earth Engine API enabled.
    Set your project via: earthengine set_project YOUR_PROJECT_ID
    """
    import ee
    
    try:
        ee.Initialize()
        print("  ✓ Google Earth Engine initialized successfully.")
    except Exception as e:
        if "no project found" in str(e).lower():
            print("  ⚠ GEE requires a Cloud Project.")
            print("    To set up:")
            print("    1. Go to https://console.cloud.google.com/projectcreate")
            print("    2. Create a project (e.g., 'my-ee-project')")
            print("    3. Enable Earth Engine API for the project")
            print("    4. Run: earthengine set_project YOUR_PROJECT_ID")
            print()
            print("  → Falling back to direct download method...")
            return None
        
        print("  → Authenticating with Google Earth Engine...")
        try:
            ee.Authenticate()
            ee.Initialize()
            print("  ✓ Authenticated and initialized.")
        except Exception as auth_err:
            print(f"  ⚠ GEE authentication failed: {auth_err}")
            print("  → Falling back to direct download method...")
            return None
    
    return ee


def download_worldcover_gee(ee_module):
    """
    Download ESA WorldCover 2021 via Google Earth Engine.
    """
    ee = ee_module
    
    if os.path.exists(WORLDCOVER_FILE):
        print(f"  ✓ WorldCover file already exists: {WORLDCOVER_FILE}")
        return
    
    print("  → Downloading ESA WorldCover 2021 via GEE...")
    
    west, south, east, north = AOI_BOUNDS
    aoi = ee.Geometry.Rectangle([west, south, east, north])
    
    worldcover = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .filterBounds(aoi)
        .first()
        .select("Map")
        .clip(aoi)
    )
    
    with rasterio.open(SENTINEL_2020) as src:
        target_crs = str(src.crs)
    
    try:
        import geemap
        geemap.ee_export_image(
            worldcover,
            filename=WORLDCOVER_FILE,
            scale=10,
            region=aoi,
            crs=target_crs,
            file_per_band=False
        )
        print(f"  ✓ Downloaded WorldCover via GEE")
    except Exception as e:
        print(f"  ⚠ GEE download failed: {e}")
        import urllib.request
        url = worldcover.getDownloadURL({
            'scale': 10, 'region': aoi,
            'crs': target_crs, 'format': 'GEO_TIFF'
        })
        urllib.request.urlretrieve(url, WORLDCOVER_FILE)
        print(f"  ✓ Downloaded WorldCover (fallback)")
    
    # Align to Sentinel-2 grid
    _align_worldcover_gee()


def _align_worldcover_gee():
    """Reproject GEE-downloaded WorldCover to match Sentinel-2 grid."""
    if not os.path.exists(WORLDCOVER_FILE):
        return
    
    with rasterio.open(SENTINEL_2020) as ref:
        target_transform = ref.transform
        target_width = ref.width
        target_height = ref.height
        target_crs = ref.crs
    
    with rasterio.open(WORLDCOVER_FILE) as wc:
        if wc.width == target_width and wc.height == target_height:
            print("  ✓ WorldCover already aligned.")
            return
        
        wc_data = wc.read(1)
        wc_transform = wc.transform
        wc_crs = wc.crs
    
    aligned = np.zeros((target_height, target_width), dtype=np.uint8)
    reproject(
        source=wc_data,
        destination=aligned,
        src_transform=wc_transform,
        src_crs=wc_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest
    )
    
    aligned_file = WORLDCOVER_FILE + ".aligned.tif"
    with rasterio.open(
        aligned_file, 'w', driver='GTiff',
        height=target_height, width=target_width,
        count=1, dtype='uint8',
        crs=target_crs, transform=target_transform
    ) as dst:
        dst.write(aligned, 1)
    
    os.replace(aligned_file, WORLDCOVER_FILE)
    print(f"  ✓ Aligned WorldCover to Sentinel-2 grid: {target_width}x{target_height}")


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────

def run_download():
    """
    Main entry point for data download.
    Tries direct download first, falls back to GEE if needed.
    """
    print_banner("STEP 1: Download ESA WorldCover Labels")
    
    print("\n  Sentinel-2 images already present:")
    print(f"    → {SENTINEL_2020}")
    
    sentinel_2025 = os.path.join(DATA_DIR, "hyd_2025.tif")
    if os.path.exists(sentinel_2025):
        print(f"    → {sentinel_2025}")
    
    # Check if already downloaded
    if os.path.exists(WORLDCOVER_FILE):
        print(f"\n  ✓ WorldCover already exists: {WORLDCOVER_FILE}")
        _verify_alignment()
        return
    
    print("\n  Downloading ground truth labels...")
    
    # Method 1: Direct download (no auth required)
    print("\n  Trying direct download from ESA servers...")
    success = download_worldcover_direct()
    
    if not success:
        # Method 2: Google Earth Engine
        print("\n  Trying Google Earth Engine...")
        ee = authenticate_gee()
        if ee:
            download_worldcover_gee(ee)
        else:
            print("\n  ⚠ Could not download WorldCover automatically.")
            print("    Please download manually or set up GEE project.")
            sys.exit(1)
    
    _verify_alignment()
    print("\n  ✓ Data download complete!")


def _verify_alignment():
    """Verify that WorldCover is aligned with Sentinel-2."""
    if not os.path.exists(WORLDCOVER_FILE):
        return
    
    with rasterio.open(SENTINEL_2020) as s2:
        with rasterio.open(WORLDCOVER_FILE) as wc:
            if s2.width == wc.width and s2.height == wc.height:
                print(f"\n  ✓ Alignment verified: {s2.width}x{s2.height} pixels")
            else:
                print(f"\n  ⚠ Size mismatch: S2={s2.width}x{s2.height}, WC={wc.width}x{wc.height}")
                print("    Re-aligning...")
                _align_worldcover_gee()


if __name__ == "__main__":
    run_download()
