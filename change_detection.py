"""
============================================================
CHANGE DETECTION — LULC U-Net Semantic Segmentation
============================================================
Computes and visualizes land use/land cover changes between
two prediction periods. Allows highlighting specific gain/loss
transitions with specific colors.
============================================================
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import PREDICTIONS_DIR, PLOTS_DIR

# The user's requested mapping for change detection logic
CD_CLASS_NAMES = {
    1: "Water",
    2: "Vegetation",
    4: "Barren",
    5: "Built-up",
    6: "Canopy",
}

# Base pipeline to Change Detection class IDs
# Pipeline: 0=Vegetation, 1=Canopy, 2=Built-up, 3=Water, 4=Barren
PIPELINE_TO_CD = {
    0: 2, # Vegetation
    1: 6, # Canopy
    2: 5, # Built-up
    3: 1, # Water
    4: 4, # Barren
    255: 0 # No Data
}

# ── LC Colormap — matches base code exactly ──
LC_COLORMAP = {
    0: (0,   0,   0,   0),    # No Data    → Transparent
    1: (0,   0,   255, 255),  # Water      → Blue
    2: (0,   255, 0,   255),  # Vegetation → Green
    4: (210, 180, 140, 255),  # Barren     → Tan
    5: (255, 0,   0,   255),  # Built-up   → Red
    6: (0,   100, 0,   255),  # Canopy     → Dark Green
}

# ── Transition colors — matches base code exactly ──
TRANSITION_COLORS = {
    (0, 0): (0,   0,   0,   0),

    # No change — Dark Gray
    (1, 1): (80,  80,  80,  255),
    (2, 2): (80,  80,  80,  255),
    (4, 4): (80,  80,  80,  255),
    (5, 5): (80,  80,  80,  255),
    (6, 6): (80,  80,  80,  255),

    # Urban expansion — warm colors
    (2, 5): (255, 0,   0,   255),  # Vegetation → Built-up → Red
    (4, 5): (255, 165, 0,   255),  # Barren     → Built-up → Orange
    (1, 5): (128, 0,   128, 255),  # Water      → Built-up → Purple
    (6, 5): (255, 69,  0,   255),  # Canopy     → Built-up → Red-Orange KEY

    # Vegetation gain — green tones
    (4, 2): (0,   255, 0,   255),  # Barren     → Vegetation → Green
    (5, 2): (255, 255, 0,   255),  # Built-up   → Vegetation → Yellow
    (1, 2): (255, 0,   255, 255),  # Water      → Vegetation → Magenta
    (6, 2): (144, 238, 144, 255),  # Canopy     → Vegetation → Light Green

    # Canopy gain — dark green tones
    (2, 6): (0,   100, 0,   255),  # Vegetation → Canopy → Dark Green KEY
    (4, 6): (34,  139, 34,  255),  # Barren     → Canopy → Forest Green
    (5, 6): (0,   128, 0,   255),  # Built-up   → Canopy → Medium Green
    (1, 6): (0,   200, 100, 255),  # Water      → Canopy → Teal Green

    # Canopy loss
    (6, 4): (205, 133, 63,  255),  # Canopy → Barren  → Peru Brown KEY
    (6, 1): (0,   180, 180, 255),  # Canopy → Water   → Teal Blue

    # Barren transitions
    (2, 4): (255, 200, 100, 255),  # Vegetation → Barren → Light Orange
    (5, 4): (139, 69,  19,  255),  # Built-up   → Barren → Brown
    (1, 4): (180, 140, 100, 255),  # Water      → Barren → Khaki

    # Water transitions
    (2, 1): (0,   0,   180, 255),  # Vegetation → Water → Deep Blue
    (4, 1): (0,   255, 255, 255),  # Barren     → Water → Cyan
    (5, 1): (135, 206, 235, 255),  # Built-up   → Water → Light Blue
}

def generate_change_map(y1_tif_path, y2_tif_path, output_filename="change_map", output_dir=None):
    """
    Generate a change detection map using predicted GeoTIFFs.
    """
    out_d = output_dir if output_dir else PREDICTIONS_DIR
    
    print(f"\n  → Loading Y1 predictions: {os.path.basename(y1_tif_path)}")
    print(f"  → Loading Y2 predictions: {os.path.basename(y2_tif_path)}")
    
    with rasterio.open(y1_tif_path) as src1:
        y1_pred = src1.read(1)
        profile = src1.profile
        
    with rasterio.open(y2_tif_path) as src2:
        y2_pred = src2.read(1)
    
    # Ensure dimensions match
    if y1_pred.shape != y2_pred.shape:
        print("  ⚠ Dimension mismatch. Truncating to minimum shape...")
        min_h = min(y1_pred.shape[0], y2_pred.shape[0])
        min_w = min(y1_pred.shape[1], y2_pred.shape[1])
        y1_pred = y1_pred[:min_h, :min_w]
        y2_pred = y2_pred[:min_h, :min_w]
        
    print("  → Mapping classes to Change Detection layout...")
    
    # Map from Pipeline classes to CD classes
    y1_mapped = np.zeros_like(y1_pred, dtype=np.uint8)
    y2_mapped = np.zeros_like(y2_pred, dtype=np.uint8)
    
    for pipe_id, cd_id in PIPELINE_TO_CD.items():
        y1_mapped[y1_pred == pipe_id] = cd_id
        y2_mapped[y2_pred == pipe_id] = cd_id
    
    # Apply No Data where either y1 or y2 is 0 or unmapped
    no_data_mask = (y1_mapped == 0) | (y2_mapped == 0)
    y1_mapped[no_data_mask] = 0
    y2_mapped[no_data_mask] = 0
    
    # Calculate Built-up area increase
    res_x, res_y = src1.res
    pixel_area_m2 = abs(res_x * res_y) # absolute value in case of negative res
    
    # If the image CRS is geographic (degrees), convert to square kilometers approximately
    if src1.crs and src1.crs.is_geographic:
        # Approximate 1 degree ~ 111.32 km. For area: (111.32)^2 * cos(lat)
        # Using center latitude
        bounds = src1.bounds
        center_lat = (bounds.bottom + bounds.top) / 2.0
        # 1 deg lat = 111.32 km, 1 deg lon = 111.32 * cos(lat) km
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * np.cos(np.radians(center_lat))
        pixel_area_sqkm = (res_x * km_per_deg_lon) * (res_y * km_per_deg_lat)
        pixel_area_sqkm = abs(pixel_area_sqkm)
    else:
        # Assuming coordinates are already in meters for projected CRS
        pixel_area_sqkm = pixel_area_m2 / 1_000_000.0

    builtup_id = 5
    builtup_y1_pixels = np.sum((y1_mapped == builtup_id) & (~no_data_mask))
    builtup_y2_pixels = np.sum((y2_mapped == builtup_id) & (~no_data_mask))
        
    builtup_diff_pixels = builtup_y2_pixels - builtup_y1_pixels
    builtup_diff_sqkm = builtup_diff_pixels * pixel_area_sqkm  # Convert to sq km
        
    if builtup_y1_pixels > 0:
        builtup_pct_change = (builtup_diff_pixels / builtup_y1_pixels) * 100
    else:
        builtup_pct_change = 0.0

    total_valid_pixels = np.sum(~no_data_mask)
    if total_valid_pixels > 0:
        pre_builtup_pct = (builtup_y1_pixels / total_valid_pixels) * 100
        post_builtup_pct = (builtup_y2_pixels / total_valid_pixels) * 100
    else:
        pre_builtup_pct = 0.0
        post_builtup_pct = 0.0

    print(f"    Pre-Built-up Area: {pre_builtup_pct:.2f}% of map")
    print(f"    Post-Built-up Area: {post_builtup_pct:.2f}% of map")
    print(f"    Built-up Area Change (Relative): {builtup_pct_change:+.2f}%")

    # Create the transition RGBA array
    print("  → Applying Transition Colors...")
    h, w = y1_mapped.shape
    change_rgb = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Vectorized application is tricky for pairwise maps without building lookup table,
    # Let's map standard transitions iteratively or construct a 2D lookup array
    
    # Max ID is 6, so 7x7 lookup table is enough (0 to 6)
    lookup_table = np.zeros((7, 7, 4), dtype=np.uint8)
    
    for (c1, c2), color in TRANSITION_COLORS.items():
        lookup_table[c1, c2] = color
        
    # Map colors
    change_rgb = lookup_table[y1_mapped, y2_mapped]
    
    # Save as PNG Figure
    print("  → Generating visualization plot...")
    plot_path = os.path.join(out_d, f"{output_filename}.png")
    
    # Calculate transition stats
    valid_pixels = np.sum(~no_data_mask)
    if valid_pixels > 0:
        no_change_mask = (y1_mapped == y2_mapped) & (~no_data_mask)
        change_pct = (np.sum(~no_data_mask) - np.sum(no_change_mask)) / valid_pixels * 100
        print(f"    Total changed area: {change_pct:.2f}%")
        
        # Breakdown
        print("    Top Transitions:")
        unique_pairs, counts = np.unique(np.stack([y1_mapped[~no_change_mask], y2_mapped[~no_change_mask]], axis=1), axis=0, return_counts=True)
        # sort by count descending
        sorted_idx = np.argsort(-counts)
        for i in sorted_idx[:5]:
            p1, p2 = unique_pairs[i]
            if p1 == p2: continue
            name1 = CD_CLASS_NAMES.get(p1, str(p1))
            name2 = CD_CLASS_NAMES.get(p2, str(p2))
            pct = counts[i] / valid_pixels * 100
            print(f"      {name1} → {name2}: {pct:.2f}%")
            
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(change_rgb)
    ax.set_title("LULC Change Detection Transitions")
    ax.axis("off")
    
    # Stamp the text on the map
    analysis_text = (
        f"Urban Expansion Analysis\n"
        f"Pre Built-up: {pre_builtup_pct:.1f}%\n"
        f"Post Built-up: {post_builtup_pct:.1f}%\n"
        f"Relative Growth: {builtup_pct_change:+.1f}%"
    )
    
    # Place it in the top-left corner
    ax.text(0.02, 0.98, analysis_text, transform=ax.transAxes, 
            fontsize=12, color='white', weight='bold', 
            va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6, edgecolor='none'))

    # Build legend
    legend_elements = []
    
    # Add a few key transition legends based on the keys the user wants
    key_transitions = [
        ((1, 1), "No Change (Stable)"),
        ((2, 5), "Veg → Built-up (Urban Expansion)"),
        ((4, 5), "Barren → Built-up"),
        ((2, 6), "Veg → Canopy (Forest Growth)"),
        ((6, 4), "Canopy → Barren (Deforestation)"),
        ((4, 2), "Barren → Veg"),
        ((1, 5), "Water → Built-up")
    ]
    
    for trans, label in key_transitions:
        if trans in TRANSITION_COLORS:
            color = np.array(TRANSITION_COLORS[trans]) / 255.0
            legend_elements.append(mpatches.Patch(color=color, label=label))
            
    ax.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
              
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved Change Detection map plot to: {plot_path}")
    
    # We could also save an output GEOTIFF
    # To do that, we'd need to encode the transitions into single byte values.
    # e.g. class = y1 * 10 + y2, then write a colormap. 
    # For now, we will create a TIFF of just the RGBA layers so it can be opened in QGIS
    
    tif_path = os.path.join(out_d, f"{output_filename}.tif")
    profile.update(
        count=4,
        dtype=rasterio.uint8,
        nodata=0,
        compress='lzw',
    )
    
    with rasterio.open(tif_path, "w", **profile) as dst:
        for b in range(4):
            dst.write(change_rgb[:, :, b], b+1)
            
    print(f"  ✓ Saved Change Detection GeoTIFF to: {tif_path}")
    return True
    
if __name__ == "__main__":
    from config import SENTINEL_2020, SENTINEL_2025
    
    y1_pred = os.path.join(PREDICTIONS_DIR, f"predicted_lulc_{os.path.basename(SENTINEL_2020)}")
    y2_pred = os.path.join(PREDICTIONS_DIR, f"predicted_lulc_{os.path.basename(SENTINEL_2025)}")
    
    if os.path.exists(y1_pred) and os.path.exists(y2_pred):
        generate_change_map(y1_pred, y2_pred)
    else:
        print("Predictions for both years not found. Run predictions first.")
