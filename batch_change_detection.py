import os
import argparse
from glob import glob
from collections import defaultdict
from predict import load_trained_model, predict_full_image
from change_detection import generate_change_map

def find_tif_in_dir(directory):
    """Finds a single .tif file in a directory and returns its path."""
    tifs = glob(os.path.join(directory, "**", "*.tif"), recursive=True)
    # Exclude any previously generated predicted files
    tifs = [t for t in tifs if not os.path.basename(t).startswith("predicted_")]
    if len(tifs) == 0:
        return None
    elif len(tifs) == 1:
        return tifs[0]
    else:
        # If there are multiple, print a warning and pick the first
        print(f"  ⚠ Multiple .tif files found in {directory}. Using {tifs[0]}")
        return tifs[0]

def process_batch_change_detection(input_dir, output_dir):
    """
    Scans the input directory for region folders containing .tif images,
    applies LULC prediction to all .tif files within each region, 
    and runs change detection natively between them if there are exactly two.
    """
    print(f"Scanning input directory '{input_dir}' for regions...")
    
    # Identify all valid subdirectories in the input_dir (regions)
    regions = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    if not regions:
        print(f"No valid region folders found inside {input_dir}")
        return
        
    print("Loading AI Model...")
    model = load_trained_model()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for region in regions:
        region_dir = os.path.join(input_dir, region)
        # Find all .tif files in the region folder
        tifs = glob(os.path.join(region_dir, "**", "*.tif"), recursive=True)
        tifs = [t for t in tifs if not os.path.basename(t).startswith("predicted_")]
        tifs = [t for t in tifs if not "change_map" in os.path.basename(t)]
        
        # Need exactly two files to do change detection
        if len(tifs) != 2:
            print(f"Skipping region '{region}': Expected 2 .tif files for change detection, found {len(tifs)}")
            continue
            
        # Sort by filename naturally (assuming dates are in filenames like _2021-..., _2025-...)
        tifs.sort()
        y1_tif = tifs[0]
        y2_tif = tifs[1]
        
        print(f"\n" + "═"*60)
        print(f" PROCESSING REGION: {region.upper()}")
        print(f"   Date 1 TIF: {os.path.basename(y1_tif)}")
        print(f"   Date 2 TIF: {os.path.basename(y2_tif)}")
        print("═"*60)
        
        # Target output directory for this region
        region_out_dir = os.path.join(output_dir, region)
        os.makedirs(region_out_dir, exist_ok=True)
        
        # 1. Apply LULC to Date 1
        print(f"\n--- Predicting LULC for Date 1 ---")
        predict_full_image(model, y1_tif, output_dir=region_out_dir)
        y1_pred_file = os.path.join(region_out_dir, f"predicted_lulc_{os.path.basename(y1_tif)}")
        
        # 2. Apply LULC to Date 2
        print(f"\n--- Predicting LULC for Date 2 ---")
        predict_full_image(model, y2_tif, output_dir=region_out_dir)
        y2_pred_file = os.path.join(region_out_dir, f"predicted_lulc_{os.path.basename(y2_tif)}")
        
        # 3. Change Detection
        print(f"\n--- Generating Change Detection Map ---")
        if os.path.exists(y1_pred_file) and os.path.exists(y2_pred_file):
            generate_change_map(
                y1_pred_file, 
                y2_pred_file, 
                output_filename=f"{region}_change_map",
                output_dir=region_out_dir
            )
        else:
            print("  ⚠ Prediction files were not generated properly. Skipping change detection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate LULC & Change Detection across multiple regions.")
    parser.add_argument("--input", required=True, help="Input directory containing the separated region folders (e.g. central_2025-11-29).")
    parser.add_argument("--output", required=True, help="Output directory to save the categorized predictions and maps.")
    
    args = parser.parse_args()
    
    process_batch_change_detection(args.input, args.output)