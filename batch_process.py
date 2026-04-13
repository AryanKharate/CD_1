import os
import argparse
from glob import glob
from predict import load_trained_model, predict_full_image

def process_directory(input_dir, output_dir):
    """
    Recursively scans the input_dir for .tif files,
    runs predict_full_image on them, and stores the results
    in output_dir mimicking the original folder structure.
    """
    # Load the best trained model once for all predictions
    print("Loading model...")
    model = load_trained_model()
    
    # Ensure the main output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .tif files recursively
    search_pattern = os.path.join(input_dir, "**", "*.tif")
    tif_files = glob(search_pattern, recursive=True)
    
    if not tif_files:
        print(f"No .tif files found in {input_dir}")
        return
        
    print(f"Found {len(tif_files)} .tif files. Starting batch prediction...\n")
    
    for count, tif_file in enumerate(tif_files, start=1):
        print(f"[{count}/{len(tif_files)}] Processing {tif_file}")
        
        # Get the relative path of the file compared to input directory
        # e.g., if input_dir is /data/ and file is /data/punjab/file.tif, relative is punjab/file.tif
        rel_path = os.path.relpath(tif_file, input_dir)
        rel_dir = os.path.dirname(rel_path)
        
        # Determine current target output directory, maintaining folder structure
        current_out_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(current_out_dir, exist_ok=True)
        
        try:
            # Run prediction on full image directly and output it to the specified location
            predict_full_image(model, tif_file, output_dir=current_out_dir)
        except Exception as e:
            print(f"Error processing {tif_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process multiple Sentinel-2 .tif files.")
    parser.add_argument("--input", required=True, help="Input directory containing .tif files.")
    parser.add_argument("--output", required=True, help="Output directory mimicking the structural format.")
    
    args = parser.parse_args()
    
    process_directory(args.input, args.output)
