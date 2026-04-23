import os
import argparse
from pathlib import Path
from predict import load_trained_model, predict_full_image
from config import PREDICTIONS_DIR

def process_directory(input_dir, output_dir, model):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Iterate through all .tif files in the input directory and subdirectories
    for tif_file in input_path.rglob('*.tif'):
        if tif_file.name.endswith('.xml'):
            continue
            
        print(f"\nProcessing: {tif_file}")
        
        # Calculate relative path
        rel_path = tif_file.relative_to(input_path)
        
        # Create corresponding output directory
        target_dir = output_path / rel_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Run prediction
        try:
            predict_full_image(
                model=model,
                sentinel_filepath=str(tif_file),
                output_dir=str(target_dir)
            )
            print(f"Successfully processed {tif_file.name}")
        except Exception as e:
            print(f"Failed to process {tif_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process GeoTIFF files for LULC prediction")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing .tif files")
    parser.add_argument("--output", "-o", type=str, default=PREDICTIONS_DIR, help="Output directory for predictions")
    args = parser.parse_args()
    
    print("Loading model...")
    model = load_trained_model()
    
    print(f"Starting batch prediction from {args.input} to {args.output}")
    process_directory(args.input, args.output, model)
    print("Batch processing complete.")