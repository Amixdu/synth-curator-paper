"""
Standardize Real Images.
Creates a temporary 512x512 Lanczos-resized version of the Real Dataset.
This ensures the metric pipeline uses the exact same pixel data as the training pipeline.
"""
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Adjust path to find utils
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.helpers import load_and_resolve_config

def center_crop_and_resize(img_path, resolution):
    """
    Center crop to the shorter side, then Lanczos resize, as in the EDM2 data preparation.
    """
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    
    # 1. Center Crop Logic (Min Dimension)
    crop = np.min(img_array.shape[:2])
    img_array = img_array[(img_array.shape[0] - crop) // 2 : (img_array.shape[0] + crop) // 2, 
                          (img_array.shape[1] - crop) // 2 : (img_array.shape[1] + crop) // 2]
    
    # 2. Lanczos Resize Logic
    img = Image.fromarray(img_array, 'RGB')
    img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_and_resolve_config(Path(args.config))

    # --- Read Paths ---
    # The raw original images
    source_dir = Path(config['raw_image_dir'])
    # The new destination for standardized images
    dest_dir = Path(config['standardized_dir'])
    
    resolution = 512 # Force 512 to match EDM2 Training

    print(f"Standardizing Images to {resolution}x{resolution} (Lanczos)...")
    print(f"Source: {source_dir}")
    print(f"Dest:   {dest_dir}")

    if dest_dir.exists():
        print("Destination exists. Skipping (Assume IDEMPOTENT).")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Gather Images
    exts = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for ext in exts:
        files.extend(list(source_dir.rglob(ext)))

    print(f"Processing {len(files)} images...")

    for f in tqdm(files):
        try:
            # Maintain relative structure if recursive, or flatten?
            # Flattening is safer for metric parsers usually.
            # Using unique filename to prevent collisions.
            clean_name = f"{f.stem}.png" 
            save_path = dest_dir / clean_name
            
            img = center_crop_and_resize(f, resolution)
            img.save(save_path)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print("Standardization Complete.")

if __name__ == "__main__":
    main()