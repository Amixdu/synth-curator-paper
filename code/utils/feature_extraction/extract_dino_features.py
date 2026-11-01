"""
Purpose:
    A utility script to perform feature extraction using the DINOv2 model on
    a folder of 512x512 images. The input folder (feature_input_dir) and the output
    file (output_features_pt) are given in the config file.

    This is an expensive, one-time pre-computation step. If the script is
    interrupted, it must be re-run from the beginning.

Usage:
    $ python utils/feature_extraction/extract_dino_features.py --config fsms/configs/extract_test_features.yml
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

try:
    from skimage import io
    from tqdm import tqdm
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from utils.helpers import load_and_resolve_config
except ImportError as e:
    print(f"Error: A required library is not installed. {e}", file=sys.stderr)
    print("Please create and activate the feature_extraction environment (code/environments/feature_extraction.yml).", file=sys.stderr)
    sys.exit(1)

# --- CONSTANTS ---
MODEL_NAME = "facebook/dinov2-base"

def extract_feature_vector(image_path: Path, model, processor, device):
    """Extracts the DINOv2 [CLS] token feature vector from a single image."""
    image = io.imread(str(image_path))
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0]
    return features.squeeze().cpu()

def main():
    parser = argparse.ArgumentParser(description=f"Extract DINOv2 features")
    parser.add_argument('--config', required=True, type=str, help="Path to the master YAML config file.")
    args = parser.parse_args()
    config_path = Path(args.config)
    
    # --- Model and Device Initialization ---
    print(f"Initializing DINOv2 model ({MODEL_NAME})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("Initialization complete.")
    
    cfg = load_and_resolve_config(config_path)
    
    try:
        # Read the input and output paths directly from the config.
        image_dir = Path(cfg['feature_input_dir'])
        output_path = Path(cfg['output_features_pt'])
    except KeyError as e:
        print(f"[FATAL ERROR] A required key is missing from your config: {e} (needs feature_input_dir and output_features_pt)", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing images from: {image_dir}")
    print(f"Saving features to:   {output_path}")

    supported_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    
    # Loop through each supported extension and collect all matching file paths.
    # The '**/' pattern ensures we search recursively through all subdirectories.
    for ext in supported_extensions:
        image_paths.extend(image_dir.glob(f'**/{ext}'))

    if not image_paths:
        print(f"[FATAL ERROR] No supported images (.jpg, .jpeg, .png) found in {image_dir} or its subdirectories.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images to process.")
    
    features_dict = {}
    for img_path in tqdm(image_paths, desc=f"Extracting DINOv2 Features"):
        try:
            vec = extract_feature_vector(img_path, model, processor, device)
            rel_path = str(img_path.relative_to(image_dir))
            features_dict[rel_path] = vec
        except Exception as e:
            print(f"\n[Warning] Skipping {img_path.name}: {e}", file=sys.stderr)

    # This single save operation happens only after the entire loop completes successfully.
    print("\nProcessing complete. Saving all features to disk...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features_dict, output_path)
    print(f"DINOv2 features for asset saved to: {output_path}")

if __name__ == "__main__":
    main()