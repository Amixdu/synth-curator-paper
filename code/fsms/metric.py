"""
FSMS (Feature-Space Memorization Score) - core library.

FSMS = 1 - mean(max_cosine_similarity_to_nearest_real_image)
Higher = safer (synthetic images are further from any real training image).

Feature files are torch.load() dicts keyed by filename string.
No GPU required - pure batched cosine similarity on CPU.

Imported by:
  benchmark/scripts/2_compute_fsms.py
"""
from pathlib import Path

import pandas as pd
import torch

BATCH_SIZE = 512


def load_feature_dict(path: Path) -> dict:
    """Load a {filename: tensor} feature dict saved with torch.save()."""
    print(f"  Loading features: {path}")
    d = torch.load(path, map_location='cpu', weights_only=False)
    print(f"  {len(d)} entries loaded.")
    return d


def build_real_class_lookup(real_train_dir: Path) -> dict:
    """
    Build {filename_stem: class_name} by scanning Real_Train's class
    subfolders directly (AKIEC/, BCC/, ...).

    Needed because real_train_features.pt's keys are bare filenames, not
    '{class}/{filename}': standardize_images.py flattens the class subfolder
    away before DINOv2 feature extraction ever sees the images, so class
    info can't be recovered from the feature dict's keys and has to come
    from the untouched, still-class-subfoldered source directory instead.
    Matches by stem (not full filename) since standardization always
    re-saves as '{stem}.png' regardless of the original extension.
    """
    real_train_dir = Path(real_train_dir)
    lookup = {}
    for class_dir in sorted(p for p in real_train_dir.iterdir() if p.is_dir()):
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            for img_path in class_dir.glob(ext):
                lookup[img_path.stem] = class_dir.name
    return lookup


def _build_safe_key_map(feat_dict: dict) -> dict:
    """
    Build {safe_filename → original_key} for feat_dict.

    Feature dict keys use forward slashes (e.g. 'ckpt_25k/ISIC_xxx.jpg').
    CSV safe_filename replaces '/' with '_' (e.g. 'ckpt_25k_ISIC_xxx.jpg').
    This map lets us resolve safe_filename → original key for lookup.
    """
    return {k.replace('/', '_').replace('\\', '_'): k for k in feat_dict}


def compute_fsms_for_csv(csv_path: Path, synth_feat_dict: dict, real_feat_matrix: torch.Tensor) -> float:
    """
    Compute FSMS for one curated dataset CSV.

    CSV must have a 'safe_filename' column (or 'source_path' as fallback) that
    identifies each image in synth_feat_dict.

    Returns FSMS in [0, 1], or NaN if no features can be matched.
    """
    df = pd.read_csv(csv_path)

    # Resolve lookup keys
    if 'safe_filename' in df.columns:
        filenames = df['safe_filename'].tolist()
        # safe_filename has '/' replaced with '_'; feature dict keys use '/'.
        # Build reverse map once so lookup is O(1) per image.
        safe_to_orig = _build_safe_key_map(synth_feat_dict)
        resolve = lambda fn: safe_to_orig.get(fn, fn)
    elif 'source_path' in df.columns:
        filenames = ['/'.join(Path(p).parts[-2:]) for p in df['source_path']]
        resolve = lambda fn: fn
    else:
        print(f"    [ERROR] CSV has no 'safe_filename' or 'source_path' column: {csv_path.name}")
        return float('nan')

    synth_vecs = []
    missing = 0
    for fn in filenames:
        key = resolve(fn)
        if key in synth_feat_dict:
            synth_vecs.append(synth_feat_dict[key])
        else:
            missing += 1

    if missing > 0:
        print(f"    [WARNING] {missing}/{len(filenames)} images not found in feature dict.")

    if not synth_vecs:
        print(f"    [ERROR] No features found for {csv_path.name}. Returning NaN.")
        return float('nan')

    synth_matrix = torch.stack(synth_vecs)  # [N_synth, D]

    synth_norm = synth_matrix / (synth_matrix.norm(dim=1, keepdim=True) + 1e-8)
    real_norm  = real_feat_matrix / (real_feat_matrix.norm(dim=1, keepdim=True) + 1e-8)

    max_sims = []
    for i in range(0, len(synth_norm), BATCH_SIZE):
        batch = synth_norm[i:i + BATCH_SIZE]
        sims  = batch @ real_norm.T  # [batch, N_real]
        max_sims.append(sims.max(dim=1).values)

    max_sims_all = torch.cat(max_sims)
    return 1.0 - max_sims_all.mean().item()
