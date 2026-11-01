"""
FSMS (Feature-Space Memorization Score) computation

For each curated dataset CSV, computes how close the synthetic images in that
dataset are to the real training set in DINOv2 feature space.

FSMS = 1 - mean(max_cosine_similarity_to_nearest_real_image)
Higher FSMS = safer (synthetic images are further from any real training image).

Feature files are torch.load() dicts keyed by 'filename' column values.
No GPU required - pure KNN in feature space.

Output: {output.fsms_lookup}
    columns: condition, arch, seed, fsms_score

Usage:
    python 2_compute_fsms.py --config ../configs/eval/compute_fsms.yml
"""
import sys
import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

try:
    import torch
    from utils.helpers import load_and_resolve_config
    from fsms.metric import load_feature_dict, compute_fsms_for_csv
except ImportError as e:
    print(f"[FATAL] {e}", file=sys.stderr)
    sys.exit(1)


def enumerate_curated_csvs(cfg: dict):
    """
    Yields (condition_label, arch, seed_str, csv_path) for every curated CSV.

    Config schema:
        curated_dirs:
          shared:    {condition_prefix: dir}              # one CSV serves both archs
          per_arch:  {condition_prefix: {arch: dir}}       # separate CSV per arch
    """
    dirs = cfg['curated_dirs']

    for prefix, shared_dir in dirs.get('shared', {}).items():
        shared_dir = Path(shared_dir)
        for csv_path in sorted(shared_dir.glob("seed_*.csv")):
            i = re.search(r'seed_(\d+)', csv_path.stem).group(1)
            for arch in ['resnet50', 'swin_b']:
                yield f"{prefix}_seed_{i}", arch, i, csv_path

    for prefix, arch_dirs in dirs.get('per_arch', {}).items():
        for arch, arch_dir in arch_dirs.items():
            arch_dir = Path(arch_dir)
            for csv_path in sorted(arch_dir.glob("seed_*.csv")):
                i = re.search(r'seed_(\d+)', csv_path.stem).group(1)
                yield f"{prefix}_seed_{i}", arch, i, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_and_resolve_config(Path(args.config))

    real_feat_path  = Path(cfg['real_train_features'])
    synth_feat_path = Path(cfg['synth_pool_features'])
    fsms_out        = Path(cfg['output_fsms'])
    fsms_out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FSMS COMPUTATION")
    print("=" * 60)

    real_feat_dict  = load_feature_dict(real_feat_path)
    synth_feat_dict = load_feature_dict(synth_feat_path)

    # Stack real features into a single matrix [N_real, D]
    real_feat_matrix = torch.stack(list(real_feat_dict.values()))
    print(f"\nReal feature matrix: {real_feat_matrix.shape}")

    rows = []
    total = 0
    for condition, arch, seed, csv_path in enumerate_curated_csvs(cfg):
        total += 1
        print(f"\n[{total}] {condition} | arch={arch} | {csv_path.name}")
        fsms = compute_fsms_for_csv(csv_path, synth_feat_dict, real_feat_matrix)
        print(f"  FSMS = {fsms:.6f}")
        rows.append({'condition': condition, 'arch': arch, 'seed': seed, 'fsms_score': fsms})

    df = pd.DataFrame(rows)
    df.to_csv(fsms_out, index=False)

    print(f"\n{'='*60}")
    print(f"Saved {len(df)} FSMS scores to: {fsms_out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
