"""
Organizes all .pth files needed for the metric tournament into one unified models_dir,
so eval_and_compile.py can evaluate everything in a single fresh pass.

Sources (the --src folder of trained models):
  - Instance-level metric models (both archs) - copied as-is, already correctly named
  - 30-seed random control models (both archs) - renamed to put arch at the end
  - Real-only baseline (both archs)             - renamed to put arch at the end

Skips:
  - Single-seed distributional metric models (fid/kid/swd), replaced by the 30-seed models
    already present in models_dir from the training step

Usage:
    python organize_tournament_inputs.py \
        --src  $SCRATCH/trained_models \
        --dst  $SCRATCH/metric_tournament/models
"""
import sys
import shutil
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

INSTANCE_METRICS = [
    'psnr_score', 'ssim_score', 'fsim_score', 'lpips_score', 'brisqe_score', 'fscs_score',
    'clip_similarity_cancer', 'clip_similarity_artifact', 'clip_log_odds_score',
    'diversity_score', 'realism_score', 'is_contribution_score',
]
DISTRIBUTIONAL_METRICS = ['fid_inception', 'fid_dino', 'kid_dino', 'swd_dino']
ARCHS = ['resnet50', 'swin_b']
RANDOM_SEEDS = range(42, 72)  # seeds 42-71


def copy(src: Path, dst: Path, label: str, found: list, missing: list):
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  [OK]      {label}")
        found.append(label)
    else:
        print(f"  [MISSING] {label}  (src: {src})")
        missing.append(label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help="classification/assets dir on scratch")
    parser.add_argument('--dst', required=True, help="metric_tournament/models dir on scratch")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    found, missing = [], []

    print("\n=== Real-only baselines ===")
    for arch in ARCHS:
        old_name = f"{arch}_real_only_baseline.pth"
        new_name = f"real_only_baseline_{arch}.pth"
        copy(src / old_name, dst / new_name, new_name, found, missing)

    print("\n=== Random control (30 seeds) ===")
    for seed in RANDOM_SEEDS:
        for arch in ARCHS:
            old_name = f"edm2_{arch}_random_control_seed{seed}.pth"
            new_name = f"edm2_random_control_seed{seed}_{arch}.pth"
            copy(src / old_name, dst / new_name, new_name, found, missing)

    print("\n=== Instance-level metric models ===")
    for metric in INSTANCE_METRICS:
        for arch in ARCHS:
            fname = f"edm2_{metric}_{arch}.pth"
            copy(src / fname, dst / fname, fname, found, missing)

    print(f"\n{'='*60}")
    print(f"Found:   {len(found)}")
    print(f"Missing: {len(missing)}")
    if missing:
        print("\nMissing files (likely deleted by a previous eval run):")
        for m in missing:
            print(f"  - {m}")
        print("\nThese conditions will be absent from the fresh tournament.")
        print("If this is not acceptable, they need to be retrained.")

    existing_30seed = [f for f in dst.glob("*.pth")
                       if any(m in f.name for m in DISTRIBUTIONAL_METRICS)]
    print(f"\n30-seed distributional models already in dst: {len(existing_30seed)}")
    print(f"Total .pth files now in dst: {len(list(dst.glob('*.pth')))}")
    print("\nDone. Run eval_and_compile.py next.")


if __name__ == "__main__":
    main()
