"""
Pack feature dicts into contiguous matrices.

extract_dino_features.py writes {filename: tensor[768]}. For the 275,472-image pool
that is 275k separate tensor objects: ~1.5 GB of Python overhead on top of ~850 MB of
data, and several seconds just to load. Packing to {'names': [...], 'matrix': [N, D]}
drops peak RAM to roughly the data size and makes load near-instant.

Usage:
    python fsms/scripts/pack_features.py --config fsms/configs/pack_features.yml
"""
import sys
import argparse
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from utils.helpers import load_and_resolve_config


def pack(src: Path, dst: Path) -> bool:
    """Pack one feature file. Returns True if written, False if the source is missing."""
    if not src.exists():
        print(f"  [SKIP] not found: {src}")
        return False

    print(f"\n  {src.name}")
    obj = torch.load(src, map_location='cpu', weights_only=False)

    if isinstance(obj, dict) and 'names' in obj and 'matrix' in obj:
        print("    already packed - copying through unchanged")
        names, matrix = list(obj['names']), obj['matrix'].float()
    else:
        names = list(obj.keys())
        matrix = torch.stack([obj[k] for k in names]).float()

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'names': names, 'matrix': matrix}, dst)

    src_mb, dst_mb = src.stat().st_size / 1e6, dst.stat().st_size / 1e6
    shrink = f"  [{src_mb / dst_mb:.0f}x smaller]" if src_mb > dst_mb * 1.5 else ""
    print(f"    {len(names)} x {matrix.shape[1]}  {src_mb:.0f} MB -> {dst_mb:.0f} MB{shrink}")
    print(f"    wrote {dst}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_and_resolve_config(Path(args.config))

    print("=" * 70)
    print("PACK FEATURES")
    print("=" * 70)

    packed, skipped = 0, 0
    for entry in cfg['pack']:
        if pack(Path(entry['input']), Path(entry['output'])):
            packed += 1
        else:
            skipped += 1

    print(f"\nDone - {packed} packed" + (f", {skipped} skipped." if skipped else "."))
    if skipped:
        print("  Skipped inputs are missing; rerun once they are in place.")
    print("  Next: validate.py and calibrate_threshold.py read these via their configs.")
    print("  Make sure its packed_dir matches the one in this config.")


if __name__ == '__main__':
    main()
