"""
Unified real+synthetic training script (benchmark)

Trains ResNet-50 or Swin-B on real data alone, or real + a synthetic CSV manifest.
Model variance is locked: seed=42, deterministic CUDA. All spread across seeds is
pure data variance from the curated CSV.

The real training data is an ImageFolder directory. The synthetic images are one curated set
of a released selection file (data/benchmark/curated/selections_<cohort>.csv.gz), picked by
--seed; --pool_dir is the folder that holds the ckpt_*_cfg_* folders of the synthetic pool.
Both are staged into $TMPDIR/combined_train/ as a single merged ImageFolder.

Usage:
    # real_only
    python 1_train_augmented.py --config ../configs/train/resnet50.yml \
        --condition real_only

    # augmented
    python 1_train_augmented.py --config ../configs/train/resnet50.yml \
        --condition random_synth_seed_0 \
        --selections ../../../data/benchmark/curated/selections_random_synth.csv.gz \
        --seed 0 --pool_dir /path/to/synthetic_pool
"""
import sys
import argparse
import shutil
import random
import os
import json
import time
import copy
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms
    from tqdm import tqdm
    from utils.helpers import load_and_resolve_config
    from classifiers.train_utils import get_optimizer_param_groups
except ImportError as e:
    print(f"[FATAL] Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

SEED = 42  # locked - do not make this a CLI argument


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']


def load_selection(selections_csv: Path, seed: int, arch: str, pool_dir: Path) -> pd.DataFrame:
    """One curated set from a selections file as a manifest (source_path, target_class, safe_filename).
    The file holds the columns seed, filename and, for cohorts curated per classifier, arch.
    The class is the class_<n> part of the filename."""
    d = pd.read_csv(selections_csv)
    d = d[d.seed == seed]
    if 'arch' in d:
        d = d[d.arch == arch]
    names = d.filename.reset_index(drop=True)
    return pd.DataFrame({
        'source_path': str(pool_dir).rstrip('/') + '/' + names,
        'target_class': names.str.extract(r'/class_(\d)/')[0].astype(int).map(dict(enumerate(CLASS_NAMES))),
        'safe_filename': 'edm2_' + names.str.replace('/', '_', regex=False),
    })


def stage_combined(real_train_dir: Path, manifest, tmpdir: Path) -> Path:
    """
    Stages real (ImageFolder) + optional synthetic (manifest DataFrame) into a single
    merged ImageFolder at tmpdir/combined_train/.
    """
    combined = tmpdir / "combined_train"
    combined.mkdir(parents=True, exist_ok=True)

    # Stage real data
    print("  Staging real data...")
    shutil.copytree(real_train_dir, combined, dirs_exist_ok=True)
    print(f"  Real data staged: {real_train_dir}")

    # Stage synthetic data (if provided)
    if manifest is not None:
        print("  Staging synthetic data...")
        df = manifest
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Copying synthetic images"):
            src  = Path(row['source_path'])
            dst_dir = combined / str(row['target_class'])
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_name = row.get('safe_filename', src.name)
            shutil.copy2(src, dst_dir / dst_name)
        print(f"  Synthetic data staged: {len(df)} images")

    return combined


def stage_val(real_val_dir: Path, tmpdir: Path) -> Path:
    """Stages validation directory (ImageFolder) to TMPDIR."""
    val_stage = tmpdir / "val"
    print("  Staging validation data...")
    shutil.copytree(real_val_dir, val_stage, dirs_exist_ok=True)
    return val_stage


def build_model(arch: str, num_classes: int, device: torch.device,
                 finetune: bool = False, lr=None, backbone_lr=None, head_lr=None):
    if arch == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head_module = model.fc
    elif arch == 'swin_b':
        model = models.swin_b(weights='IMAGENET1K_V1')
        model.head = nn.Linear(model.head.in_features, num_classes)
        head_module = model.head
    else:
        raise ValueError(f"Unknown arch: {arch}")
    model = model.to(device)
    optimizer_param_groups = get_optimizer_param_groups(
        model, head_module, finetune, lr=lr, backbone_lr=backbone_lr, head_lr=head_lr)
    return model, optimizer_param_groups


def train(model, criterion, optimizer, dataloaders, device, num_epochs, patience, output_path):
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    no_improve = 0
    start = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_correct = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"  {phase}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss    += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data)

            epoch_acc = running_correct.double() / len(dataloaders[phase].dataset)
            print(f"  {phase} Acc: {epoch_acc:.4f}")

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_weights = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), output_path)
                    print(f"  New best val acc: {best_acc:.4f} - model saved.")
                    no_improve = 0
                else:
                    no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement.")
            break

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed//60:.0f}m {elapsed%60:.0f}s | Best val acc: {best_acc:.4f}")
    model.load_state_dict(best_weights)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',        required=True,  help="Path to configs/train/resnet50.yml or swin_b.yml")
    parser.add_argument('--condition',     required=True,  help="Experiment label used for output filename, e.g. 'real_only' or 'random_synth_seed_0'")
    parser.add_argument('--selections',    default=None,   help="Selections file of the cohort (omit for real_only)")
    parser.add_argument('--seed',          type=int, default=None, help="Curation seed of the set to use from --selections")
    parser.add_argument('--pool_dir',      default=None,   help="Folder that holds the ckpt_*_cfg_* folders of the synthetic pool")
    args = parser.parse_args()

    cfg = load_and_resolve_config(Path(args.config))
    t   = cfg['training']

    arch         = t['arch']
    num_classes  = t['num_classes']
    batch_size   = t['batch_size']
    num_epochs   = t['num_epochs']
    lr           = t['learning_rate']
    patience     = t['early_stopping_patience']
    num_workers  = t['num_workers']
    finetune     = t.get('finetune', False)
    backbone_lr  = t.get('backbone_lr')
    head_lr      = t.get('head_lr')
    real_train   = Path(cfg['data']['real_train'])
    real_val     = Path(cfg['data']['real_val'])
    models_dir   = Path(cfg['output']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)

    output_path  = models_dir / f"{args.condition}_{arch}.pth"

    if output_path.exists():
        print(f"[SKIP] Model already exists: {output_path}")
        return

    manifest = None
    if args.selections:
        if args.seed is None or not args.pool_dir:
            raise SystemExit("--selections needs --seed and --pool_dir")
        manifest = load_selection(Path(args.selections), args.seed, arch, Path(args.pool_dir))

    print("=" * 60)
    print(f"FINAL EXPERIMENT TRAINING")
    print(f"  Arch:      {arch}")
    print(f"  Mode:      {'fine-tune' if finetune else 'linear-probe'}")
    print(f"  Condition: {args.condition}")
    print(f"  Synthetic: {f'{args.selections} seed {args.seed}' if args.selections else 'None (real_only)'}")
    print(f"  Output:    {output_path}")
    print("=" * 60)

    set_seed(SEED)

    # Stage data
    tmpdir_env = os.environ.get('TMPDIR')
    if not tmpdir_env:
        raise RuntimeError("$TMPDIR not set. Cannot stage data.")
    job_id  = os.environ.get('SLURM_JOB_ID', 'local')
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    tmpdir  = Path(tmpdir_env) / f"final_exp_{job_id}_{task_id}"
    tmpdir.mkdir(parents=True, exist_ok=True)

    train_dir = stage_combined(real_train, manifest, tmpdir)
    val_dir   = stage_val(real_val, tmpdir)

    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, train_tf)
    val_dataset   = datasets.ImageFolder(val_dir,   val_tf)
    print(f"Dataset sizes - train: {len(train_dataset)}, val: {len(val_dataset)}")

    g = torch.Generator()
    g.manual_seed(SEED)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, worker_init_fn=seed_worker, generator=g),
        'val':   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, worker_init_fn=seed_worker, generator=g),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model, optimizer_param_groups = build_model(
        arch, num_classes, device, finetune=finetune, lr=lr,
        backbone_lr=backbone_lr, head_lr=head_lr)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(optimizer_param_groups)

    train(model, criterion, optimizer, dataloaders, device, num_epochs, patience, output_path)

    print(f"\n[DONE] Saved: {output_path}")


if __name__ == "__main__":
    main()
