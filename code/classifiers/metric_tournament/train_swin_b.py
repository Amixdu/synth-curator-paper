"""
Swin-B train script for the metric tournament, invoked via CLI (one run per
curated set and seed). main() handles arg parsing and config
loading, then delegates to run_training(params, output_model_path). Shared training
utilities (set_seed, stage_data_to_tmpdir, train_model) live in
classifiers/train_utils.py.
"""
import sys
import argparse
import random
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms
    from utils.helpers import load_and_resolve_config
except ImportError as e:
    print(f"Error: A required library is not installed. {e}", file=sys.stderr)
    sys.exit(1)

import numpy as np
from classifiers.train_utils import (
    set_seed, stage_data_to_tmpdir, train_model, get_optimizer_param_groups,
)


def run_training(params: dict, output_model_path: Path):
    """
    Runs one full Swin-B training experiment from a plain params dict - the
    direct-call equivalent of the original script's main(), minus config-file loading.

    Required keys: input_training_dir, input_validation_dir, num_classes, batch_size,
    num_epochs, learning_rate, early_stopping_patience, num_workers. Optional: seed (42).
    """
    orig_train_input = Path(params['input_training_dir'])
    orig_val_input    = Path(params['input_validation_dir'])
    num_classes  = params['num_classes']
    batch_size   = params['batch_size']
    num_epochs   = params['num_epochs']
    lr           = params['learning_rate']
    patience     = params['early_stopping_patience']
    num_workers  = params['num_workers']
    seed         = params.get('seed', 42)
    finetune     = params.get('finetune', False)
    backbone_lr  = params.get('backbone_lr')
    head_lr      = params.get('head_lr')

    set_seed(seed)

    image_dir_train, image_dir_val = stage_data_to_tmpdir(orig_train_input, orig_val_input)
    print(f"  Training Data Dir:   {image_dir_train}")
    print(f"  Validation Data Dir: {image_dir_val}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    train_dataset = datasets.ImageFolder(image_dir_train, data_transforms['train'])
    val_dataset   = datasets.ImageFolder(image_dir_val,   data_transforms['val'])

    g = torch.Generator()
    g.manual_seed(seed)

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
    print(f"Data loaded: {len(train_dataset)} training images, {len(val_dataset)} validation images.")

    mode_desc = "Fine-Tuning" if finetune else "Transfer Learning"
    print(f"\n--- Setting up Swin-B Model ({mode_desc}) ---")
    model = models.swin_b(weights='IMAGENET1K_V1')
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    param_groups = get_optimizer_param_groups(
        model, model.head, finetune, lr=lr, backbone_lr=backbone_lr, head_lr=head_lr)
    print("Model setup complete. " +
          ("Whole backbone unfrozen." if finetune else "Final head is unfrozen."))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(param_groups)

    print("\n--- Starting Training ---")
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    train_model(model, criterion, optimizer, dataloaders, device, num_epochs, patience, output_model_path)
    print(f"\nBest model saved to: {output_model_path}")


DEFAULTS = {
    "input_validation_dir":    "/path/to/datasets/main/real/split/Real_Validation",
    "num_classes":             7,
    "learning_rate":           0.001,
    "batch_size":              32,
    "num_epochs":              50,
    "early_stopping_patience": 10,
    "num_workers":             4,
    "seed":                    42,
    "finetune":                False,
    "backbone_lr":             None,
    "head_lr":                 None,
}


def main():
    parser = argparse.ArgumentParser(description="Train one Swin-B model (metric tournament and algorithm development).")
    parser.add_argument('--config', default=None,
                        help="Training YAML (training.*, data.real_val). Overrides DEFAULTS; "
                             "see configs/training/swin_b.yml")
    parser.add_argument('--input_training_dir', required=True, help="Curated CSV manifest or ImageFolder dir")
    parser.add_argument('--output_model_path', required=True)
    args = parser.parse_args()

    output_model_path = Path(args.output_model_path)
    if output_model_path.exists():
        output_model_path.unlink()

    params = dict(DEFAULTS)
    if args.config:
        cfg = load_and_resolve_config(Path(args.config))
        t = cfg.get('training', {})
        d = cfg.get('data', {})
        for key in ('num_classes', 'batch_size', 'num_epochs', 'learning_rate',
                    'early_stopping_patience', 'num_workers', 'seed',
                    'finetune', 'backbone_lr', 'head_lr'):
            if key in t:
                params[key] = t[key]
        if 'real_val' in d:
            params['input_validation_dir'] = d['real_val']
    params['input_training_dir'] = args.input_training_dir
    run_training(params, output_model_path)


if __name__ == "__main__":
    main()
