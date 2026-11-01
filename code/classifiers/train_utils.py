"""
Shared training utilities for all classifier train scripts (resnet50, swin_b -
see classifiers/metric_tournament/).
"""
import copy
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer_param_groups(model, head_module, finetune, lr=None, backbone_lr=None, head_lr=None):
    """
    Returns AdamW-ready param groups (a list of {"params", "lr"} dicts) - every caller
    does `optim.AdamW(get_optimizer_param_groups(...))` uniformly regardless of mode.

    Linear-probe (finetune=False): freezes everything except head_module (the freshly
    replaced model.fc/model.head), one group at `lr` - identical behavior to every
    script before fine-tuning was added.

    Fine-tune (finetune=True): unfreezes the whole model, two groups - the pretrained
    backbone at `backbone_lr`, the freshly-initialized head at `head_lr` - so the
    backbone moves slowly while the new head (starting from random init) can move
    faster. head_module must already be the replacement layer (post nn.Linear swap).
    """
    if not finetune:
        for p in model.parameters():
            p.requires_grad = False
        for p in head_module.parameters():
            p.requires_grad = True
        return [{"params": list(head_module.parameters()), "lr": lr}]

    for p in model.parameters():
        p.requires_grad = True
    head_ids = {id(p) for p in head_module.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": list(head_module.parameters()), "lr": head_lr},
    ]


def stage_data_to_tmpdir(train_input_path: Path, val_dir_path: Path):
    """
    Copies data from specified data source to $TMPDIR for faster access during training.
    Handles both CSV manifests and standard directories.
    """
    from tqdm import tqdm

    tmpdir = os.environ.get('TMPDIR')
    job_id  = os.environ.get('SLURM_JOB_ID', 'local')
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')

    stage_base  = Path(tmpdir) / f"staging_{job_id}_{task_id}"
    train_stage = stage_base / 'train'
    val_stage   = stage_base / 'val'

    print(f"\n--- Staging Data to Local $TMPDIR ({stage_base}) ---")

    # 1. Stage Training Data
    if train_input_path.is_file() and train_input_path.suffix == '.csv':
        if not tmpdir:
            raise ValueError("TMPDIR environment variable is not set. Please set TMPDIR to a local directory for staging when using CSV input.")
        print(f"  -> Building Training Set from CSV Manifest...")
        df = pd.read_csv(train_input_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Copying Train Images"):
            src = Path(row['source_path'])
            dst_dir = train_stage / str(row['target_class'])
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_name = row.get('safe_filename', src.name)
            shutil.copy2(src, dst_dir / dst_name)
    elif train_input_path.is_dir():
        if not tmpdir:
            print("[WARNING] $TMPDIR not found. Falling back to direct reading.")
            return train_input_path, val_dir_path
        print(f"  -> Copying Training Directory Tree...")
        shutil.copytree(train_input_path, train_stage, dirs_exist_ok=True)
    else:
        raise ValueError(f"Invalid train_input_path: {train_input_path}")

    # 2. Stage Validation Data (Always a directory)
    print(f"  -> Copying Validation Directory Tree...")
    shutil.copytree(val_dir_path, val_stage, dirs_exist_ok=True)

    print("  -> Staging Complete.\n")
    return train_stage, val_stage


def train_model(model, criterion, optimizer, dataloaders, device, num_epochs, patience, output_model_path):
    """
    Main function to handle the training and validation loop.
    """
    import torch
    from tqdm import tqdm

    start_time      = time.time()
    best_model_wts  = copy.deepcopy(model.state_dict())
    best_acc        = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss     = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"  {phase.capitalize()} Phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc  = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'  {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), output_model_path)
                print(f"  ✨ New best model saved with validation accuracy: {best_acc:.4f}")
                epochs_no_improve = 0
            elif phase == 'val':
                epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break
        print()

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model
