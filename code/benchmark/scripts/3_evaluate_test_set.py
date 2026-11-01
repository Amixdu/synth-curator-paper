"""
Test-set evaluation

Evaluates all trained models against the held-out real ISIC-2018 test set.
The test set is a flat image directory + one-hot label CSV (not ImageFolder format),
so this script uses a custom Dataset class.

Architecture is inferred from the .pth filename: contains 'resnet50' or 'swin_b'.
Condition label is the filename stripped of '_{arch}.pth'.

Outputs:
    {output.raw_predictions}/{condition}_{arch}_predictions.csv
        columns: unique_image_id, true_label, predicted_label, prob_<CLASS> (one per class)
    {output.final_results_raw}
        columns: condition, cohort, arch, seed, n_value, macro_f2, macro_f1,
                 macro_auc, macro_recall, macro_precision, global_acc

Usage:
    python 3_evaluate_test_set.py --config ../configs/eval/evaluate_test_set.yml
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image
    from sklearn.metrics import (
        fbeta_score, f1_score, roc_auc_score, accuracy_score,
        recall_score, precision_score,
    )
    from utils.helpers import load_and_resolve_config
except ImportError as e:
    print(f"[FATAL] {e}", file=sys.stderr)
    sys.exit(1)

# ISIC-2018 alphabetical class order (matches torchvision ImageFolder convention) - do not reorder.
CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']


class TestSetDataset(Dataset):
    """
    Flat test image directory + one-hot label CSV.
    CSV columns: image, MEL, NV, BCC, AKIEC, BKL, DF, VASC (+ ISIC_ prefix on image IDs)
    """
    def __init__(self, image_dir: Path, labels_csv: Path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(labels_csv)
        # The CSV has one-hot columns; convert to a single integer label
        class_cols = [c for c in CLASS_NAMES if c in df.columns]
        if not class_cols:
            raise ValueError(f"None of {CLASS_NAMES} found as columns in {labels_csv}")

        self.image_ids = []
        self.labels    = []
        for _, row in df.iterrows():
            img_id = str(row.iloc[0])  # first column is image ID (e.g. ISIC_0024306)
            label  = int(np.argmax([row[c] for c in class_cols]))
            self.image_ids.append(img_id)
            self.labels.append(label)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id  = self.image_ids[idx]
        label   = self.labels[idx]
        # Try .jpg first, then .png
        for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
            img_path = self.image_dir / f"{img_id}{ext}"
            if img_path.exists():
                break
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, img_id


def get_model(arch: str, num_classes: int, device: torch.device) -> nn.Module:
    if arch == 'resnet50':
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'swin_b':
        model = models.swin_b()
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model.to(device)


def infer_condition_and_arch(filename: str):
    """
    Expected naming convention: {condition}_{arch}.pth
    e.g. real_only_resnet50.pth, random_synth_seed_0_swin_b.pth
    """
    stem = filename.replace('.pth', '')
    if stem.endswith('_resnet50'):
        return stem[:-len('_resnet50')], 'resnet50'
    elif stem.endswith('_swin_b'):
        return stem[:-len('_swin_b')], 'swin_b'
    return None, None


def evaluate_model(model, dataloader, device):
    model.eval()
    all_probs, all_preds, all_labels, all_ids = [], [], [], []
    with torch.no_grad():
        for inputs, labels, img_ids in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs   = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_ids.extend(list(img_ids))
    return (
        np.concatenate(all_probs,  axis=0),
        np.concatenate(all_preds,  axis=0),
        np.concatenate(all_labels, axis=0),
        all_ids,
    )


def parse_seed_from_condition(condition: str) -> str:
    """Extract seed index from condition label, e.g. 'random_synth_seed_5' → '5'. 'real_only' → 'N/A'."""
    if '_seed_' in condition:
        return condition.split('_seed_')[-1]
    return 'N/A'


def parse_cohort_from_condition(condition: str) -> str:
    """Extract cohort name, e.g. 'algorithm_seed_5' → 'algorithm'."""
    for cohort in [
        'real_only',
        'random_synth_n20000',
        'random_synth',
        'best_metric',
        'algorithm_n20000',
        'algorithm_safety090',  # safety-sweep legs - must precede plain 'algorithm' below
        'algorithm_safety093',
        'algorithm_safety099',
        'algorithm',
    ]:
        if condition.startswith(cohort):
            return cohort
    return condition


def parse_n_value_from_condition(condition: str) -> int:
    """Extract synthetic volume N, e.g. 'algorithm_n20000_seed_5' → 20000. Default 8012."""
    return 20000 if 'n20000' in condition else 8012



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_and_resolve_config(Path(args.config))

    models_dir       = Path(cfg['models_dir'])
    test_images_dir  = Path(cfg['test_images_dir'])
    test_labels_csv  = Path(cfg['test_labels_csv'])
    raw_preds_dir    = Path(cfg['raw_predictions_dir'])
    results_csv      = Path(cfg['output_csv'])

    raw_preds_dir.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("FINAL TOURNAMENT - held-out test set")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset    = TestSetDataset(test_images_dir, test_labels_csv, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.get('batch_size', 64), shuffle=False, num_workers=cfg.get('num_workers', 4))
    print(f"Test set: {len(dataset)} images\n")

    pth_files = sorted(models_dir.glob("*.pth"))
    print(f"Found {len(pth_files)} models in {models_dir}\n")

    results = []

    for i, pth_file in enumerate(pth_files):
        condition, arch = infer_condition_and_arch(pth_file.name)
        if condition is None:
            print(f"[{i+1}] [SKIP] Cannot parse arch from: {pth_file.name}")
            continue

        print(f"[{i+1}/{len(pth_files)}] {condition} ({arch})...", end=" ", flush=True)

        preds_csv = raw_preds_dir / f"{condition}_{arch}_predictions.csv"

        model = get_model(arch, len(CLASS_NAMES), device)
        model.load_state_dict(torch.load(pth_file, map_location=device, weights_only=True))

        probs, preds, true_labels, img_ids = evaluate_model(model, dataloader, device)

        # Save raw predictions + per-class probabilities for downstream analysis
        # and any future probability-dependent analysis (calibration, ROC curves,
        # per-class AUC, confidence-based subgroup checks) without re-running eval.
        preds_df = pd.DataFrame({
            'unique_image_id':  img_ids,
            'true_label':       true_labels,
            'predicted_label':  preds,
        })
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            preds_df[f'prob_{cls_name}'] = probs[:, cls_idx]
        preds_df.to_csv(preds_csv, index=False)

        # Metrics
        macro_f2        = fbeta_score(true_labels, preds, beta=2.0, average='macro', zero_division=0)
        macro_f1        = f1_score(   true_labels, preds, average='macro',           zero_division=0)
        macro_recall    = recall_score(   true_labels, preds, average='macro', zero_division=0)
        macro_precision = precision_score(true_labels, preds, average='macro', zero_division=0)
        global_acc      = accuracy_score(true_labels, preds)
        try:
            macro_auc = roc_auc_score(true_labels, probs, multi_class='ovr', average='macro')
        except ValueError:
            macro_auc = float('nan')

        cohort  = parse_cohort_from_condition(condition)
        seed    = parse_seed_from_condition(condition)
        n_value = parse_n_value_from_condition(condition)

        results.append({
            'condition':       condition,
            'cohort':          cohort,
            'arch':            arch,
            'seed':            seed,
            'n_value':         n_value,
            'macro_f2':        macro_f2,
            'macro_f1':        macro_f1,
            'macro_auc':       macro_auc,
            'macro_recall':    macro_recall,
            'macro_precision': macro_precision,
            'global_acc':      global_acc,
        })

        print(f"F2={macro_f2:.4f}  F1={macro_f1:.4f}  AUC={macro_auc:.4f}  "
              f"Recall={macro_recall:.4f}  Precision={macro_precision:.4f}")

        del model

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(['cohort', 'arch', 'seed'])
    df_results.to_csv(results_csv, index=False)

    print(f"\n{'='*60}")
    print(f"Saved: {results_csv}  ({len(results)} rows)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
