"""
Shared evaluation and compilation for the metric tournament and for the algorithm
development runs. Run once per phase with a different --models_dir/--output_csv/--phase.

Evaluates against the real validation set. The held-out test set is reserved for the
final benchmark.

All needed .pth files (random control, real baseline, instance-level metrics, 30-seed
distributional models) are present in models_dir before this script runs (placed there by
organize_tournament_inputs.py for metric_tournament). Every model is evaluated fresh.

After a successful run, deletes the evaluated .pth files; only the final benchmark's
checkpoints are kept.

Usage:
    python eval_and_compile.py --phase metric_tournament \
        --config configs/eval.yml \
        --models_dir $SCRATCH/metric_tournament/models \
        --output_csv $SCRATCH/metric_tournament/tournament_results.csv

    python eval_and_compile.py --phase temperature_sweep \
        --config configs/eval.yml \
        --models_dir $SCRATCH/temperature_sweep/models \
        --output_csv $SCRATCH/temperature_sweep/tournament_results.csv
"""
import sys
import argparse
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from sklearn.metrics import (
        fbeta_score, f1_score, roc_auc_score, recall_score, precision_score, accuracy_score,
    )
    from utils.helpers import load_and_resolve_config
    from classifiers.evaluation import get_model, evaluate_model
except ImportError as e:
    print(f"[FATAL] {e}", file=sys.stderr)
    sys.exit(1)


def infer_condition_and_arch(filename: str):
    """Parses arch and condition from a .pth filename.

    All files must end with _resnet50.pth or _swin_b.pth - this convention is
    enforced by organize_tournament_inputs.py (which renames files during copy)
    and by the training scripts (which write checkpoints with this suffix).

    Returns (condition, arch), or (None, None) if the name doesn't match.
    Example: 'edm2_fid_inception_seed0_resnet50.pth' → ('edm2_fid_inception_seed0', 'resnet50')
    """
    stem = filename.replace('.pth', '')
    if stem.endswith('_resnet50'):
        return stem[:-len('_resnet50')], 'resnet50'
    elif stem.endswith('_swin_b'):
        return stem[:-len('_swin_b')], 'swin_b'
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help="Eval YAML (evaluation.*, data.validation_dir); see configs/eval.yml")
    parser.add_argument('--phase', required=True, choices=['metric_tournament', 'temperature_sweep'],
                        help="metric_tournament: evaluates all tournament conditions. "
                             "temperature_sweep: evaluates the algorithm development candidates only (temperature sweeps).")
    parser.add_argument('--models_dir', required=True,
                        help="Directory containing all .pth files to evaluate.")
    parser.add_argument('--output_csv', required=True,
                        help="Path to write the per-model results CSV.")
    parser.add_argument('--validation_dir', default=None,
                        help="ImageFolder-compatible real validation set directory.")
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--keep_checkpoints', action='store_true',
                        help="Skip deleting .pth files after eval (default: delete)")
    parser.add_argument('--raw_predictions_dir', default=None,
                        help="If set, also saves per-image raw predictions to "
                             "{raw_predictions_dir}/{condition}_{arch}_predictions.csv "
                             "(unique_image_id, true_label, predicted_label) - same schema "
                             "classifiers/evaluation.py already uses. Optional, off by default.")
    parser.add_argument('--only-missing', action='store_true',
                        help="Evaluate only checkpoints that have no row in --output_csv "
                             "yet, then merge the new rows into it instead of rewriting "
                             "the file from scratch. Turns a full re-score of every "
                             "checkpoint into a pass over just the new ones. Matching is "
                             "by (Dataset_Condition, Model_Architecture), so a checkpoint "
                             "RETRAINED under the same name is skipped, not re-scored - "
                             "delete its row (or drop the flag) when that happens.")
    args = parser.parse_args()

    # Config file sets defaults; CLI args override if also provided.
    cfg_ev, cfg_d = {}, {}
    if args.config:
        raw = load_and_resolve_config(Path(args.config))
        cfg_ev = raw.get('evaluation', {})
        cfg_d  = raw.get('data', {})

    val_dir     = args.validation_dir or cfg_d.get('validation_dir')
    num_classes = args.num_classes    or cfg_ev.get('num_classes')
    batch_size  = args.batch_size     or cfg_ev.get('batch_size', 64)
    num_workers = args.num_workers    or cfg_ev.get('num_workers',  4)

    if val_dir is None:
        print("[FATAL] --validation_dir required (or set data.validation_dir in --config)", file=sys.stderr)
        sys.exit(1)
    if num_classes is None:
        print("[FATAL] --num_classes required (or set evaluation.num_classes in --config)", file=sys.stderr)
        sys.exit(1)

    models_dir = Path(args.models_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    raw_predictions_dir = Path(args.raw_predictions_dir) if args.raw_predictions_dir else None
    if raw_predictions_dir:
        raw_predictions_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\n{args.phase.upper()} EVALUATION\nDevice: {device}\n{'='*60}\n")

    # Standard ImageNet normalisation - matches what all classifiers were trained with.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    dataloader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Validation set: {len(val_dataset)} images\n")

    if not models_dir.exists():
        print(f"[FATAL] models_dir does not exist: {models_dir}", file=sys.stderr)
        sys.exit(1)

    pth_files = sorted(models_dir.glob("*.pth"))
    print(f"Found {len(pth_files)} models in {models_dir}")

    # Incremental mode. The development models_dir grows by about 30 checkpoints per run
    # while the rest are unchanged, and re-scoring them costs hours for numbers that cannot
    # change. Skip anything already present in the output CSV and merge.
    existing_df = None
    if args.only_missing:
        if not output_csv.exists():
            print("[only-missing] no existing output CSV - evaluating everything.\n")
        else:
            existing_df = pd.read_csv(output_csv)
            have = set(zip(existing_df['Dataset_Condition'], existing_df['Model_Architecture']))
            keep = []
            for f in pth_files:
                cond, arch = infer_condition_and_arch(f.name)
                if cond is not None and (cond, arch) in have:
                    continue
                keep.append(f)
            print(f"[only-missing] {len(existing_df)} rows already scored; "
                  f"{len(keep)} of {len(pth_files)} checkpoints are new.")
            if not keep:
                print("[only-missing] nothing new to evaluate - output CSV left unchanged.\n")
                sys.exit(0)
            print("[only-missing] NOTE: matching is by name. A checkpoint retrained under an "
                  "existing name is skipped, not re-scored.\n")
            pth_files = keep
    else:
        print()

    results = []
    for i, pth_file in enumerate(pth_files):
        condition, arch = infer_condition_and_arch(pth_file.name)
        if condition is None:
            print(f"[{i+1}] [SKIP] Cannot parse arch from: {pth_file.name}")
            continue

        print(f"[{i+1}/{len(pth_files)}] {condition} ({arch})...", end=" ", flush=True)

        try:
            # get_model builds the architecture with a randomly-initialised classification head
            # (ResNet-50: replaces fc; Swin-B: replaces head). Weights_only=True is a PyTorch
            # safety flag that prevents arbitrary code execution from untrusted .pth files.
            model = get_model(arch, num_classes, device)
            model.load_state_dict(torch.load(pth_file, map_location=device, weights_only=True))

            # evaluate_model runs inference in eval mode (no dropout/BN updates) under no_grad.
            # Returns:
            #   probs       - softmax class probabilities, shape (N, num_classes), used for AUC
            #   preds       - argmax class predictions, shape (N,), used for F1/F2/accuracy
            #   true_labels - ground-truth class indices from ImageFolder, shape (N,)
            probs, preds, true_labels = evaluate_model(model, dataloader, device, num_classes)

            if raw_predictions_dir:
                filenames = [Path(s[0]).stem for s in val_dataset.samples]
                raw_df = pd.DataFrame({
                    'unique_image_id': filenames,
                    'true_label': true_labels,
                    'predicted_label': preds,
                })
                raw_df.to_csv(raw_predictions_dir / f"{condition}_{arch}_predictions.csv", index=False)
        except Exception as e:
            print(f"[ERROR] {e} - skipping.")
            continue
        finally:
            # Explicitly flush the CUDA memory pool after each model so fragmentation
            # doesn't accumulate across 300+ sequential loads and cause a late-run OOM.
            try:
                del model
            except NameError:
                pass
            torch.cuda.empty_cache()

        # Primary metric: Macro-F2 (beta=2 weights recall twice as heavily as precision -
        # appropriate for medical imaging where missing a cancer is worse than a false alarm).
        macro_f2        = fbeta_score(true_labels, preds, beta=2.0, average='macro', zero_division=0)
        macro_f1        = f1_score(true_labels, preds, average='macro', zero_division=0)
        macro_recall    = recall_score(true_labels, preds, average='macro', zero_division=0)
        macro_precision = precision_score(true_labels, preds, average='macro', zero_division=0)
        global_acc      = accuracy_score(true_labels, preds)
        try:
            # OvR AUC needs softmax probs (not logits) and at least one sample per class.
            # ValueError is raised if a class is absent from the validation batch - rare but possible.
            macro_auc = roc_auc_score(true_labels, probs, multi_class='ovr', average='macro')
        except ValueError:
            macro_auc = float('nan')

        results.append({
            'Model_Architecture': arch,
            'Dataset_Condition':  condition,
            'Global_Accuracy':    global_acc,
            'Macro_F1':           macro_f1,
            'Macro_F2':           macro_f2,
            'Macro_AUC':          macro_auc,
            'Macro_Recall':       macro_recall,
            'Macro_Precision':    macro_precision,
        })
        print(f"F2={macro_f2:.4f} F1={macro_f1:.4f} AUC={macro_auc:.4f}")

    combined_df = pd.DataFrame(results)
    if existing_df is not None:
        n_new = len(combined_df)
        combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
        # Guard against a row being counted twice if the CSV and models_dir ever disagree.
        before = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=['Dataset_Condition', 'Model_Architecture'], keep='last')
        if len(combined_df) != before:
            print(f"[only-missing] dropped {before - len(combined_df)} duplicate row(s).")
        print(f"[only-missing] merged {n_new} new rows into {len(existing_df)} existing.")
    combined_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv} ({len(combined_df)} rows)")

    if args.only_missing and not args.keep_checkpoints:
        print("\n[only-missing] refusing to delete checkpoints: the file list was filtered, "
              "so deletion would only remove the newly-scored ones and leave the rest.")
    elif not args.keep_checkpoints:
        if len(results) == 0:
            print("\n[WARNING] Zero models evaluated successfully - skipping checkpoint deletion "
                  "to avoid destroying data without results.")
        else:
            # Only the final benchmark's checkpoints need to persist for held-out test eval.
            print(f"\nDeleting {len(pth_files)} evaluated checkpoints...")
            for pth_file in pth_files:
                pth_file.unlink(missing_ok=True)
            print("Done.")


if __name__ == "__main__":
    main()
