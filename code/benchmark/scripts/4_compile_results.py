"""
Final results compilation

Joins two data sources into the final results table:
  1. final_results_raw.csv - test metrics per model (from 3_evaluate_test_set.py)
  2. fsms_lookup.csv - FSMS safety score per curated dataset (from 2_compute_fsms.py)

Output: {output_csv}
    columns: condition, cohort, arch, seed, n_value, macro_f2, macro_f1,
             macro_auc, macro_recall, macro_precision, global_acc, fsms_score

Usage:
    python 4_compile_results.py --config ../configs/eval/compile_results.yml
"""
import sys
import argparse
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from utils.helpers import load_and_resolve_config


def load_results_raw(results_raw_path: Path) -> pd.DataFrame:
    if not results_raw_path.exists():
        print(f"[FATAL] Results not found: {results_raw_path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(results_raw_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_and_resolve_config(Path(args.config))

    results_raw_path = Path(cfg['final_results_raw'])
    fsms_path        = Path(cfg['fsms_lookup'])
    final_output     = Path(cfg['output_csv'])
    final_output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COMPILE FINAL RESULTS")
    print("=" * 60)

    # 1. Load test metrics
    results_df = load_results_raw(results_raw_path)
    print(f"Loaded {len(results_df)} model results")

    # 2. Join FSMS
    fsms_df = pd.read_csv(fsms_path)
    # FSMS is keyed by (condition, arch). For real_only there is no FSMS.
    fsms_lookup = fsms_df.set_index(['condition', 'arch'])['fsms_score'].to_dict()

    def get_fsms(row):
        return fsms_lookup.get((row['condition'], row['arch']), float('nan'))

    results_df['fsms_score'] = results_df.apply(get_fsms, axis=1)
    print(f"FSMS joined. Missing: {results_df['fsms_score'].isna().sum()} / {len(results_df)}")

    # 3. Final column ordering
    col_order = ['condition', 'cohort', 'arch', 'seed', 'n_value',
                 'macro_f2', 'macro_f1', 'macro_auc', 'macro_recall', 'macro_precision',
                 'global_acc', 'fsms_score']
    for col in col_order:
        if col not in results_df.columns:
            results_df[col] = float('nan')

    results_df = results_df[col_order].sort_values(['cohort', 'n_value', 'arch', 'seed'])
    results_df.to_csv(final_output, index=False)

    print(f"\n{'='*60}")
    print(f"Final results saved: {final_output}  ({len(results_df)} rows)")

    # Print summary stats by cohort + n_value + arch
    print("\n--- Summary (mean ± std across seeds) ---")
    for (cohort, n_value, arch), grp in results_df.groupby(['cohort', 'n_value', 'arch']):
        f2_mean = grp['macro_f2'].mean()
        f2_std  = grp['macro_f2'].std()
        fsms    = grp['fsms_score'].mean()
        print(f"  {cohort:<24} | N={n_value:<6} | {arch:<10} | n={len(grp):3d} | "
              f"F2={f2_mean:.4f}±{f2_std:.4f} | FSMS={fsms:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
