#!/usr/bin/env python3
"""Tail statistics of nearest-real similarity for curated cohorts.

For every cohort folder under --root, reads each curated manifest (seed_*.csv with a
`source_path` column), looks up the per-image similarity to the nearest real training image
(`max_sim_to_real`), and reports, per seed and across seeds:

    mean similarity (FSMS = 1 - mean), max, p99, p99.9,
    count and share of images at or above 0.95 and 0.97.

READ-ONLY: it only reads the manifests and the similarity table and writes two CSVs to --out.

Usage from the repo root (only pandas and numpy are needed):

    python benchmark/scripts/cohort_tail_stats.py \
        --root /path/to/workdir/benchmark \
        --sim  data/dataset_release/metadata.csv \
        --out  tail_stats

--sim is data/dataset_release/metadata.csv. Any CSV
with `filename` and `max_sim_to_real` columns works.

Outputs:  <out>_per_seed.csv   one row per (cohort, seed)
          <out>_summary.csv    one row per cohort, mean and SD over seeds
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COHORTS = ["random_synth", "best_metric", "algorithm", "algorithm_safety090",
           "algorithm_safety093", "algorithm_safety099"]
THRESHOLDS = [0.95, 0.97]


def pool_key(source_path: str) -> str:
    """Manifest source_path -> the pool-relative `filename` used in the similarity table,
    for example ckpt_25k_cfg_1.5/class_0/001497.png."""
    i = source_path.find("ckpt_")
    if i < 0:
        raise ValueError(f"cannot derive pool-relative path from {source_path!r}")
    return source_path[i:]


def find_manifests(cohort_dir: Path):
    files = sorted(cohort_dir.rglob("seed_*.csv"), key=lambda p: (len(p.stem), p.stem))
    keep = []
    for f in files:
        try:
            cols = pd.read_csv(f, nrows=1).columns
        except Exception:
            continue
        if "source_path" in cols:
            keep.append(f)
    return keep


def seed_stats(sims: np.ndarray) -> dict:
    out = {
        "n_images": len(sims),
        "mean_sim": sims.mean(),
        "fsms": 1.0 - sims.mean(),
        "max_sim": sims.max(),
        "p99": np.percentile(sims, 99),
        "p99_9": np.percentile(sims, 99.9),
    }
    for t in THRESHOLDS:
        c = int((sims >= t).sum())
        out[f"count_ge_{t:.2f}"] = c
        out[f"share_ge_{t:.2f}_pct"] = 100.0 * c / len(sims)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder holding one sub-folder per cohort")
    ap.add_argument("--sim", required=True, help="CSV with filename and max_sim_to_real")
    ap.add_argument("--out", default="tail_stats", help="output prefix (default: tail_stats)")
    ap.add_argument("--cohorts", nargs="+", default=COHORTS, help="cohort folder names")
    args = ap.parse_args()

    sim_df = pd.read_csv(args.sim, usecols=["filename", "max_sim_to_real"])
    if sim_df["filename"].duplicated().any():
        sys.exit("ERROR: duplicate filenames in the similarity table")
    sim = dict(zip(sim_df["filename"], sim_df["max_sim_to_real"].astype(float)))
    print(f"similarity table: {len(sim):,} images")

    rows = []
    for cohort in args.cohorts:
        cdir = Path(args.root) / cohort
        if not cdir.is_dir():
            print(f"[skip] {cohort}: {cdir} not found")
            continue
        manifests = find_manifests(cdir)
        if not manifests:
            print(f"[skip] {cohort}: no seed_*.csv manifests with a source_path column under {cdir}")
            continue
        for f in manifests:
            m = pd.read_csv(f, usecols=["source_path"])
            keys = [pool_key(p) for p in m["source_path"]]
            missing = [k for k in keys if k not in sim]
            if missing:
                sys.exit(f"ERROR: {len(missing)} images of {f} are not in the similarity table, e.g. {missing[0]}")
            sims = np.array([sim[k] for k in keys])
            rows.append({"cohort": cohort, "manifest": str(f.relative_to(args.root)), **seed_stats(sims)})
        print(f"{cohort}: {len(manifests)} manifests")

    if not rows:
        sys.exit("nothing computed")
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(f"{args.out}_per_seed.csv", index=False)

    metrics = [c for c in per_seed.columns if c not in ("cohort", "manifest")]
    g = per_seed.groupby("cohort", sort=False)
    summary = g[metrics].mean().add_suffix("_mean")
    summary = summary.join(g[metrics].std(ddof=1).add_suffix("_sd"))
    summary.insert(0, "n_seeds", g.size())
    summary["max_sim_worst_seed"] = g["max_sim"].max()
    summary.to_csv(f"{args.out}_summary.csv")

    show = summary[["n_seeds", "fsms_mean", "max_sim_mean", "max_sim_worst_seed", "p99_9_mean",
                    "count_ge_0.95_mean", "share_ge_0.95_pct_mean",
                    "count_ge_0.97_mean", "share_ge_0.97_pct_mean"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print("\n", show.to_string())
    print(f"\nwrote {args.out}_per_seed.csv and {args.out}_summary.csv")


if __name__ == "__main__":
    main()
