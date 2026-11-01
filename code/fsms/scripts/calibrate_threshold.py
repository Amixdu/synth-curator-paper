"""
Calibrate the memorisation threshold against ground truth.

WHY THIS EXISTS
---------------
The sections of validate.py characterise FSMS against a NULL: real test images, never seen by the
generator, therefore containing no memorisation by construction. That establishes what
"not a copy" looks like. It does not establish where copies BEGIN, and those are
different points with an unmeasured gap between them.

Concretely, a threshold set from that null fails on its own terms. Split-conformal
calibration on the 1,512 test images permits alpha * 275,472 false flags; at every usable
alpha the observed flag count came out 6-9% of that budget. A bound that tolerates 2,755
false alarms while the entire flag set is 239 carries no information about whether any of
the 239 is real, so it cannot license removing them.

The fix is a labelled POSITIVE class, which ISIC already provides. Many lesions were
photographed more than once - same physical lesion, different session, angle or lighting -
and those images share a `lesion_id`. Two images sharing one are the same physical thing,
which is exactly the relation a memorised generation has to its training image. So:

    same lesion_id      -> positive: "these show the same thing"
    different lesion_id -> negative: "different mole, different person"

Both classes, on the same statistic FSMS scores (DINOv2 cosine), in the same domain, from
real images only. The threshold then comes from a measured precision curve.

WHAT IT PRODUCES
----------------
    lesion_ids.csv            the fetched cache
    threshold_calibration.csv precision/recall by cosine band - the calibration curve
    pool_at_threshold.csv     pool images clearing each candidate tau, overall and per class

WHAT IT DOES NOT ESTABLISH
--------------------------
Precision, not recall against generated copies. Same-lesion pairs are re-photographs, not
diffusion reproductions; a generation could memorise in a way that lands below any
threshold calibrated here. This bounds false removals, not missed reproductions - the same
limitation the null-based approach had, stated rather than papered over.

Network is needed ONCE for the fetch (~160 batched requests, ~3 min). Cached thereafter.
Everything else is CPU-only and offline. No GPU at any point.

    python fsms/scripts/calibrate_threshold.py --config fsms/configs/calibrate.yml
    python fsms/scripts/calibrate_threshold.py --config ... --fetch-only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fsms.scripts.fsms_core import load_features  # noqa: E402
from utils.helpers import load_and_resolve_config  # noqa: E402


def head(n, title):
    print(f"\n{'=' * 70}\n{n}. {title}\n{'=' * 70}")


# ===================================================================
# FETCH
# ===================================================================

def fetch_lesion_ids(names, cfg, cache_path: Path) -> pd.DataFrame:
    """
    Map each Real_Train image to its ISIC `lesion_id`, caching the result.

    Batched as OR queries against the search endpoint: 50 ids per request returns in
    ~1.2 s, so 8,012 images cost ~160 requests rather than 8,012. Measured with zero
    misses, but any id the archive does not return is recorded with a null lesion_id
    rather than dropped - a silently shorter table would bias the calibration toward
    whatever the archive happens to serve.
    """
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"  [CACHE] {len(df):,} lesion ids from {cache_path}")
        return df

    import requests

    isic_ids = sorted({Path(n).stem for n in names})
    print(f"  Fetching lesion_id for {len(isic_ids):,} images "
          f"({len(isic_ids) // cfg['batch_size'] + 1} batched requests)...")

    got, t0 = {}, time.time()
    bs = cfg['batch_size']
    for i in range(0, len(isic_ids), bs):
        chunk = isic_ids[i:i + bs]
        query = " OR ".join(f"isic_id:{x}" for x in chunk)
        for attempt in range(cfg['retries']):
            try:
                r = requests.get(cfg['isic_api'],
                                 params={"query": query, "limit": len(chunk)},
                                 timeout=cfg['request_timeout'])
                r.raise_for_status()
                for rec in r.json().get('results', []):
                    got[rec['isic_id']] = rec.get('metadata', {}).get('clinical', {}).get('lesion_id')
                break
            except Exception as e:                      # noqa: BLE001
                if attempt == cfg['retries'] - 1:
                    print(f"    [WARN] batch at {i} failed after {cfg['retries']} tries: {e}")
                else:
                    time.sleep(2 ** attempt)
        if (i // bs) % 20 == 0 and i:
            print(f"    {len(got):>6,}/{len(isic_ids):,}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame({'isic_id': isic_ids,
                       'lesion_id': [got.get(x) for x in isic_ids]})
    miss = df.lesion_id.isna().sum()
    print(f"  Fetched {len(df) - miss:,}/{len(df):,} in {time.time() - t0:.0f}s"
          + (f"  ({miss} missing)" if miss else ""))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  [CACHE] written to {cache_path}")
    return df


# ===================================================================
# CALIBRATION
# ===================================================================

def build_pairs(mat: torch.Tensor, names, floor: float) -> pd.DataFrame:
    """
    Every pair of real training images with cosine >= floor.

    The full 8012x8012 matrix is only ~250 MB, but it is built in row blocks anyway so
    the script stays usable on a laptop alongside the 938 MB pool matrix. Only the upper
    triangle is kept (j > i): a pair is one observation, not two, and counting both would
    halve the apparent variance of every precision estimate.
    """
    n = mat.shape[0]
    x = mat / (mat.norm(dim=1, keepdim=True) + 1e-8)
    rows, block = [], 512
    for s in range(0, n, block):
        sim = (x[s:s + block] @ x.T).numpy()
        for local, i in enumerate(range(s, min(s + block, n))):
            j = np.where(sim[local] >= floor)[0]
            j = j[j > i]                                # upper triangle, excludes self
            if len(j):
                rows.append(pd.DataFrame({'i': i, 'j': j, 'cos': sim[local][j]}))
    pairs = pd.concat(rows, ignore_index=True) if rows else \
        pd.DataFrame(columns=['i', 'j', 'cos'])
    stem = [Path(v).stem for v in names]
    pairs['a'] = [stem[k] for k in pairs.i]
    pairs['b'] = [stem[k] for k in pairs.j]
    return pairs.drop(columns=['i', 'j'])


def calibrate(pairs: pd.DataFrame, bands, out: Path) -> pd.DataFrame:
    """Same-lesion precision per cosine band - the calibration curve."""
    total_pos = int(pairs.same_lesion.sum())
    rows = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        b = pairs[(pairs.cos >= lo) & (pairs.cos < hi)]
        if not len(b):
            continue
        pos = int(b.same_lesion.sum())
        at_or_above = pairs[pairs.cos >= lo]
        rows.append({
            'band': f"{lo:.3f}-{hi:.3f}" if hi <= 1.0 else f">={lo:.3f}",
            'pairs': len(b),
            'same_lesion': pos,
            'precision_in_band': pos / len(b),
            'pairs_at_or_above': len(at_or_above),
            'precision_at_or_above': at_or_above.same_lesion.mean(),
            'recall_at_or_above': (at_or_above.same_lesion.sum() / total_pos) if total_pos else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / 'threshold_calibration.csv', index=False)
    print(f"\n  {'band':>14} {'pairs':>8} {'same lesion':>12} {'precision':>10} "
          f"| {'>= lo: precision':>17} {'recall':>8}")
    for _, r in df.iterrows():
        print(f"  {r.band:>14} {r.pairs:>8,} {r.same_lesion:>12,} {r.precision_in_band:>9.1%} "
              f"| {r.precision_at_or_above:>16.1%} {r.recall_at_or_above:>7.1%}")
    return df


def pool_impact(cfg, taus, out: Path):
    """How many synthetic images clear each candidate threshold, overall and per class."""
    pool = pd.read_csv(cfg['pool_per_image'])
    # the pool table keys carry an 'edm2/' prefix that the released metadata does not have
    pool['filename'] = pool.filename.str.removeprefix('edm2/')
    md = pd.read_csv(cfg['pool_metadata'], usecols=['filename', 'target_class_label'])
    pool = pool.merge(md, on='filename', how='left')
    n = len(pool)
    rows = []
    for t in taus:
        hit = pool[pool.max_sim_to_real >= t]
        row = {'tau': t, 'flagged': len(hit), 'pct_of_pool': 100 * len(hit) / n}
        for c, g in pool.groupby('target_class_label'):
            row[f'n_{c}'] = int((g.max_sim_to_real >= t).sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'pool_at_threshold.csv', index=False)
    print(f"\n  {'tau':>7} {'flagged':>9} {'% pool':>9}   per class")
    for _, r in df.iterrows():
        per = {k[2:]: int(v) for k, v in r.items() if k.startswith('n_') and v}
        print(f"  {r.tau:>7.3f} {int(r.flagged):>9,} {r.pct_of_pool:>8.4f}%   {per}")

    # A class whose entire pool maximum sits below tau can never be filtered at all. That
    # is not "few removals" - it is zero protection for that class, and it is the argument
    # for keeping the filter class-conditional rather than global.
    print("\n  Per-class pool maximum (a class below tau is unfilterable at that tau):")
    for c, g in pool.groupby('target_class_label'):
        mx = g.max_sim_to_real.max()
        below = [f"{t:.3f}" for t in taus if mx < t]
        print(f"    {c:<6} max={mx:.4f}" + (f"   unfilterable at tau in {below}" if below else ""))
    return df


# ===================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--fetch-only', action='store_true',
                    help="fetch and cache lesion ids, then stop")
    args = ap.parse_args()
    cfg = load_and_resolve_config(Path(args.config))
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)

    head(1, "LESION IDS - the labelled positive class")
    names, mat = load_features(Path(cfg['real_train_features']))
    lid = fetch_lesion_ids(names, cfg, Path(cfg['lesion_id_cache']))
    lid = lid.dropna(subset=['lesion_id'])
    grp = lid.groupby('lesion_id').size()
    print(f"  {len(lid):,} images over {len(grp):,} distinct lesions")
    print(f"  lesions with >1 image in Real_Train: {int((grp > 1).sum()):,} "
          f"-> {int(((grp * (grp - 1)) // 2).sum()):,} same-lesion pairs available")
    if args.fetch_only:
        return

    head(2, "PAIRS - every real/real pair above the floor")
    pairs = build_pairs(mat, names, cfg['pair_floor'])
    m = dict(zip(lid.isic_id, lid.lesion_id))
    pairs['la'] = pairs.a.map(m)
    pairs['lb'] = pairs.b.map(m)
    pairs = pairs.dropna(subset=['la', 'lb'])
    pairs['same_lesion'] = pairs.la == pairs.lb
    print(f"  {len(pairs):,} pairs at cos >= {cfg['pair_floor']}, "
          f"{int(pairs.same_lesion.sum()):,} of them the same physical lesion "
          f"({pairs.same_lesion.mean():.1%})")
    pairs.to_csv(out / 'real_pairs.csv', index=False)

    head(3, "CALIBRATION - where does 'same physical lesion' actually begin?")
    calibrate(pairs, cfg['bands'], out)

    head(4, "POOL IMPACT - what each candidate threshold costs")
    pool_impact(cfg, cfg['candidate_taus'], out)

    print(f"\nWritten to {out}/7_*.csv")
    print("\nRead the calibration table for the threshold, then the pool table for its cost.")
    print("Precision is against re-photographs of the same lesion, not against generated")
    print("copies: this bounds false removals, not missed reproductions.")


if __name__ == '__main__':
    main()
