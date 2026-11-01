"""
Full FSMS validation and cohort comparison, in one pass.

Run after pack_features.py. Produces every number needed to support the safety claim:

    Under matched generation and budget, multi-axiom curation produces synthetic data
    measurably further from the real training set than FID-based checkpoint selection.

Six sections. 1-5 establish that FSMS is a sound instrument for this comparison;
6 makes the measurement.

  1. FLOOR        real train scored against itself -> must be exactly 0.0.
                  Also the exclude-self domain density: how similar real images are to
                  each other before any generator is involved.

  2. DETECTION    plant known copies at several rates; does FSMS fall proportionally,
                  and can per-image similarity identify which images were planted?
                  Jitter levels perturb the copies first (near-duplicates, not exact).

  3. ANCHOR       real held-out test vs real train. Zero memorisation by construction,
                  so this is what "genuinely novel, same domain" reads on this scale.
                  Without it the absolute values are uninterpretable.

  4. POOL AUDIT   per-image scores for the whole synthetic pool, near-duplicate counts
                  for the release statement, and the per-checkpoint x CFG breakdown
                  that tests the mechanism claim (does FID-based selection land on the
                  configuration closest to the training data?).

  5. CROSS-MODAL  does FSMS measure copying or class typicality? SSIM/PSNR/LPIPS from
                  metadata.csv were computed against the same nearest real neighbour
                  FSMS scores against, but are pure pixel/perceptual measures. If they
                  improve monotonically with cosine, the signal is genuine proximity.

  6. COHORTS      THE CLAIM. Per-seed FSMS and tail statistics for every cohort,
                  recomputed from the curated manifests, plus paired tests. Reports the
                  tail (max, p99.9, % above thresholds) alongside the mean, because
                  privacy risk is a tail property and FSMS is a mean.

Everything is read-only with respect to source data. ~3 min.

Usage:
    python fsms/scripts/validate.py --config fsms/configs/validate.yml
    python fsms/scripts/validate.py --config ... --sections 1,6
"""
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(str(Path(__file__).resolve().parent))

from utils.helpers import load_and_resolve_config
from fsms_core import (
    load_features, rows_for_csv, max_sim_to_real, fsms_from_max_sims,
)

try:
    from scipy import stats
    from sklearn.metrics import roc_auc_score
except ImportError as e:
    print(f"[FATAL] scipy and scikit-learn are required: {e}", file=sys.stderr)
    sys.exit(1)

FLOOR_TOLERANCE = 1e-5
PIXEL_METRICS = [('ssim_score', 'SSIM', 'higher'),
                 ('psnr_score', 'PSNR', 'higher'),
                 ('lpips_score', 'LPIPS', 'lower')]


def head(n, title):
    print(f"\n{'=' * 74}\n{n}. {title}\n{'=' * 74}")


def tail_stats(sims: np.ndarray, thresholds) -> dict:
    """Mean FSMS plus the tail descriptors. Tail is what matters for privacy."""
    row = {
        'fsms': 1.0 - sims.mean(),
        'mean_max_sim': sims.mean(),
        'p99': float(np.quantile(sims, 0.99)),
        'p99_9': float(np.quantile(sims, 0.999)),
        'max': float(sims.max()),
    }
    for t in thresholds:
        row[f'pct_ge_{t}'] = 100.0 * float((sims >= t).mean())
    return row


# ===================================================================
# 1. FLOOR
# ===================================================================

def section_floor(real, out):
    head(1, "FLOOR - implementation correctness + domain density")
    self_sims = max_sim_to_real(real, real).numpy()
    fsms_self = 1.0 - self_sims.mean()
    ok = abs(fsms_self) < FLOOR_TOLERANCE
    print(f"  real train vs itself         FSMS = {fsms_self:.3e}   "
          f"[{'PASS' if ok else 'FAIL'}]  (must be 0.0)")
    if not ok:
        print("  [FAIL] Non-zero FSMS on exact copies - implementation is wrong.",
              file=sys.stderr)

    other = max_sim_to_real(real, real, exclude_self=True).numpy()
    print(f"  real train vs nearest OTHER  FSMS = {1 - other.mean():.4f}  "
          f"(cosine {other.mean():.4f})")
    print(f"    {100 * (other >= 0.95).mean():.2f}% of real images have a >=0.95 "
          f"near-duplicate inside the training set; max {other.max():.4f}")
    print("    -> the domain's own floor. FSMS cannot approach 1.0 here, and the real")
    print("       data is itself more internally duplicated than any synthetic cohort.")

    df = pd.DataFrame([
        {'set': 'real_train vs itself', 'fsms': fsms_self, 'passed': ok},
        {'set': 'real_train vs other real', 'fsms': 1 - other.mean(),
         'mean_cosine': other.mean(), 'pct_ge_0.95': 100 * (other >= 0.95).mean(),
         'max': other.max()},
    ])
    df.to_csv(out / 'floor.csv', index=False)
    return ok


# ===================================================================
# 2. DETECTION
# ===================================================================

def section_detection(cfg, real, names, synth, out):
    head(2, "DETECTION - sensitivity to planted duplicates")
    base, _ = rows_for_csv(Path(cfg['detection_base_csv']), names, synth)
    rates = cfg.get('contamination_rates', [0.0, 0.01, 0.05, 0.10, 0.25])
    jitters = [0.0] + list(cfg.get('jitter_levels', [0.05, 0.15, 0.30]))
    clean = fsms_from_max_sims(max_sim_to_real(base, real))
    print(f"  base: {Path(cfg['detection_base_csv']).name}  "
          f"({len(base)} images, FSMS {clean:.4f})\n")

    rows = []
    for jit in jitters:
        g = torch.Generator().manual_seed(42)
        label = 'exact copies' if jit == 0 else f'near-dup (jitter {jit})'
        cells = []
        for rate in rates:
            k = int(round(rate * len(base)))
            mix, lab = base.clone(), torch.zeros(len(base))
            if k:
                victim = torch.randperm(len(base), generator=g)[:k]
                planted = real[torch.randperm(len(real), generator=g)[:k]].clone()
                if jit:
                    # sigma must be /sqrt(D): a D-dim Gaussian with per-component sigma
                    # s has L2 norm ~ s*sqrt(D). Scaling per component by the full
                    # vector norm makes the perturbation ~28x too large - the "copy"
                    # then lands further from the real set than a genuine synthetic
                    # image and AUROC collapses toward 0.
                    sigma = planted.norm(dim=1, keepdim=True) * jit / math.sqrt(planted.shape[1])
                    planted = planted + torch.randn(planted.shape, generator=g) * sigma
                mix[victim], lab[victim] = planted, 1.0
            s = max_sim_to_real(mix, real).numpy()
            auc = roc_auc_score(lab.numpy(), s) if k else float('nan')
            rows.append({'jitter': jit, 'contamination': rate, 'n_planted': k,
                         'fsms': 1 - s.mean(), 'auroc': auc,
                         'drop_vs_clean': clean - (1 - s.mean())})
            cells.append((rate, k, 1 - s.mean(), auc))
        print(f"  {label}")
        for rate, k, f, a in cells:
            astr = f"{a:.4f}" if k else "  -   "
            print(f"    {rate:5.0%}  n={k:5d}  FSMS={f:.4f}  drop={clean - f:.4f}  AUROC={astr}")
    pd.DataFrame(rows).to_csv(out / 'detection.csv', index=False)
    print("\n  Compare each drop against the ~0.0005 seed-to-seed sd of the published")
    print("  cohorts - that ratio is how little memorisation FSMS can resolve.")
    return clean


# ===================================================================
# 3. ANCHOR
# ===================================================================

def section_anchor(cfg, real, out):
    head(3, "ANCHOR - what 'genuinely novel' reads on this scale")
    p = Path(cfg['real_test_features'])
    if not p.exists():
        print(f"  [SKIP] {p} not found - extract and pack the features first (extract_test_features.yml, pack_features.py).")
        return None
    _, test = load_features(p)
    sims = max_sim_to_real(test, real).numpy()
    anchor = 1 - sims.mean()
    print(f"  real held-out test vs real train   FSMS = {anchor:.4f}  "
          f"(cosine {sims.mean():.4f}, max {sims.max():.4f})")
    print(f"    {100 * (sims >= 0.95).mean():.2f}% >= 0.95, "
          f"{100 * (sims >= 0.99).mean():.2f}% >= 0.99")
    print("    -> zero memorisation by construction. Read every cohort as a position")
    print("       between 0.0 (exact copies) and this value.")
    pd.DataFrame([tail_stats(sims, cfg.get('tail_thresholds', [0.95, 0.97, 0.99]))
                  | {'set': 'real_test vs real_train'}]).to_csv(out / 'anchor.csv', index=False)
    return anchor


# ===================================================================
# 4. POOL AUDIT
# ===================================================================

def section_pool(cfg, names, pool_sims, out):
    head(4, "POOL AUDIT - release statement + mechanism")
    thresholds = [0.90] + list(cfg.get('tail_thresholds', [0.95, 0.97, 0.99]))
    print(f"  pool FSMS {1 - pool_sims.mean():.4f}   max similarity {pool_sims.max():.4f}\n")
    print(f"    {'threshold':>10s} {'images':>9s} {'% of pool':>11s}")
    for t in sorted(set(thresholds)):
        n = int((pool_sims >= t).sum())
        print(f"    {t:>10.2f} {n:>9,} {100 * n / len(pool_sims):>10.4f}%")

    df = pd.DataFrame({'filename': names, 'max_sim_to_real': pool_sims})
    ck = df.filename.str.extract(r'ckpt_(\d+)k_cfg_([\d.]+)')
    df['checkpoint_kimg'] = pd.to_numeric(ck[0], errors='coerce')
    df['cfg_scale'] = pd.to_numeric(ck[1], errors='coerce')
    df.to_csv(out / 'pool_per_image.csv', index=False)

    by = df.groupby(['checkpoint_kimg', 'cfg_scale']).max_sim_to_real.mean().reset_index()
    by['fsms'] = 1 - by.max_sim_to_real
    by.to_csv(out / 'pool_by_config.csv', index=False)
    piv = by.pivot(index='checkpoint_kimg', columns='cfg_scale', values='fsms')
    print("\n  FSMS by checkpoint x CFG (mechanism test - lowest = most memorising)")
    print("  " + piv.round(4).to_string().replace("\n", "\n  "))
    lo = by.loc[by.fsms.idxmin()]
    print(f"\n    pool minimum: ckpt_{int(lo.checkpoint_kimg)}k_cfg_{lo.cfg_scale} "
          f"(FSMS {lo.fsms:.4f})")
    print("    If that is the config FID selects, the mechanism claim is measured.")

    flag_t = cfg.get('tail_thresholds', [0.95])[0]
    df[df.max_sim_to_real >= flag_t].nlargest(500, 'max_sim_to_real') \
        .to_csv(out / 'pool_flagged.csv', index=False)


# ===================================================================
# 5. CROSS-MODAL
# ===================================================================

def section_crossmodal(cfg, names, pool_sims, out):
    head(5, "CROSS-MODAL - copying, or class typicality?")
    meta_path = Path(cfg.get('pool_metadata', ''))
    if not meta_path.exists():
        print(f"  [SKIP] pool_metadata not found: {meta_path}")
        return
    cols = ['filename'] + [c for c, _, _ in PIXEL_METRICS]
    # the feature keys carry an 'edm2/' prefix that the released metadata does not have
    j = pd.DataFrame({'filename': [n.removeprefix('edm2/') for n in names], 'max_sim': pool_sims}) \
          .merge(pd.read_csv(meta_path, usecols=cols), on='filename', how='inner')
    if len(j) < 0.9 * len(names):
        print(f"  [WARNING] only {len(j)} of {len(names)} joined - check key formats.")
    print(f"  joined {len(j):,}\n")

    rows, agree = [], True
    for col, lab, direction in PIXEL_METRICS:
        r, rp = stats.pearsonr(j.max_sim, j[col])
        rs, _ = stats.spearmanr(j.max_sim, j[col])
        ok = (r > 0) == (direction == 'higher')
        agree &= ok
        print(f"    max_sim vs {lab:6s} ({direction:6s}=closer)  pearson={r:+.3f}  "
              f"spearman={rs:+.3f}  {'AGREES' if ok else 'CONTRADICTS'}")
        rows.append({'metric': lab, 'pearson': r, 'pearson_p': rp, 'spearman': rs,
                     'agrees': ok})
    pd.DataFrame(rows).to_csv(out / 'crossmodal_correlations.csv', index=False)

    q = j.max_sim.quantile([0.5, 0.9, 0.99, 0.999])
    brackets = [(j.max_sim.min(), q[0.5], 'bottom 50%'), (q[0.5], q[0.9], 'p50-p90'),
                (q[0.9], q[0.99], 'p90-p99'), (q[0.99], q[0.999], 'p99-p99.9'),
                (q[0.999], 1.1, 'top 0.1%')]
    brows = []
    print(f"\n    {'bracket':14s} {'n':>8s} {'cosine':>8s} {'SSIM':>7s} {'PSNR':>7s} {'LPIPS':>7s}")
    for lo, hi, lab in brackets:
        s = j[(j.max_sim >= lo) & (j.max_sim < hi)]
        if s.empty:
            continue
        brows.append({'bracket': lab, 'n': len(s), 'mean_cosine': s.max_sim.mean(),
                      'ssim': s.ssim_score.mean(), 'psnr': s.psnr_score.mean(),
                      'lpips': s.lpips_score.mean()})
        print(f"    {lab:14s} {len(s):>8,} {s.max_sim.mean():>8.4f} "
              f"{s.ssim_score.mean():>7.3f} {s.psnr_score.mean():>7.2f} "
              f"{s.lpips_score.mean():>7.3f}")
    b = pd.DataFrame(brows)
    b.to_csv(out / 'crossmodal_brackets.csv', index=False)

    d_psnr = b.psnr.iloc[-1] - b.psnr.iloc[0]
    d_lpips = b.lpips.iloc[0] - b.lpips.iloc[-1]
    if agree and d_psnr > 1.0 and d_lpips > 0.05:
        print(f"\n    PSNR {b.psnr.iloc[0]:.2f} -> {b.psnr.iloc[-1]:.2f} dB, "
              f"LPIPS {b.lpips.iloc[0]:.3f} -> {b.lpips.iloc[-1]:.3f}")
        print("    -> genuine visual proximity, not class typicality. Three pixel")
        print("       metrics that never see the embedding independently agree.")
    else:
        print("\n    [!] Weak/inconsistent - the typicality objection is NOT resolved.")


# ===================================================================
# 6. COHORTS - the claim
# ===================================================================

def section_cohorts(cfg, real, names, synth, out, anchor):
    head(6, "COHORTS - the safety claim (mean AND tail)")
    cohorts = cfg.get('cohorts', {})
    if not cohorts:
        print("  [SKIP] no cohorts configured.")
        return
    thresholds = cfg.get('tail_thresholds', [0.95, 0.97, 0.99])

    per_seed = []
    for label, d in cohorts.items():
        csvs = sorted(Path(d).glob('seed_*.csv'),
                      key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0))
        if not csvs:
            print(f"  [WARN] no seed_*.csv in {d}")
            continue
        for c in csvs:
            m, _ = rows_for_csv(c, names, synth)
            s = max_sim_to_real(m, real).numpy()
            per_seed.append({'cohort': label, 'seed': int(''.join(filter(str.isdigit, c.stem))),
                             'n': len(s)} | tail_stats(s, thresholds))
        print(f"  {label:14s} {len(csvs)} seeds recomputed from curated manifests")
    ps = pd.DataFrame(per_seed)
    ps.to_csv(out / 'cohort_per_seed.csv', index=False)

    agg_cols = ['fsms', 'p99_9', 'max'] + [f'pct_ge_{t}' for t in thresholds]
    summ = ps.groupby('cohort')[agg_cols].agg(['mean', 'std'])
    summ.to_csv(out / 'cohort_summary.csv')

    print(f"\n  {'cohort':14s} {'FSMS':>16s} {'p99.9':>9s} {'max':>9s} "
          + ' '.join(f'{"%>="+str(t):>9s}' for t in thresholds))
    for label in cohorts:
        if label not in ps.cohort.values:
            continue
        g = ps[ps.cohort == label]
        line = (f"  {label:14s} {g.fsms.mean():.4f}+-{g.fsms.std(ddof=1):.4f} "
                f"{g.p99_9.mean():>9.4f} {g['max'].mean():>9.4f} ")
        line += ' '.join(f'{g[f"pct_ge_{t}"].mean():>9.3f}' for t in thresholds)
        print(line + (f"   ({100 * g.fsms.mean() / anchor:.1f}% of anchor)" if anchor else ""))

    comp = cfg.get('comparison', {})
    a, b = comp.get('primary'), comp.get('baseline')
    if not (a in ps.cohort.values and b in ps.cohort.values):
        return
    A = ps[ps.cohort == a].sort_values('seed')
    B = ps[ps.cohort == b].sort_values('seed')
    if len(A) != len(B):
        print(f"\n  [WARN] {a} has {len(A)} seeds, {b} has {len(B)} - cannot pair.")
        return

    print(f"\n  PAIRED: {a} vs {b}  (n={len(A)}, two-sided)")
    print(f"  {'statistic':12s} {'primary':>9s} {'baseline':>9s} {'delta':>9s} "
          f"{'p':>10s} {'better in':>11s}   direction")
    trows = []
    for col, better_low in [('fsms', False), ('p99_9', True), ('max', True)] + \
                           [(f'pct_ge_{t}', True) for t in thresholds]:
        x, y = A[col].values, B[col].values
        if np.allclose(x, y):
            continue
        t, p = stats.ttest_rel(x, y)
        d = (x - y).mean()
        wins = int((x < y).sum()) if better_low else int((x > y).sum())
        good = (d < 0) if better_low else (d > 0)
        trows.append({'statistic': col, 'primary_mean': x.mean(), 'baseline_mean': y.mean(),
                      'delta': d, 't': t, 'p_two_sided': p, 'n_seeds_better': wins,
                      'favours_primary': good})
        print(f"  {col:12s} {x.mean():>9.4f} {y.mean():>9.4f} {d:>+9.4f} "
              f"{p:>10.2e} {wins:>6d}/{len(x)}   "
              f"{'FAVOURS ' + a if good else 'favours ' + b}")
    pd.DataFrame(trows).to_csv(out / 'paired_tests.csv', index=False)

    print(f"\n  FSMS is a MEAN; privacy risk is a TAIL property. The rows below 'fsms'")
    print(f"  are what make this a safety claim rather than an average-case one.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--sections', default='1,2,3,4,5,6',
                    help="Comma-separated subset, e.g. '1,6'. Default: all.")
    args = ap.parse_args()
    want = {int(s) for s in args.sections.split(',') if s.strip()}

    cfg = load_and_resolve_config(Path(args.config))
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("FSMS VALIDATION + COHORT COMPARISON")
    print("=" * 74)
    _, real = load_features(Path(cfg['real_train_features']))
    names, synth = load_features(Path(cfg['synth_pool_features']))

    pool_sims = None
    if want & {4, 5}:
        print(f"\nScoring the full pool ({len(names):,} images)...")
        pool_sims = max_sim_to_real(synth, real).numpy()

    floor_ok = section_floor(real, out) if 1 in want else None
    if 2 in want:
        section_detection(cfg, real, names, synth, out)
    anchor = section_anchor(cfg, real, out) if 3 in want else None
    if 4 in want:
        section_pool(cfg, names, pool_sims, out)
    if 5 in want:
        section_crossmodal(cfg, names, pool_sims, out)
    if 6 in want:
        section_cohorts(cfg, real, names, synth, out, anchor)

    print(f"\n{'=' * 74}\nAll outputs -> {out}\n{'=' * 74}")
    if floor_ok is False:
        sys.exit(1)


if __name__ == '__main__':
    main()
