"""Shared constants, data loaders and formatting helpers of make_tables.py and make_figures.py.

All inputs are read from the repository's data/ folder.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import fbeta_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data'
TABLES = HERE / 'outputs' / 'tables'
FIGURES = HERE / 'outputs' / 'figures'

CLASSES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
TAUS = [(0.90, 'tau090'), (0.93, 'tau093'), (0.97, 'tau097'), (0.99, 'tau099')]

NAMES = {
    'fid_inception': 'FID-Inception', 'fid_dino': 'FID-DINOv2', 'kid_dino': 'KID-DINOv2',
    'swd_dino': 'SWD-DINOv2', 'clip_log_odds_score': 'CLIP log-odds',
    'clip_similarity_cancer': 'CLIP sim. (cancer)', 'clip_similarity_artifact': 'CLIP sim. (artifact)',
    'fscs_score': 'FS-CS', 'lpips_score': 'LPIPS', 'fsim_score': 'FSIM', 'brisqe_score': 'BRISQUE',
    'psnr_score': 'PSNR', 'ssim_score': 'SSIM', 'diversity_score': 'Diversity',
    'realism_score': 'Realism', 'is_contribution_score': 'IS contribution',
}
DIST = ['fid_inception', 'fid_dino', 'kid_dino', 'swd_dino']
INST = [k for k in NAMES if k not in DIST]
PORTFOLIO15 = [k for k in NAMES if k != 'fid_dino']     # the 15 metrics the algorithm uses
ARCHS = ['swin_b', 'resnet50']


# --------------------------------------------------------------------------- data loading
def f2_row(path):
    """Per-class F2 (7 values) followed by Macro-F2 for one prediction file."""
    d = pd.read_csv(path, usecols=['true_label', 'predicted_label'])
    per = fbeta_score(d.true_label, d.predicted_label, beta=2, average=None, labels=range(7), zero_division=0)
    mac = fbeta_score(d.true_label, d.predicted_label, beta=2, average='macro', zero_division=0)
    return list(per) + [mac]


def seed_of(p):
    m = re.search(r'seed_?(\d+)', p.name)
    return int(m.group(1)) if m else -1


_cache = {}


def tournament():
    """{arch: {arm: array (runs, 8) ordered by seed}}; arms are metric names, random_control, real_only_baseline."""
    if 'tour' in _cache:
        return _cache['tour']
    out = {}
    for arch in ARCHS:
        groups = {}
        for p in (DATA / 'metric_tournament/raw_predictions').glob(f'*_{arch}_predictions.csv'):
            n = p.name.replace(f'_{arch}_predictions.csv', '').replace('edm2_', '')
            groups.setdefault(re.sub(r'_seed\d+$', '', n), []).append(p)
        out[arch] = {k: np.array([f2_row(p) for p in sorted(v, key=seed_of)]) for k, v in groups.items()}
    _cache['tour'] = out
    return out


def final():
    """{(arch, cohort): array (runs, 8)}; cohorts include tau090/tau093/tau097/tau099 for swin_b."""
    if 'final' in _cache:
        return _cache['final']
    base = DATA / 'benchmark'
    out = {}
    for arch in ARCHS:
        out[(arch, 'real_only')] = np.array([f2_row(base / f'raw_predictions/real_only_{arch}_predictions.csv')])
        for c in ['random_synth', 'random_synth_n20000', 'best_metric', 'algorithm', 'algorithm_n20000']:
            ps = sorted((base / 'raw_predictions').glob(f'{c}_seed_*_{arch}_predictions.csv'), key=seed_of)
            out[(arch, c)] = np.array([f2_row(p) for p in ps])
    for tag in ['090', '093', '099']:
        ps = sorted((base / 'safety_sweep/raw_predictions').glob(f'algorithm_safety{tag}_seed_*_swin_b_predictions.csv'), key=seed_of)
        out[('swin_b', f'tau{tag}')] = np.array([f2_row(p) for p in ps])
    out[('swin_b', 'tau097')] = out[('swin_b', 'algorithm')]
    _cache['final'] = out
    return out


def fsms_all():
    """{(arch, cohort): FSMS per run}, from final_results.csv and the safety sweep lookup (tau cohorts: Swin-B only)."""
    m = pd.read_csv(DATA / 'benchmark/final_results.csv')
    out = {}
    for arch in ARCHS:
        for c in ['random_synth', 'best_metric', 'algorithm', 'random_synth_n20000', 'algorithm_n20000']:
            out[(arch, c)] = m[(m.arch == arch) & (m.cohort == c)].fsms_score.values
    fs = pd.read_csv(DATA / 'benchmark/safety_sweep/fsms_lookup.csv')
    fs['tag'] = fs.condition.str.extract(r'safety(\d+)_seed')[0]
    for tag in ['090', '093', '099']:
        out[('swin_b', f'tau{tag}')] = fs[(fs.arch == 'swin_b') & (fs.tag == tag)].fsms_score.values
    out[('swin_b', 'tau097')] = out[('swin_b', 'algorithm')]
    return out


def fsms_final():
    """FSMS per run for the Swin-B cohorts at N=8,012, keyed by cohort."""
    fs = fsms_all()
    keys = ['random_synth', 'best_metric', 'algorithm', 'tau090', 'tau093', 'tau099', 'tau097']
    return {k: fs[('swin_b', k)] for k in keys}


# --------------------------------------------------------------------------- statistics
def welch(x, y):
    """Difference of means, its 95% Welch confidence interval and the Welch p-value."""
    t = stats.ttest_ind(x, y, equal_var=False)
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    h = stats.t.ppf(.975, t.df) * se
    d = x.mean() - y.mean()
    return d, d - h, d + h, t.pvalue


def locked_weights(arch='swin_b'):
    """Per-class weights of the locked configuration (Equations 1 and 2 of the paper)."""
    T = tournament()[arch]
    rnd = T['random_control'].mean(0)[:7]            # 60-seed mean for Swin-B, 30-seed for ResNet-50
    F = {k: T[k].mean(0)[:7] for k in PORTFOLIO15}
    out = {}
    for j, c in enumerate(CLASSES):
        f = {k: F[k][j] for k in PORTFOLIO15}
        floor = min(f.values())
        u = {k: v - floor + 1e-4 for k, v in f.items()}
        tot = sum(u.values())
        w = {k: v / tot for k, v in u.items()}
        dist = ['fid_inception', 'kid_dino', 'swd_dino']
        share = sum(w[d] for d in dist)
        ub = {d: max(0.0, f[d] - rnd[j]) for d in dist}
        tb = sum(ub.values())
        if tb > 0:
            for d in dist:
                w[d] = share * ub[d] / tb
        out[c] = w
    return pd.DataFrame(out)[CLASSES]


# --------------------------------------------------------------------------- LaTeX formatting
def signed(x, k=4):
    """Signed number with k decimals and a typeset minus sign."""
    return f'{x:+.{k}f}'.replace('-', '$-$')


def pvalue(p, digits=4, floor_exp=None):
    """p-value: fixed decimals from 0.001, scientific notation below; optionally '<10^-floor_exp' below that."""
    if floor_exp is not None and p < 10.0 ** -floor_exp:
        return f'$<10^{{-{floor_exp}}}$'
    if p >= 0.001:
        return f'{p:.{digits}f}'
    e = int(np.floor(np.log10(p)))
    return f'${p / 10 ** e:.1f}\\times10^{{{e}}}$'
