"""Builds every data figure of the paper from the repo's own data/ folder.

Usage:  python make_figures.py                     (writes PDFs to outputs/figures/)
        python make_figures.py ranking heatmap     (only the named figures)

The TikZ diagram of Figure 1 (tournament runs) is drawn in LaTeX and is not produced here.

Figures
    ranking        Tournament ranking of the 16 metrics (Macro-F2, both architectures)
    heatmap        Per-class tournament scores relative to random sampling (Swin-B)
    perclass_bars  Per-class F2 of FID-Inception, random control and the best metric (Swin-B)
    weights        Locked per-class metric weights (Swin-B)
    calibration    Safety-threshold calibration: precision, recall and pool images removed
    forest         Benchmark comparisons with 95% confidence intervals
    benchmark      Test Macro-F2 of every cohort, one point per curation seed (Swin-B)
    safety         Test Macro-F2 and FSMS against the safety threshold (Swin-B)
    agreement      Swin-B against ResNet-50 tournament Macro-F2 per arm

Figures for ResNet-50 (resnet_*) are written by the same functions as their Swin-B counterparts.
"""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from scipy import stats

from common import (CLASSES, DATA, DIST, FIGURES, INST, NAMES, PORTFOLIO15, TAUS,
                    final, fsms_final, locked_weights, tournament, welch)

OUT = FIGURES
OUT.mkdir(parents=True, exist_ok=True)

# Colours (validated categorical palette)
BLUE, AQUA, ORANGE, GRAY, DARK = '#2a78d6', '#1a9a6c', '#eb6834', '#8a8a85', '#52514e'
TAU_RAMP = ['#f6b48a', '#ee8148', '#d4551a', '#8c3510']
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.linewidth': .6,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT / f'{name}.pdf')


# --------------------------------------------------------------------------- helpers

def merged_arms(T, arch='swin_b'):
    """Arm keys with FID-DINOv2 and KID-DINOv2 merged (identical results), sorted by `arch` Macro-F2."""
    keys = [k for k in DIST + INST if k != 'kid_dino']
    keys.sort(key=lambda k: -T[arch][k][:, 7].mean())
    return keys


def all_arms(T, arch='swin_b'):
    """All 16 arms sorted by `arch` Macro-F2. FID-DINOv2 and KID-DINOv2 have identical results and appear as two rows."""
    keys = DIST + INST
    keys.sort(key=lambda k: -T[arch][k][:, 7].mean())
    return keys


def label(k):
    return NAMES[k]


# --------------------------------------------------------------------------- figures
def fig_ranking():
    T = tournament()
    for arch, title, name in [('swin_b', 'Swin-B', 'tournament_ranking'), ('resnet50', 'ResNet-50', 'resnet_tournament_ranking')]:
        keys = all_arms(T, arch)
        n = len(keys)
        fig, ax = plt.subplots(figsize=(4.9, 4.2))
        rnd = T[arch]['random_control'][:, 7]
        real = T[arch]['real_only_baseline'][0, 7]
        ax.axvspan(rnd.mean() - rnd.std(ddof=1), rnd.mean() + rnd.std(ddof=1), color=GRAY, alpha=.22, lw=0)
        ax.axvline(rnd.mean(), color=GRAY, lw=1)
        ax.axvline(real, color=DARK, lw=1, ls='--')
        for i, k in enumerate(keys):
            y = n - 1 - i
            v = T[arch][k][:, 7]
            if k in DIST:
                ax.errorbar(v.mean(), y, xerr=v.std(ddof=1), fmt='o', color=BLUE, ms=4.5, lw=1, capsize=2, zorder=3)
            else:
                ax.plot(v.mean(), y, 'D', color=AQUA, ms=4.2, zorder=3)
        ax.set_xlabel('Macro-$F_2$ (validation)')
        ax.set_xlim(0.2, 0.6)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_yticks(range(n)); ax.set_yticklabels([label(k) for k in reversed(keys)])
        ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.grid(True, alpha=.3, lw=.5); ax.set_axisbelow(True)
        handles = [Line2D([], [], marker='o', color=BLUE, ls='', ms=4.5, label='Distributional arm (30 seeds, $\\pm$ 1 SD)'),
                   Line2D([], [], marker='D', color=AQUA, ls='', ms=4.2, label='Instance arm (deterministic dataset)'),
                   Patch(color=GRAY, alpha=.35, label='Random control (mean $\\pm$ 1 SD)'),
                   Line2D([], [], color=DARK, ls='--', lw=1, label='Real data')]
        fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.07))
        fig.tight_layout()
        save(fig, name)


def fig_heatmap():
    for arch, name in [('swin_b', 'tournament_heatmap')]:
        _heatmap(arch, name)


def _heatmap(arch, name):
    T = tournament()[arch]
    keys = all_arms(tournament(), arch)
    rnd_mean = T['random_control'].mean(0)
    cols = CLASSES + ['Macro']
    M = np.array([T[k].mean(0) for k in keys])
    D = M - rnd_mean
    cmap = LinearSegmentedColormap.from_list('div', [ORANGE, '#f7f6f3', BLUE])
    norm = Normalize(-0.25, 0.25)
    nrow = len(keys)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    xs = list(range(7)) + [7.35]
    for i, k in enumerate(keys):
        for j, x in enumerate(xs):
            d = D[i, j]
            ax.add_patch(Rectangle((x - .5, nrow - 1 - i - .5), 1, 1, fc=cmap(norm(d)), ec='white', lw=.8))
            ax.text(x, nrow - 1 - i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.3,
                    color='white' if abs(d) > .17 else 'black')
    for j, x in enumerate(xs):
        for i in np.flatnonzero(np.isclose(M[:, j], M[:, j].max(), rtol=0, atol=1e-12)):   # ties are all marked
            ax.add_patch(Rectangle((x - .5, nrow - 1 - i - .5), 1, 1, fc='none', ec='black', lw=1.5))
    for r, (nm, vec) in enumerate([('Random control', rnd_mean), ('Real data only', T['real_only_baseline'][0])]):
        y = -1.6 - r
        for j, x in enumerate(xs):
            ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc='#e8e7e3', ec='white', lw=.8))
            ax.text(x, y, f'{vec[j]:.2f}', ha='center', va='center', fontsize=6.3)
    ax.set_xlim(-.6, 7.9); ax.set_ylim(-3.2, nrow - .4)
    ax.set_xticks(xs); ax.set_xticklabels(cols); ax.xaxis.tick_top()
    ax.set_yticks(list(range(nrow - 1, -1, -1)) + [-1.6, -2.6])
    ax.set_yticklabels([label(k) for k in keys] + ['Random control', 'Real data only'])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=.03, pad=.02)
    cb.set_label('$F_2$ minus random control (same class)', fontsize=7)
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, name)


def fig_perclass_bars():
    T = tournament()['swin_b']
    synth = DIST + INST
    fid = T['fid_inception'][:, :7]
    rnd = T['random_control'][:, :7]
    bk = [max(synth, key=lambda k: T[k][:, j].mean()) for j in range(7)]
    bv = np.array([T[k][:, j].mean() for j, k in enumerate(bk)])
    x = np.arange(7); w = .27
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(x - w, fid.mean(0), w, yerr=fid.std(0, ddof=1), color=BLUE, label='FID-Inception', error_kw={'lw': .8, 'capsize': 1.5})
    ax.bar(x, rnd.mean(0), w, yerr=rnd.std(0, ddof=1), color=GRAY, label='Random control', error_kw={'lw': .8, 'capsize': 1.5})
    ax.bar(x + w, bv, w, color=ORANGE, label='Best metric for class')
    for i, k in enumerate(bk):
        tie = 'FID-DINOv2,\nKID-DINOv2' if k in ('fid_dino', 'kid_dino') else None
        ax.text(x[i] + w, bv[i] + .012, tie or (label(k).replace('-', '-\n', 1) if k in DIST else NAMES[k]),
                ha='center', va='bottom', fontsize=5.5, linespacing=.9)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES)
    ax.set_ylabel('Per-class $F_2$ (Swin-B)'); ax.set_ylim(0, 1.12)
    ax.spines[['top', 'right']].set_visible(False); ax.yaxis.grid(True, alpha=.3, lw=.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper right', ncol=3, bbox_to_anchor=(1, 1.12), fontsize=7)
    fig.tight_layout()
    save(fig, 'tournament_per_class')


def fig_weights():
    for arch, name in [('swin_b', 'weights_heatmap')]:
        _weights(arch, name)


def _weights(arch, name):
    W = locked_weights(arch) * 100
    order = ['fid_inception', 'kid_dino', 'swd_dino'] + INST
    W = W.loc[order]
    nrow = len(order)
    cmap = LinearSegmentedColormap.from_list('seq', ['#f7f6f3', BLUE])
    norm = Normalize(0, 36)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for i, k in enumerate(order):
        y = nrow - 1 - i + (0 if i < 3 else -.35)
        for j, c in enumerate(CLASSES):
            v = W.loc[k, c]
            ax.add_patch(Rectangle((j - .5, y - .5), 1, 1, fc=cmap(norm(v)), ec='white', lw=.8))
            ax.text(j, y, f'{v:.0f}' if v >= .5 else '', ha='center', va='center', fontsize=6.5,
                    color='white' if v > 21 else 'black')
    share = W.loc[['fid_inception', 'kid_dino', 'swd_dino']].sum()
    for j, c in enumerate(CLASSES):
        ax.text(j, nrow + .1, f'{share[c]:.0f}%', ha='center', va='bottom', fontsize=6.5, color=DARK)
    ax.text(-.7, nrow + .1, 'Distributional share', ha='right', va='bottom', fontsize=6.5, color=DARK)
    ax.set_xlim(-.6, 6.6); ax.set_ylim(-1.2, nrow + .9)
    ax.set_xticks(range(7)); ax.set_xticklabels(CLASSES); ax.xaxis.tick_top()
    ys = [nrow - 1 - i + (0 if i < 3 else -.35) for i in range(nrow)]
    ax.set_yticks(ys); ax.set_yticklabels([NAMES[k] for k in order])
    ax.axhline(nrow - 3 - .675, color='black', lw=.6, xmin=0, xmax=1)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=.035, pad=.02)
    cb.set_label('Weight (%)', fontsize=7); cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, name)


def fig_calibration():
    cal = pd.read_csv(DATA / 'fsms_validation/threshold_calibration.csv')
    pool = pd.read_csv(DATA / 'fsms_validation/pool_at_threshold.csv')
    start = {'0.900-0.920': .90, '0.920-0.930': .92, '0.930-0.940': .93, '0.940-0.950': .94, '0.950-0.965': .95,
             '0.965-0.970': .965, '0.970-0.980': .97, '0.980-0.990': .98, '>=0.990': .99}
    cal['tau'] = cal.band.map(start)
    x = cal.tau.values
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axs[0]
    ax.plot(x, 100 * cal.precision_at_or_above, '-o', color=BLUE, ms=3.5, lw=1.3)
    ax.plot(x, 100 * cal.recall_at_or_above, '-s', color=ORANGE, ms=3.5, lw=1.3)
    ax.text(0.925, 8, 'Precision', color=BLUE, fontsize=7.5)
    ax.text(0.935, 82, 'Recall', color=ORANGE, fontsize=7.5)
    ax.axhline(95, color=GRAY, lw=.7, ls=':'); ax.text(0.9455, 96.5, '95%', color=DARK, fontsize=6.5)
    ax.set_ylabel('Same-lesion pairs at or above $\\tau$ (%)'); ax.set_ylim(0, 105)
    ax = axs[1]
    p = pool.set_index('tau').flagged
    xs = [t for t in x if (np.abs(p.index.values - t) < 1e-9).any()]
    ys = [p[p.index[np.abs(p.index.values - t) < 1e-9][0]] for t in xs]
    ax.plot(xs, ys, '-o', color=DARK, ms=3.5, lw=1.3)
    ax.set_yscale('symlog', linthresh=1)
    ax.set_ylim(0, 1e5)
    ax.set_ylabel('Pool images removed (of 275,472)')
    for a in axs:
        for t, nm in [(t, t) for t, _ in TAUS]:
            a.axvline(t, color=GRAY, lw=.7, ls='--', zorder=0)
        a.set_xlabel('Cosine threshold $\\tau$')
        a.set_xticks([t for t, _ in TAUS]); a.set_xticklabels([f'{t:.2f}' for t, _ in TAUS])
        a.spines[['top', 'right']].set_visible(False)
        a.yaxis.grid(True, alpha=.3, lw=.5); a.set_axisbelow(True)
    fig.tight_layout()
    save(fig, 'calibration_curve')


def fig_forest():
    F = final()
    sw = lambda c: F[('swin_b', c)][:, 7]
    swin_rows = [('Algorithm ($\\tau=%.2f$) $-$ best-metric' % t, sw(k), sw('best_metric')) for t, k in TAUS]
    swin_rows += [None]
    swin_rows += [('Algorithm ($\\tau=%.2f$) $-$ random' % t, sw(k), sw('random_synth')) for t, k in TAUS]
    swin_rows += [None]
    swin_rows += [('Best-metric $-$ random', sw('best_metric'), sw('random_synth')),
                  ('Algorithm $-$ random ($N=20{,}000$)', sw('algorithm_n20000'), sw('random_synth_n20000')),
                  ('Algorithm: $N=20{,}000$ $-$ $N=8{,}012$', sw('algorithm_n20000'), sw('algorithm')),
                  ('Random: $N=20{,}000$ $-$ $N=8{,}012$', sw('random_synth_n20000'), sw('random_synth'))]
    for rows, col, name, size in [(swin_rows, ORANGE, 'benchmark_forest', (6.4, 3.9))]:
        fig, ax = plt.subplots(figsize=size)
        n = len(rows)
        labels = []
        for i, r in enumerate(rows):
            y = n - 1 - i
            if r is None:
                labels.append(''); continue
            nm, a, b = r
            d, lo, hi, p = welch(a, b)
            ax.plot([lo, hi], [y, y], color=col, lw=1.4)
            ax.plot(d, y, 'o', color=col, mfc=col if p < .05 else 'white', mew=1.2, ms=5)
            labels.append(nm)
        ax.axvline(0, color=DARK, lw=.8)
        ax.set_yticks(range(n)); ax.set_yticklabels(list(reversed(labels)), fontsize=7)
        ax.set_ylim(-.7, n - .3)
        ax.set_xlabel('Difference in test Macro-$F_2$ (95% CI)\npositive: the first-named cohort scores higher', fontsize=7.5, linespacing=1.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.grid(True, alpha=.3, lw=.5); ax.set_axisbelow(True)
        handles = [Line2D([], [], marker='o', color=DARK, mfc=DARK, ls='', ms=5, label='$p<0.05$'),
                   Line2D([], [], marker='o', color=DARK, mfc='white', ls='', ms=5, label='$p\\geq0.05$')]
        ax.legend(handles=handles, loc='lower right' if name == 'benchmark_forest' else 'upper right', frameon=False, fontsize=7)
        fig.tight_layout()
        save(fig, name)


def fig_benchmark():
    F = final()
    for arch, name, order, names, cols in [
        ('swin_b', 'benchmark_swin_b',
         ['random_synth', 'best_metric', 'tau090', 'tau093', 'tau097', 'tau099', 'random_synth_n20000', 'algorithm_n20000'],
         ['Random\n8,012', 'Best-metric\n8,012', 'Algorithm\n$\\tau=0.90$', 'Algorithm\n$\\tau=0.93$', 'Algorithm\n$\\tau=0.97$',
          'Algorithm\n$\\tau=0.99$', 'Random\n20,000', 'Algorithm\n20,000\n$\\tau=0.97$'],
         [GRAY, BLUE, ORANGE, ORANGE, ORANGE, ORANGE, GRAY, ORANGE]),
        ('resnet50', 'resnet_benchmark',
         ['random_synth', 'best_metric', 'algorithm', 'random_synth_n20000', 'algorithm_n20000'],
         ['Random\n8,012', 'Best-metric\n8,012', 'Algorithm\n8,012', 'Random\n20,000', 'Algorithm\n20,000'],
         [GRAY, BLUE, ORANGE, GRAY, ORANGE])]:
        ro = F[(arch, 'real_only')][0, 7]
        rng = np.random.default_rng(0)
        fig, ax = plt.subplots(figsize=(7.2 if arch == 'swin_b' else 5.2, 3.1))
        for i, (c, col) in enumerate(zip(order, cols)):
            v = F[(arch, c)][:, 7]
            ax.boxplot(v, positions=[i], widths=.5, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor='none', edgecolor=col, lw=1), medianprops=dict(color=col, lw=1.4),
                       whiskerprops=dict(color=col, lw=1), capprops=dict(color=col, lw=1))
            ax.scatter(i + rng.uniform(-.14, .14, len(v)), v, s=7, color=col, alpha=.55, lw=0)
            ax.scatter([i], [v.mean()], marker='D', s=16, color='white', edgecolor=col, zorder=5)
        ax.axhline(ro, color=DARK, ls='--', lw=.8)
        ax.text(len(order) - .55, ro + .002, 'Real data only', ha='right', va='bottom', fontsize=7, color=DARK)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(names, fontsize=6.5); ax.set_ylabel('Macro-$F_2$ (test)')
        ax.spines[['top', 'right']].set_visible(False); ax.yaxis.grid(True, alpha=.3, lw=.5); ax.set_axisbelow(True)
        fig.tight_layout()
        save(fig, name)


def fig_safety():
    F = final()
    FS = fsms_final()
    fid = F[('swin_b', 'best_metric')][:, 7]; rnd = F[('swin_b', 'random_synth')][:, 7]
    taus = [t for t, _ in TAUS]
    f2m = np.array([F[('swin_b', k)][:, 7].mean() for _, k in TAUS])
    f2s = np.array([F[('swin_b', k)][:, 7].std(ddof=1) for _, k in TAUS])
    fxm = np.array([FS[k].mean() for _, k in TAUS])
    anchor = pd.read_csv(DATA / 'fsms_validation/anchor.csv').fsms.iloc[0]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axs[0]
    ax.errorbar(taus, f2m, yerr=f2s, color=ORANGE, marker='o', ms=4, lw=1.3, capsize=2)
    ax.axhline(fid.mean(), color=BLUE, ls='--', lw=1); ax.text(0.9895, fid.mean() - .0042, 'Best-metric (FID)', color=BLUE, fontsize=7, ha='right')
    ax.axhline(rnd.mean(), color=GRAY, ls=':', lw=1); ax.text(0.9895, rnd.mean() - .0042, 'Random', color=GRAY, fontsize=7, ha='right')
    ax.set_ylim(0.545, 0.612); ax.set_ylabel('Macro-$F_2$ (test)'); ax.set_xlabel('Safety threshold $\\tau$')
    ax = axs[1]
    ax.plot(taus, fxm, color=ORANGE, marker='o', ms=4, lw=1.3)
    ax.axhline(FS['best_metric'].mean(), color=BLUE, ls='--', lw=1); ax.text(0.9005, FS['best_metric'].mean() + .0012, 'Best-metric (FID)', color=BLUE, fontsize=7)
    ax.axhline(FS['random_synth'].mean(), color=GRAY, ls=':', lw=1); ax.text(0.9005, FS['random_synth'].mean() - .0045, 'Random', color=GRAY, fontsize=7)
    ax.axhline(anchor, color=DARK, ls='-.', lw=.8); ax.text(0.9005, anchor - .0042, 'Real test vs. real train', color=DARK, fontsize=7)
    ax.set_ylim(0.1075, 0.1665); ax.set_ylabel('FSMS (higher = safer)'); ax.set_xlabel('Safety threshold $\\tau$')
    for ax in axs:
        ax.set_xticks(taus); ax.set_xticklabels([f'{t:.2f}' for t in taus], fontsize=7)
        ax.spines[['top', 'right']].set_visible(False); ax.yaxis.grid(True, alpha=.3, lw=.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, 'safety_vs_tau')


def fig_agreement():
    T = tournament()
    keys = merged_arms(T, 'swin_b')
    a = np.array([T['swin_b'][k][:, 7].mean() for k in keys])
    b = np.array([T['resnet50'][k][:, 7].mean() for k in keys])
    rho = stats.spearmanr(a, b)[0]
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    lim = (0.2, 0.56)
    ax.plot(lim, lim, color=GRAY, lw=.8, ls=':')
    for k, x, y in zip(keys, a, b):
        ax.plot(x, y, 'o' if k in DIST else 'D', color=BLUE if k in DIST else AQUA, ms=4.5)
    notable = {'fid_inception': (-18, 5), 'brisqe_score': (4, -8), 'diversity_score': (4, 3), 'ssim_score': (4, -8), 'fscs_score': (-38, 6)}
    for k, (dx, dy) in notable.items():
        i = keys.index(k)
        ax.annotate(label(k), (a[i], b[i]), xytext=(dx, dy), textcoords='offset points', fontsize=6.3)
    ax.text(0.04, 0.95, f'Spearman $\\rho = {rho:.2f}$', transform=ax.transAxes, va='top', fontsize=8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Swin-B Macro-$F_2$ (validation)'); ax.set_ylabel('ResNet-50 Macro-$F_2$ (validation)')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(True, alpha=.3, lw=.5); ax.set_axisbelow(True)
    handles = [Line2D([], [], marker='o', color=BLUE, ls='', ms=4.5, label='Distributional arm'),
               Line2D([], [], marker='D', color=AQUA, ls='', ms=4.2, label='Instance arm')]
    ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, 'arch_agreement')


FIGS = {'ranking': fig_ranking, 'heatmap': fig_heatmap, 'perclass_bars': fig_perclass_bars, 'weights': fig_weights,
        'calibration': fig_calibration, 'forest': fig_forest,
        'benchmark': fig_benchmark, 'safety': fig_safety, 'agreement': fig_agreement}

if __name__ == '__main__':
    want = sys.argv[1:] or list(FIGS)
    for name in want:
        FIGS[name]()
