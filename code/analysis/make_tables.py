"""Builds every table of the paper from the repo's own data/ folder.

Usage:  python make_tables.py                      (writes .tex files to outputs/tables/)
        python make_tables.py tournament benchmark (only the named groups)

Groups and the tables they write
    pool         pool_composition, distributional_pool, backbone_arms
    tournament   tournament_overall_{swin_b,resnet50}, tournament_per_class[_resnet50],
                 tournament_paired_{swin_b,resnet50}, tournament_fid_per_class
    algorithm    algorithm_weights_full
    benchmark    benchmark_cohorts_{swin_b,resnet50}, benchmark_stats[_resnet50],
                 benchmark_per_class[_resnet50], benchmark_per_class_diff, benchmark_primary,
                 appendix_cohort_metrics_{swin_b,resnet50}, cohort_tail_stats
    safety       safety_pool_config, safety_detection, fsms_reference_points, fsms_crossmodal,
                 appendix_calibration_bands
    resnet       resnet_leaders, resnet_weights_compare, resnet_temperature

Per-class F2 is recomputed from the saved per-image predictions (beta = 2, one-vs-rest per class).
Tests between cohorts are unpaired because the curation procedures differ between cohorts.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import (ARCHS, CLASSES, DATA, DIST, INST, NAMES, TABLES, TAUS,
                    final, fsms_all, fsms_final, locked_weights, pvalue, signed, tournament, welch)


def write(name, rows):
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text('\n'.join(rows) + '\n')
    print('wrote', TABLES / name)


# =========================================================================== synthetic pool
def pool():
    # images per checkpoint x guidance scale x class
    d = pd.read_csv(DATA / 'dataset_release/metadata.csv',
                    usecols=['source_checkpoint_kimg', 'cfg_scale', 'target_class_label'])
    g = d.groupby(['source_checkpoint_kimg', 'cfg_scale', 'target_class_label']).size().unstack()[CLASSES]
    rows = []
    for ck in sorted(g.index.get_level_values(0).unique()):
        for i, (cfg, r) in enumerate(g.loc[ck].iterrows()):
            label = f'{ck // 1000}k' if i == 0 else ''
            rows.append(f"{label} & {cfg:.1f} & " + ' & '.join(f'{int(v):,}' for v in r) + f" & {int(r.sum()):,} \\\\")
        rows.append('\\midrule')
    tot = g.sum()
    rows.append("All 16 & & " + ' & '.join(f'{int(v):,}' for v in tot) + f" & {int(tot.sum()):,} \\\\")
    rows.append('\\bottomrule')
    write('pool_composition.tex', rows)

    # scores of the four distributional metrics for the 16 generation configurations (lower is better; best in bold)
    ck = pd.read_csv(DATA / 'synthetic_pool/config_distributional_metrics.csv')
    cols = [('fid_inception', 1), ('fid_dino', 1), ('kid_dino', 2), ('swd_dino', 3)]
    best = {k: ck[k].idxmin() for k, _ in cols}
    srt = ck.sort_values(['checkpoint', 'hyperparam'])
    rows = []
    for ci, (kimg, grp) in enumerate(srt.groupby('checkpoint')):
        if ci:
            rows.append('\\midrule')
        for ri, (idx, r) in enumerate(grp.iterrows()):
            cells = []
            for k, dgt in cols:
                v = f"{r[k]:.{dgt}f}"
                cells.append(f"\\textbf{{{v}}}" if idx == best[k] else v)
            rows.append(f"{f'{int(kimg) // 1000}k' if ri == 0 else ''} & {r.hyperparam:.1f} & " + ' & '.join(cells) + ' \\\\')
    rows.append('\\bottomrule')
    write('distributional_pool.tex', rows)

    # FID on the two feature backbones: selected configuration and validation Macro-F2 on each architecture
    T = tournament()
    rows = []
    for key, name, backbone in [('fid_inception', 'FID-Inception', 'InceptionV3'), ('fid_dino', 'FID-DINOv2', 'DINOv2')]:
        w = ck.loc[ck[key].idxmin()]
        cfg = f"{int(w.checkpoint) // 1000}k kimg, guidance {w.hyperparam:.1f}"
        cells = [f"${T[a][key][:, 7].mean():.3f} \\pm {T[a][key][:, 7].std(ddof=1):.3f}$" for a in ARCHS]
        rows.append(f"{name} & {backbone} & {cfg} & {cells[0]} & {cells[1]} \\\\")
    rows.append('\\midrule')
    cells = [f"${T[a]['random_control'][:, 7].mean():.3f} \\pm {T[a]['random_control'][:, 7].std(ddof=1):.3f}$" for a in ARCHS]
    rows.append(f"Random control & --- & --- & {cells[0]} & {cells[1]} \\\\")
    rows.append('\\bottomrule')
    write('backbone_arms.tex', rows)


# =========================================================================== metric tournament
def tournament_group():
    T = tournament()
    mean = {a: {k: v.mean(0) for k, v in T[a].items()} for a in T}
    sd = {a: {k: (v.std(0, ddof=1) if len(v) > 1 else None) for k, v in T[a].items()} for a in T}
    n_runs = {a: {k: len(v) for k, v in T[a].items()} for a in T}

    def cell(a, k):
        m = mean[a][k][7]
        s = sd[a][k]
        return f'{m:.3f} $\\pm$ {s[7]:.3f}' if s is not None else f'{m:.3f}'

    def delta(a, k):
        return f"{mean[a][k][7] - mean[a]['random_control'][7]:+.3f}".replace('-', '$-$')

    # overall Macro-F2 per arm
    for arch in ARCHS:
        lines = []
        srt = lambda keys: sorted(keys, key=lambda k: -mean[arch][k][7])
        for k in srt(DIST):
            lines.append(f"{NAMES[k]} & Distributional & 30 & {cell(arch, k)} & {delta(arch, k)} \\\\")
        lines.append('\\midrule')
        for k in srt(INST):
            lines.append(f"{NAMES[k]} & Instance & 1 & {cell(arch, k)} & {delta(arch, k)} \\\\")
        lines.append('\\midrule')
        lines.append(f"Random control & Baseline & {n_runs[arch]['random_control']} & {cell(arch, 'random_control')} & --- \\\\")
        lines.append(f"Real data only & Reference & 1 & {cell(arch, 'real_only_baseline')} & {delta(arch, 'real_only_baseline')} \\\\")
        lines.append('\\bottomrule')
        write(f'tournament_overall_{arch}.tex', lines)

    # per-class F2 per arm, best synthetic arm per column in bold
    synth = DIST + INST
    for a, fname in [('swin_b', 'tournament_per_class.tex'), ('resnet50', 'tournament_per_class_resnet50.tex')]:
        best = {j: max(mean[a][k][j] for k in synth) for j in range(8)}

        def pc_cells(k):
            out = []
            for j in range(8):
                t = f'{mean[a][k][j]:.3f}'
                if k in synth and abs(mean[a][k][j] - best[j]) < 1e-9:
                    t = f'\\textbf{{{t}}}'
                out.append(t)
            return ' & '.join(out)

        ordr = lambda keys: sorted(keys, key=lambda k: -mean[a][k][7])
        L = [f'{NAMES[k]} & {pc_cells(k)} \\\\' for k in ordr(DIST)] + ['\\midrule']
        L += [f'{NAMES[k]} & {pc_cells(k)} \\\\' for k in ordr(INST)] + ['\\midrule']
        L += [f'Random control & {pc_cells("random_control")} \\\\', f'Real data only & {pc_cells("real_only_baseline")} \\\\', '\\bottomrule']
        write(fname, L)

    # differences from the random control (Welch's t-test against all random-control runs of the architecture)
    def arm_vs_random(arch, arm):
        a, b = T[arch][arm], T[arch]['random_control']
        assert len(a) == 30
        return a, b

    for arch in ARCHS:
        rows = []
        # FID-DINOv2 and KID-DINOv2 select the same configuration, so both labels share one result
        for arm, labs in [('fid_inception', ['FID-Inception']), ('swd_dino', ['SWD-DINOv2']),
                          ('kid_dino', ['FID-DINOv2', 'KID-DINOv2'])]:
            x, y = arm_vs_random(arch, arm)
            t = stats.ttest_ind(x[:, 7], y[:, 7], equal_var=False)
            for lab in labs:
                rows.append(f"{lab} & {signed(x[:, 7].mean() - y[:, 7].mean(), 3)} & {pvalue(t.pvalue, 3)} \\\\")
        rows.append('\\bottomrule')
        write(f'tournament_paired_{arch}.tex', rows)

    a, b = arm_vs_random('swin_b', 'fid_inception')
    rows = []
    for j, c in enumerate(CLASSES + ['Macro']):
        t = stats.ttest_ind(a[:, j], b[:, j], equal_var=False)
        rows.append(f"{c} & {a[:, j].mean():.3f} & {b[:, j].mean():.3f} & {signed(a[:, j].mean() - b[:, j].mean(), 3)} & {pvalue(t.pvalue, 3)} \\\\")
        if c == 'VASC':
            rows.append('\\midrule')
    rows.append('\\bottomrule')
    write('tournament_fid_per_class.tex', rows)


# =========================================================================== evaluation algorithm
def algorithm():
    # per-class weights of the locked configuration (Swin-B), in percent
    active = ['fid_inception', 'kid_dino', 'swd_dino', 'clip_log_odds_score', 'fscs_score',
              'clip_similarity_cancer', 'clip_similarity_artifact', 'lpips_score', 'fsim_score',
              'brisqe_score', 'psnr_score', 'ssim_score', 'diversity_score', 'realism_score',
              'is_contribution_score']
    labels = {'fid_inception': 'FID-Inc.', 'kid_dino': 'KID-DINO', 'swd_dino': 'SWD-DINO',
              'clip_log_odds_score': 'CLIP log-odds', 'fscs_score': 'FS-CS',
              'clip_similarity_cancer': 'CLIP cancer', 'clip_similarity_artifact': 'CLIP artifact',
              'lpips_score': 'LPIPS', 'fsim_score': 'FSIM', 'brisqe_score': 'BRISQUE', 'psnr_score': 'PSNR',
              'ssim_score': 'SSIM', 'diversity_score': 'Diversity', 'realism_score': 'Realism',
              'is_contribution_score': 'IS contr.'}
    dist = ['fid_inception', 'kid_dino', 'swd_dino']
    W = locked_weights('swin_b').T.loc[CLASSES]
    order = dist + [m for m in active if m not in dist]
    L = []
    for m in order:
        L.append(labels[m] + ' & ' + ' & '.join((f'{100 * W.loc[c, m]:.1f}' if 100 * W.loc[c, m] >= 0.05 else '0') for c in CLASSES) + ' \\\\')
        if m == dist[-1]:
            L.append('\\midrule')
    L.append('\\midrule')
    L.append('Total & ' + ' & '.join('100' for _ in CLASSES) + ' \\\\')
    L.append('\\bottomrule')
    write('algorithm_weights_full.tex', L)


# =========================================================================== clinical augmentation benchmark
LABEL = {'real_only': 'Real data only', 'random_synth': 'Random', 'best_metric': 'Best-metric (FID-Inception)',
         'algorithm': 'Algorithm', 'random_synth_n20000': 'Random', 'algorithm_n20000': 'Algorithm'}
TAU_LABELS = [('0.90', 'tau090'), ('0.93', 'tau093'), ('0.97', 'tau097'), ('0.99', 'tau099')]


def _pm(x, k=4):
    return f'{x.mean():.{k}f} $\\pm$ {x.std(ddof=1):.{k}f}' if len(x) > 1 else f'{x.mean():.{k}f}'


def _cohort_rows(arch):
    """Ordered (key, label, N) rows. Swin-B reports all four tau values at N=8,012."""
    if arch == 'swin_b':
        alg8 = [(k, f'Algorithm, $\\tau={t}$', '8{,}012') for t, k in TAU_LABELS]
    else:
        alg8 = [('algorithm', 'Algorithm, $\\tau=0.97$', '8{,}012')]
    return ([('real_only', 'Real data only', None), ('random_synth', 'Random', '8{,}012'),
             ('best_metric', 'Best-metric (FID-Inception)', '8{,}012')] + alg8 +
            [('random_synth_n20000', 'Random', '20{,}000'),
             ('algorithm_n20000', 'Algorithm, $\\tau=0.97$', '20{,}000')])


def benchmark():
    R = final()
    FS = fsms_all()

    # cohort table: test Macro-F2 and FSMS per cohort
    for arch, fname in [('swin_b', 'benchmark_cohorts_swin_b.tex'), ('resnet50', 'benchmark_cohorts_resnet50.tex')]:
        last = 'tau099' if arch == 'swin_b' else 'algorithm'
        rows = []
        for c, lab, n in _cohort_rows(arch):
            r = R[(arch, c)][:, 7]
            f = FS.get((arch, c))
            rows.append(f"{lab} & {'---' if n is None else '$' + n + '$'} & {len(r)} & {_pm(r)} & {_pm(f) if f is not None else '---'} \\\\")
            if c in ('real_only', 'best_metric'):
                rows.append('\\midrule')
            if c == last:
                rows.append('\\midrule')
        rows.append('\\bottomrule')
        write(fname, rows)

    def compare(a, b, arch='swin_b'):
        x = R[(arch, a)][:, 7]
        y = R[(arch, b)][:, 7]
        w = stats.ttest_ind(x, y, equal_var=False)
        mw = stats.mannwhitneyu(x, y, alternative='two-sided')
        sp = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        h = stats.t.ppf(.975, w.df) * se
        d = x.mean() - y.mean()
        return dict(d=d, lo=d - h, hi=d + h, pw=w.pvalue, pm=mw.pvalue, cd=d / sp)

    cd2 = lambda x: f'{x:.2f}'.replace('-', '$-$')

    # Swin-B comparisons: Welch CI, effect size, Bonferroni (algorithm vs best metric) and Mann-Whitney
    rows = []
    for t, c in TAU_LABELS:
        s = compare(c, 'best_metric')
        rows.append(f"Algorithm ($\\tau={t}$) vs.\\ Best-metric & {signed(s['d'])} & [{signed(s['lo'])}, {signed(s['hi'])}] & {cd2(s['cd'])} & "
                    f"{pvalue(s['pw'])} & {pvalue(min(1, 4 * s['pw']))} & {pvalue(s['pm'])} \\\\")
    rows.append('\\midrule')
    sec = [(f'Algorithm ($\\tau={t}$) vs.\\ Random, $N=8{{,}}012$', c, 'random_synth') for t, c in TAU_LABELS]
    sec += [('Best-metric vs.\\ Random, $N=8{,}012$', 'best_metric', 'random_synth'),
            ('Algorithm ($\\tau=0.97$) vs.\\ Random, $N=20{,}000$', 'algorithm_n20000', 'random_synth_n20000'),
            ('Algorithm ($\\tau=0.97$), $N=20{,}000$ vs.\\ $N=8{,}012$', 'algorithm_n20000', 'algorithm'),
            ('Random $N=20{,}000$ vs.\\ $N=8{,}012$', 'random_synth_n20000', 'random_synth')]
    for lab, a, b in sec:
        s = compare(a, b)
        rows.append(f"{lab} & {signed(s['d'])} & [{signed(s['lo'])}, {signed(s['hi'])}] & {cd2(s['cd'])} & {pvalue(s['pw'])} & --- & {pvalue(s['pm'])} \\\\")
    rows.append('\\bottomrule')
    write('benchmark_stats.tex', rows)

    # per-class test F2 per cohort
    for arch, fname, breaks in [('swin_b', 'benchmark_per_class.tex', ('real_only', 'best_metric', 'tau099')),
                                ('resnet50', 'benchmark_per_class_resnet50.tex', ('real_only', 'best_metric', 'algorithm'))]:
        rows = []
        for c, lab, n in _cohort_rows(arch):
            v = R[(arch, c)].mean(0)
            suffix = '' if n is None else ' ($N=' + n + '$)'
            rows.append(f"{lab}{suffix} & " + ' & '.join(f'{x:.3f}' for x in v) + ' \\\\')
            if c in breaks:
                rows.append('\\midrule')
        rows.append('\\bottomrule')
        write(fname, rows)

    # per-class differences of the algorithm at each tau
    def cmp_class(a, b, j):
        x = R[('swin_b', a)][:, j]
        y = R[('swin_b', b)][:, j]
        return x.mean() - y.mean(), stats.ttest_ind(x, y, equal_var=False).pvalue

    rows = []
    for ref, title in [('best_metric', 'vs.\\ Best-metric (FID-Inception)'), ('random_synth', 'vs.\\ Random')]:
        rows.append(f"\\textit{{Algorithm {title}}} & & & & & & & & \\\\")
        for j, c in enumerate(CLASSES + ['Macro']):
            cells = []
            for t, key in TAU_LABELS:
                d, p = cmp_class(key, ref, j)
                cells += [signed(d, 3), pvalue(p)]
            rows.append(f"{c} & " + ' & '.join(cells) + ' \\\\')
        rows.append('\\midrule')
    rows[-1] = '\\bottomrule'
    write('benchmark_per_class_diff.tex', rows)

    # ResNet-50 comparisons
    rows = []
    for lab, a, b in [('Algorithm vs.\\ Best-metric, $N=8{,}012$', 'algorithm', 'best_metric'),
                      ('Algorithm vs.\\ Random, $N=8{,}012$', 'algorithm', 'random_synth'),
                      ('Best-metric vs.\\ Random, $N=8{,}012$', 'best_metric', 'random_synth'),
                      ('Algorithm vs.\\ Random, $N=20{,}000$', 'algorithm_n20000', 'random_synth_n20000'),
                      ('Algorithm, $N=20{,}000$ vs.\\ $N=8{,}012$', 'algorithm_n20000', 'algorithm')]:
        s = compare(a, b, 'resnet50')
        rows.append(f"{lab} & {signed(s['d'])} & [{signed(s['lo'])}, {signed(s['hi'])}] & {cd2(s['cd'])} & {pvalue(s['pw'])} \\\\")
    rows.append('\\bottomrule')
    write('benchmark_stats_resnet50.tex', rows)

    # primary tests: algorithm minus best-metric at the four thresholds, on test Macro-F2 and on FSMS (Bonferroni over four)
    F = final()
    FSM = fsms_final()
    outcomes = [('Macro-$F_2$', {k: F[('swin_b', k)][:, 7] for _, k in TAUS}, F[('swin_b', 'best_metric')][:, 7]),
                ('FSMS', {k: FSM[k] for _, k in TAUS}, FSM['best_metric'])]
    rows = []
    for oi, (name, arms, best) in enumerate(outcomes):
        for ti, (tau, key) in enumerate(TAUS):
            a = arms[key]
            t = stats.ttest_ind(a, best, equal_var=False)
            d, lo, hi, p = welch(a, best)
            sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(best) - 1) * best.var(ddof=1)) / (len(a) + len(best) - 2))
            dd = d / sp
            tt = t.statistic
            rows.append(f"{name if ti == 0 else ''} & {tau:.2f} & {signed(d)} & [{signed(lo)}, {signed(hi)}] & "
                        f"{str(f'{tt:.2f}').replace('-', '$-$')} & {pvalue(p, floor_exp=6)} & {pvalue(min(1, 4 * p), floor_exp=6)} & "
                        f"{str(f'{dd:.2f}').replace('-', '$-$')} \\\\")
        if oi == 0:
            rows.append('\\midrule')
    rows.append('\\bottomrule')
    write('benchmark_primary.tex', rows)

    # appendix: all test metrics per cohort
    m = pd.read_csv(DATA / 'benchmark/final_results.csv')
    s = pd.read_csv(DATA / 'benchmark/safety_sweep/final_results_raw.csv')
    s['arch'] = 'swin_b'
    cols = ['macro_f2', 'macro_f1', 'macro_auc', 'macro_recall', 'macro_precision', 'global_acc']
    lab = {'real_only': ('Real data only', '---'), 'random_synth': ('Random', '8{,}012'),
           'best_metric': ('Best-metric (FID-Inception)', '8{,}012'),
           'algorithm': ('Algorithm, $\\tau=0.97$', '8{,}012'),
           'random_synth_n20000': ('Random', '20{,}000'), 'algorithm_n20000': ('Algorithm, $\\tau=0.97$', '20{,}000')}
    tau = [('algorithm_safety090', '0.90'), ('algorithm_safety093', '0.93'), ('algorithm_safety099', '0.99')]

    def fmt(x):
        return f'{x.mean():.3f}' + (f' $\\pm$ {x.std(ddof=1):.3f}' if len(x) > 1 else '')

    for arch in ARCHS:
        d = pd.concat([m[m.arch == arch], s[s.arch == arch]]) if arch == 'swin_b' else m[m.arch == arch]
        rows = []

        def add(key, name, n):
            g = d[d.cohort == key]
            rows.append(f"{name} & {'$' + n + '$' if n != '---' else n} & " + ' & '.join(fmt(g[c]) for c in cols) + ' \\\\')

        add('real_only', *lab['real_only'])
        rows.append('\\midrule')
        add('random_synth', *lab['random_synth'])
        add('best_metric', *lab['best_metric'])
        rows.append('\\midrule')
        if arch == 'swin_b':
            for key, t in tau[:2]:
                add(key, f'Algorithm, $\\tau={t}$', '8{,}012')
            add('algorithm', *lab['algorithm'])
            add(tau[2][0], f'Algorithm, $\\tau={tau[2][1]}$', '8{,}012')
        else:
            add('algorithm', *lab['algorithm'])
        rows.append('\\midrule')
        add('random_synth_n20000', *lab['random_synth_n20000'])
        add('algorithm_n20000', *lab['algorithm_n20000'])
        rows.append('\\bottomrule')
        write(f'appendix_cohort_metrics_{arch}.tex', rows)

    # similarity tails of the curated sets (Swin-B cohorts)
    seed = pd.read_csv(DATA / 'benchmark/tail_stats/tail_stats_per_seed.csv')
    # The tau = 0.97 cohort was curated for both classifiers. The table reports its Swin-B cohort.
    seed = seed[~((seed.cohort == 'algorithm') & ~seed.manifest.str.contains('swin_b'))]
    summ = seed.groupby('cohort').mean(numeric_only=True)
    items = [('random_synth', 'Random'), ('best_metric', 'Best-metric (FID-Inception)'), None,
             ('algorithm_safety090', 'Algorithm ($\\tau=0.90$)'), ('algorithm_safety093', 'Algorithm ($\\tau=0.93$)'),
             ('algorithm', 'Algorithm ($\\tau=0.97$)'), ('algorithm_safety099', 'Algorithm ($\\tau=0.99$)')]
    rows = []
    for item in items:
        if item is None:
            rows.append('\\midrule')
            continue
        key, label = item
        sm = summ.loc[key]
        worst = seed.loc[seed.cohort == key, 'max_sim'].max()
        rows.append(f"{label} & {sm.max_sim:.4f} & {worst:.4f} & {sm.p99_9:.4f} & "
                    f"{sm['count_ge_0.95']:.1f} ({sm['share_ge_0.95_pct']:.2f}\\%) & "
                    f"{sm['count_ge_0.97']:.1f} ({sm['share_ge_0.97_pct']:.3f}\\%) \\\\")
    rows.append('\\bottomrule')
    write('cohort_tail_stats.tex', rows)


# =========================================================================== memorization score
def safety():
    V = DATA / 'fsms_validation'

    # FSMS of the pool by generation configuration
    cfg = pd.read_csv(V / 'pool_by_config.csv')
    piv = cfg.pivot(index='checkpoint_kimg', columns='cfg_scale', values='fsms')
    rows = []
    for ck, r in piv.iterrows():
        cells = []
        for c in [1.0, 1.5, 2.0, 3.0]:
            v = f'{r[c]:.4f}'
            if ck == 25 and c == 1.0:
                v = f'\\textbf{{{v}}}'
            cells.append(v)
        rows.append(f'{ck}k & ' + ' & '.join(cells) + ' \\\\')
    rows.append('\\bottomrule')
    write('safety_pool_config.tex', rows)

    # detection of planted copies
    det = pd.read_csv(V / 'detection.csv')
    rows = []
    for cont in [0.01, 0.05, 0.10, 0.25]:
        a = det[(det.jitter == 0.0) & (det.contamination == cont)].iloc[0]
        b = det[(det.jitter == 0.3) & (det.contamination == cont)].iloc[0]
        rows.append(f"{int(cont * 100)}\\% ({int(a.n_planted)}) & {a.drop_vs_clean:.4f} & {a.auroc:.3f} & {b.drop_vs_clean:.4f} & {b.auroc:.3f} \\\\")
    rows.append('\\bottomrule')
    write('safety_detection.tex', rows)

    # reference FSMS values: real training set against itself, against its other images, and the held-out test images
    fl = pd.read_csv(V / 'floor.csv')
    an = pd.read_csv(V / 'anchor.csv').iloc[0]
    other = fl[fl.set == 'real_train vs other real'].iloc[0]
    rows = [
        "Training images against themselves & 0 & 1.0000 & --- & --- & --- \\\\",
        f"Training images against the other training images & {other.fsms:.4f} & {other.mean_cosine:.4f} & {other['pct_ge_0.95']:.2f} & --- & {other['max']:.3f} \\\\",
        f"Held-out test images against the training images & {an.fsms:.4f} & {an.mean_max_sim:.4f} & {an['pct_ge_0.95']:.2f} & {an['pct_ge_0.97']:.2f} & {an['max']:.3f} \\\\",
        '\\bottomrule',
    ]
    write('fsms_reference_points.tex', rows)

    # pixel-level agreement of the DINOv2 similarity: five similarity brackets of the pool and the correlations
    br = pd.read_csv(V / 'crossmodal_brackets.csv')
    co = pd.read_csv(V / 'crossmodal_correlations.csv').set_index('metric')
    names = {'bottom 50%': 'Least similar 50\\%', 'p50-p90': '50th to 90th percentile',
             'p90-p99': '90th to 99th percentile', 'p99-p99.9': '99th to 99.9th percentile',
             'top 0.1%': 'Most similar 0.1\\%'}
    rows = []
    for _, r in br.iterrows():
        rows.append(f"{names[r.bracket]} & {int(r.n):,} & {r.mean_cosine:.3f} & {r.ssim:.3f} & {r.psnr:.1f} & {r.lpips:.3f} \\\\")
    rows.insert(len(rows), '\\midrule')
    sg = lambda x: f'{x:.2f}'.replace('-', '$-$')
    rows.append(f"Correlation with the cosine ($r$) & & & {sg(co.loc['SSIM', 'pearson'])} & {sg(co.loc['PSNR', 'pearson'])} & {sg(co.loc['LPIPS', 'pearson'])} \\\\")
    rows.append('\\bottomrule')
    write('fsms_crossmodal.tex', rows)

    # calibration of the threshold: same-lesion precision and recall by cosine band
    cal = pd.read_csv(V / 'threshold_calibration.csv')
    rows = []
    for _, r in cal.iterrows():
        band = r.band.replace('>=', '$\\ge$').replace('-', '--')
        rows.append(f"{band} & {int(r.pairs):,} & {int(r.same_lesion)} & {100 * r.precision_in_band:.1f}\\% & {int(r.pairs_at_or_above):,} & {100 * r.precision_at_or_above:.1f}\\% & {100 * r.recall_at_or_above:.1f}\\% \\\\")
    rows.append('\\bottomrule')
    write('appendix_calibration_bands.tex', rows)

    # one-way ANOVA and Kruskal-Wallis of test Macro-F2 across the four thresholds (printed, quoted in the text)
    m = pd.read_csv(DATA / 'benchmark/final_results.csv')
    s = pd.read_csv(DATA / 'benchmark/safety_sweep/final_results_raw.csv')
    sw = m[m.arch == 'swin_b']
    sets = {0.97: sw[sw.cohort == 'algorithm']}
    for c in s.cohort.unique():
        sets[float('0.' + c[-2:])] = s[s.cohort == c]
    allf2 = [sets[t].macro_f2.values for t in [0.90, 0.93, 0.97, 0.99]]
    print('ANOVA across tau', stats.f_oneway(*allf2), 'Kruskal', stats.kruskal(*allf2))


# =========================================================================== ResNet-50
def resnet():
    T = tournament()
    arms = DIST + INST
    keys = [k for k in arms if k != 'kid_dino']

    # best arm per class on each architecture and the rank correlation of the two rankings
    rows = []
    for j, c in enumerate(CLASSES):
        cells = []
        for arch in ARCHS:
            k = max(arms, key=lambda a: T[arch][a][:, j].mean())
            name = 'FID-DINOv2/KID-DINOv2' if k in ('fid_dino', 'kid_dino') else NAMES[k]
            cells.append(f"{name} ({T[arch][k][:, j].mean():.3f})")
        x = [T['swin_b'][k][:, j].mean() for k in keys]
        y = [T['resnet50'][k][:, j].mean() for k in keys]
        rho = stats.spearmanr(x, y)[0]
        rows.append(f"{c} & {cells[0]} & {cells[1]} & {rho:.2f} \\\\")
    rows.append('\\bottomrule')
    write('resnet_leaders.tex', rows)

    # weights on the two architectures
    dist = ['fid_inception', 'kid_dino', 'swd_dino']
    W = {a: locked_weights(a) * 100 for a in ARCHS}
    rows = []
    for c in CLASSES:
        cells = []
        for a in ARCHS:
            w = W[a][c]
            top = w.idxmax()
            cells += [f"{w[dist].sum():.0f}\\%", f"{NAMES[top]} ({w[top]:.0f}\\%)"]
        rows.append(f"{c} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")
    rows.append('\\bottomrule')
    write('resnet_weights_compare.tex', rows)

    # ResNet-50 temperature sweep with the locked configuration: validation Macro-F2 at seven temperatures
    x = pd.read_csv(DATA / 'design_choices/temperature_sweep.csv')
    r = x[x.Model_Architecture == 'resnet50']
    temps = [(0.02, 'edm2_algorithm_resnet_t002'), (0.05, 'edm2_algorithm_resnet_t005'), (0.1, 'edm2_algorithm_resnet_t010'),
             (0.3, 'edm2_algorithm_resnet_t030'), (0.5, 'edm2_algorithm_resnet_t050'), (1.0, 'edm2_algorithm_resnet_t100'),
             (2.0, 'edm2_algorithm_resnet_t200')]
    v = {t: r[r.Dataset_Condition.str.fullmatch(a + r'_seed\d+')].Macro_F2.values for t, a in temps}
    assert all(len(a) == 30 for a in v.values())
    ref = v[0.05]
    rows = []
    for t, a in temps:
        cell = f"${v[t].mean():.4f} \\pm {v[t].std(ddof=1):.4f}$"
        if t == 0.05:
            rows.append(f"{t} & {cell} & --- & --- \\\\")
        else:
            tt = stats.ttest_ind(v[t], ref, equal_var=False)
            rows.append(f"{t} & {cell} & {signed(v[t].mean() - ref.mean())} & {pvalue(tt.pvalue, floor_exp=6)} \\\\")
    rows.append('\\bottomrule')
    write('resnet_temperature.tex', rows)


GROUPS = {'pool': pool, 'tournament': tournament_group, 'algorithm': algorithm,
          'benchmark': benchmark, 'safety': safety, 'resnet': resnet}

if __name__ == '__main__':
    for name in sys.argv[1:] or list(GROUPS):
        GROUPS[name]()
