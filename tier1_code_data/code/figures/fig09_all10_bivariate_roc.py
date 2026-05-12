#!/usr/bin/env python3
"""Fig. 9 (paper) — Bivariate logistic ROC curves for all 10 metric pairs.

All ten pairs are plotted in distinct colours (tab10 palette) so each pair is
individually identifiable in the legend; line width encodes ranking — the top
three pairs (by combined AUC) are drawn thicker, and ranks #4–#10 thinner —
providing visual emphasis on the (C_M, L_90) dominance without hiding the
remaining context.

Outputs:
  - 03_src/b2/figures/F2b_top3_roc.{png,pdf}                         (legacy name kept)
  - 01_paper/output/ieee_tim_v1/figures/F2b_top3_roc.{png,pdf}       (paper asset)
"""
from itertools import combinations
from pathlib import Path
import csv
import warnings

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

warnings.filterwarnings('ignore')

ROOT = Path('/Users/lch/home/code/tunnelscanning')
WF = ROOT / 'tmp/wide_full_50.csv'
LIQ = ROOT / 'tmp/long_iq_100.csv'
OUT_SRC = ROOT / '01_tunnelscanning/03_src/b2/figures'
OUT_PAPER = ROOT / '01_tunnelscanning/01_paper/output/ieee_tim_v1/figures'

METRICS = {
    'C_M':          ('wide', 'C_M',          +1),
    'L_90':         ('wide', 'L_90',         +1),
    'MTF50_H':      ('long', 'mtf_h',        +1),
    'BEW_H':        ('long', 'bew_h',        -1),
    'sigma_motion': ('long', 'sigma_motion', -1),
}

LABEL_MAP = {
    'C_M': '$C_M$',
    'L_90': '$L_{90}$',
    'MTF50_H': 'MTF50$_H$',
    'BEW_H': 'BEW$_H$',
    'sigma_motion': '$\\sigma_{motion}$',
}

PALETTE_10 = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # dark grey
    '#bcbd22',  # olive
    '#17becf',  # cyan
]


def fnum(s):
    s = (s or '').strip()
    try:
        return float(s)
    except ValueError:
        return None


def roc_curve(scores, labels):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    order = np.argsort(-s, kind='stable')
    y_s = y[order]
    s_s = s[order]
    P = int(y_s.sum())
    N = len(y_s) - P
    tp = fp = 0
    tprs, fprs = [0.0], [0.0]
    i = 0
    while i < len(y_s):
        j = i
        while j + 1 < len(y_s) and s_s[j + 1] == s_s[i]:
            j += 1
        for k in range(i, j + 1):
            if y_s[k] == 1:
                tp += 1
            else:
                fp += 1
        tprs.append(tp / P)
        fprs.append(fp / N)
        i = j + 1
    return np.asarray(fprs), np.asarray(tprs), float(np.trapezoid(tprs, fprs))


def build_paired_data(name1, name2, wf50, liq, cond_y):
    src1, col1, dir1 = METRICS[name1]
    src2, col2, dir2 = METRICS[name2]

    if src1 == 'wide' and src2 == 'wide':
        rows = []
        for r in wf50:
            k = (int(r['speed']), int(r['iso']), float(r['dist']))
            v1 = fnum(r[col1]); v2 = fnum(r[col2])
            if v1 is None or v2 is None:
                continue
            rows.append({name1: dir1 * v1, name2: dir2 * v2, 'Y': cond_y[k]})
        return rows

    if src1 == 'long' and src2 == 'long':
        rows = []
        for r in liq:
            k = (int(r['speed']), int(r['iso']), float(r['dist']))
            v1 = fnum(r[col1]); v2 = fnum(r[col2])
            if v1 is None or v2 is None:
                continue
            if (name1 == 'sigma_motion' or name2 == 'sigma_motion') and r.get('bew_hgv') != 'Y':
                continue
            rows.append({name1: dir1 * v1, name2: dir2 * v2, 'Y': cond_y[k]})
        return rows

    # mixed (wide × long): use cam1 only
    wide_idx = {(int(r['speed']), int(r['iso']), float(r['dist'])): r for r in wf50}
    long_col, long_dir = (col2, dir2) if src2 == 'long' else (col1, dir1)
    wide_col, wide_dir = (col1, dir1) if src1 == 'wide' else (col2, dir2)
    long_name = name2 if src2 == 'long' else name1
    wide_name = name1 if src1 == 'wide' else name2
    rows = []
    for r in liq:
        if r['camera'] != 'cam1':
            continue
        k = (int(r['speed']), int(r['iso']), float(r['dist']))
        wr = wide_idx.get(k)
        if wr is None:
            continue
        v_wide = fnum(wr[wide_col]); v_long = fnum(r[long_col])
        if v_wide is None or v_long is None:
            continue
        if long_name == 'sigma_motion' and r.get('bew_hgv') != 'Y':
            continue
        rows.append({wide_name: wide_dir * v_wide, long_name: long_dir * v_long, 'Y': cond_y[k]})
    return rows


def fit_and_curve(rows, name1, name2):
    if not rows:
        return None
    arr = np.array([[r[name1], r[name2], r['Y']] for r in rows], dtype=float)
    if arr[:, 2].sum() == 0 or arr[:, 2].sum() == len(arr):
        return None
    X = sm.add_constant(arr[:, :2])
    y = arr[:, 2].astype(int)
    try:
        mod = sm.Logit(y, X).fit(disp=0)
    except Exception:
        return None
    p = mod.predict(X)
    fpr, tpr, auc = roc_curve(p, y)
    return {'fpr': fpr, 'tpr': tpr, 'auc': auc, 'n': len(arr)}


def main() -> None:
    wf50 = list(csv.DictReader(open(WF)))
    liq = list(csv.DictReader(open(LIQ)))
    cond_y = {(int(r['speed']), int(r['iso']), float(r['dist'])):
              1 if r['identifiable'] == 'Y' else 0 for r in wf50}

    pairs = list(combinations(METRICS.keys(), 2))
    fits = {}
    for n1, n2 in pairs:
        rows = build_paired_data(n1, n2, wf50, liq, cond_y)
        f = fit_and_curve(rows, n1, n2)
        if f is None:
            continue
        fits[(n1, n2)] = f
    ranked = sorted(fits.items(), key=lambda kv: -kv[1]['auc'])

    fig, ax = plt.subplots(figsize=(7.6, 7.0), dpi=120)

    # Plot ranks #4-10 first (thinner, no individual emphasis but distinct colours)
    for rank, ((n1, n2), f) in enumerate(ranked[3:], 4):
        label = (f"#{rank} ({LABEL_MAP[n1]}, {LABEL_MAP[n2]})  "
                 f"AUC = {f['auc']:.3f}")
        ax.plot(f['fpr'], f['tpr'], color=PALETTE_10[rank - 1], lw=1.2,
                alpha=0.85, label=label, zorder=2)

    # Plot top-3 last so they sit on top (thicker, full opacity)
    for rank, ((n1, n2), f) in enumerate(ranked[:3], 1):
        label = (f"#{rank} ({LABEL_MAP[n1]}, {LABEL_MAP[n2]})  "
                 f"AUC = {f['auc']:.3f}")
        ax.plot(f['fpr'], f['tpr'], color=PALETTE_10[rank - 1], lw=2.6,
                label=label, zorder=3)

    # Chance baseline
    ax.plot([0, 1], [0, 1], 'k:', lw=1.0, label='Chance baseline', zorder=1)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    # Reorder legend by rank: #1, #2, ..., #10, then Chance baseline
    handles, labels = ax.get_legend_handles_labels()
    rank_order = []
    for target_rank in range(1, 11):
        for i, lab in enumerate(labels):
            if lab.startswith(f'#{target_rank} '):
                rank_order.append(i)
                break
    chance_idx = next(i for i, lab in enumerate(labels) if 'Chance' in lab)
    order = rank_order + [chance_idx]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=8.5, framealpha=0.95, borderpad=0.6)
    fig.tight_layout()

    for stem in (OUT_SRC / 'F2b_top3_roc', OUT_PAPER / 'F2b_top3_roc'):
        fig.savefig(f'{stem}.png', dpi=200, bbox_inches='tight')
        fig.savefig(f'{stem}.pdf', bbox_inches='tight')
        print(f'  saved: {stem}.png / .pdf')


if __name__ == '__main__':
    main()
