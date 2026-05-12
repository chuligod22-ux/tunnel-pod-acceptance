#!/usr/bin/env python3
"""Sec 3.3 Image Quality Metrics — visual illustration of the 5-metric IQ suite.

4-panel figure:
  (a) ROI luminance histogram (real cam1 frame) → C_M and L_90 definition
  (b) Synthetic ESF (Gaussian-blurred step) → BEW (10%–90% rise width) definition
  (c) Synthetic MTF curve (Gaussian transfer) → MTF50 definition
  (d) Scatter BEW_H vs BEW_V (real data, n=100 cam1+cam2) → σ_motion isolation

Outputs:
  - 03_src/b2/figures/F12_iq_metrics_definitions.{png,pdf}
  - 01_paper/output/ieee_tim_v1/figures/F12_iq_metrics_definitions.{png,pdf}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import norm

PROJECT = Path('/Users/lch/home/code/tunnelscanning/01_tunnelscanning')
ROOT = PROJECT / '03_src' / 'b2' / 'figures'
PAPER_FIG = PROJECT / '01_paper' / 'output' / 'ieee_tim_v1' / 'figures'

REPRESENTATIVE_FRAME = PROJECT / '03_src' / 'data' / 'raw' / 'crack' / 'cam1' / 'cam1_v60_iso200_d25.png'
LONG_IQ_CSV = Path('/Users/lch/home/code/tunnelscanning/tmp/long_iq_100.csv')

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
})


def load_roi_luminance() -> tuple[np.ndarray, float, float, float]:
    """Load representative cam1 frame, extract Sec 3.3 ROI, return luminance + C_M, L_10, L_90."""
    img = np.asarray(Image.open(REPRESENTATIVE_FRAME).convert('L'))
    h, w = img.shape
    # Sec 3.3 ROI: rows 3–60%, columns 15–85%
    roi = img[int(0.03 * h):int(0.60 * h), int(0.15 * w):int(0.85 * w)]
    lum = roi.flatten().astype(np.float32)
    l10 = float(np.percentile(lum, 10))
    l90 = float(np.percentile(lum, 90))
    cm = (l90 - l10) / (l90 + l10)
    return lum, cm, l10, l90


SUBCAPTION_KW = {
    'transform': None,  # filled in by helper
    'ha': 'center',
    'va': 'top',
    'fontsize': 10,
    'fontweight': 'bold',
}


def add_subcaption(ax: plt.Axes, text: str) -> None:
    """Place subcaption at bottom-center of the panel, below the x-axis label."""
    ax.text(0.5, -0.22, text, transform=ax.transAxes,
            ha='center', va='top', fontsize=10)


def panel_a_histogram(ax: plt.Axes) -> None:
    lum, cm, l10, l90 = load_roi_luminance()
    ax.hist(lum, bins=64, range=(0, 255), color='#4a7ab8', edgecolor='white', linewidth=0.3)
    ax.axvline(l10, color='#d9534f', linestyle='--', linewidth=1.5, label=f'$L_{{10}}={l10:.0f}$ DN')
    ax.axvline(l90, color='#5cb85c', linestyle='--', linewidth=1.5, label=f'$L_{{90}}={l90:.0f}$ DN')
    ax.set_xlim(0, 255)
    ax.set_xlabel('Pixel luminance (DN)')
    ax.set_ylabel('Pixel count')
    eq_txt = (
        rf'$C_{{M}}=\frac{{L_{{90}}-L_{{10}}}}{{L_{{90}}+L_{{10}}}}'
        rf'=\frac{{{l90:.0f}-{l10:.0f}}}{{{l90:.0f}+{l10:.0f}}}={cm:.3f}$'
    )
    ax.text(0.97, 0.97, eq_txt, transform=ax.transAxes, ha='right', va='top',
            fontsize=8.5, bbox={'facecolor': 'white', 'edgecolor': '#888', 'pad': 4})
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    add_subcaption(ax, '(a) Exposure metrics — ROI luminance histogram')


def panel_b_esf(ax: plt.Axes, bew_h: float = 3.92) -> None:
    """Synthetic edge spread function (Gaussian-blurred step) with BEW_H from real data."""
    sigma = bew_h / (norm.ppf(0.90) - norm.ppf(0.10))  # invert 10–90% rule
    x = np.linspace(-8, 8, 401)
    esf = norm.cdf(x, scale=sigma)
    x10 = sigma * norm.ppf(0.10)
    x90 = sigma * norm.ppf(0.90)
    ax.plot(x, esf, color='#4a7ab8', linewidth=2.0)
    ax.axhline(0.10, color='#d9534f', linestyle=':', linewidth=1.0)
    ax.axhline(0.90, color='#5cb85c', linestyle=':', linewidth=1.0)
    ax.axvline(x10, color='#d9534f', linestyle='--', linewidth=1.0)
    ax.axvline(x90, color='#5cb85c', linestyle='--', linewidth=1.0)
    ax.annotate('', xy=(x90, 0.05), xytext=(x10, 0.05),
                arrowprops={'arrowstyle': '<->', 'color': 'black', 'lw': 1.4})
    ax.text((x10 + x90) / 2, 0.13, f'BEW = {bew_h:.2f} px', ha='center', va='bottom',
            fontsize=9, bbox={'facecolor': 'white', 'edgecolor': '#888', 'pad': 3})
    ax.scatter([x10, x90], [0.10, 0.90], color=['#d9534f', '#5cb85c'], zorder=5, s=40)
    ax.text(-7.5, 0.93, '90%', color='#5cb85c', fontsize=8.5)
    ax.text(-7.5, 0.05, '10%', color='#d9534f', fontsize=8.5)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Distance from edge (pixels)')
    ax.set_ylabel('Normalised ESF')
    add_subcaption(ax, '(b) Resolution metric — Edge spread function and BEW')


def panel_c_mtf(ax: plt.Axes, mtf50_h: float = 0.128) -> None:
    """Synthetic MTF curve (Gaussian transfer fitting actual MTF50_H)."""
    # Gaussian MTF: MTF(f) = exp(-2 (pi sigma_psf f)^2). Solve for sigma_psf at MTF=0.5, f=mtf50.
    sigma_psf = np.sqrt(np.log(2) / 2) / (np.pi * mtf50_h)
    f = np.linspace(0, 0.5, 501)
    mtf = np.exp(-2.0 * (np.pi * sigma_psf * f) ** 2)
    ax.plot(f, mtf, color='#4a7ab8', linewidth=2.0)
    ax.axhline(0.5, color='#d9534f', linestyle=':', linewidth=1.0)
    ax.axvline(mtf50_h, color='#d9534f', linestyle='--', linewidth=1.2)
    ax.scatter([mtf50_h], [0.5], color='#d9534f', zorder=5, s=50)
    ax.text(mtf50_h + 0.01, 0.55,
            f'MTF50 = {mtf50_h:.3f} cy/px',
            color='#d9534f', fontsize=8.5,
            bbox={'facecolor': 'white', 'edgecolor': '#888', 'pad': 3})
    ax.text(0.01, 0.53, 'contrast = 0.5', color='#d9534f', fontsize=8)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Spatial frequency (cycles/pixel)')
    ax.set_ylabel('Normalised MTF')
    add_subcaption(ax, '(c) Resolution metric — MTF and MTF50')


def panel_d_motion_blur(ax: plt.Axes) -> None:
    df = pd.read_csv(LONG_IQ_CSV)
    sub = df[df['bew_h'].notna() & df['bew_v'].notna()].copy()
    motion_regime = sub['bew_hgv'].astype(str).str.upper() == 'Y'
    excluded = ~motion_regime
    ax.scatter(sub.loc[motion_regime, 'bew_v'], sub.loc[motion_regime, 'bew_h'],
               s=22, color='#4a7ab8', alpha=0.7, edgecolor='white', linewidth=0.4,
               label=f'BEW$_H>$BEW$_V$ (n={int(motion_regime.sum())}, motion regime)')
    ax.scatter(sub.loc[excluded, 'bew_v'], sub.loc[excluded, 'bew_h'],
               s=32, color='#d9534f', alpha=0.85, marker='x', linewidth=1.2,
               label=f'excluded (n={int(excluded.sum())}, regime undefined)')
    lim_lo = float(min(sub['bew_h'].min(), sub['bew_v'].min())) - 0.5
    lim_hi = 11.0  # extended beyond data max (~8 px) to create empty bottom-right space for the legend
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color='black', linestyle=':', linewidth=1.0,
            label=r'BEW$_H=$BEW$_V$ (no motion blur)')
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel('BEW$_V$ (pixels) — vertical edge')
    ax.set_ylabel('BEW$_H$ (pixels) — horizontal edge')
    add_subcaption(ax, '(d) Motion-blur metric — H/V decomposition isolates $\\sigma_{motion}$')
    eq_txt = r'$\sigma_{motion}=\sqrt{\mathrm{BEW}_{H}^{2}-\mathrm{BEW}_{V}^{2}}$'
    ax.text(0.04, 0.96, eq_txt, transform=ax.transAxes, va='top',
            fontsize=9, bbox={'facecolor': 'white', 'edgecolor': '#888', 'pad': 4})
    # anchor legend snug against the bottom-right corner so it stays well below the diagonal.
    ax.legend(loc='lower right', bbox_to_anchor=(0.995, 0.005), frameon=True, framealpha=0.95,
              fontsize=7.5, borderpad=0.4, handlelength=1.6, handletextpad=0.5)
    ax.set_aspect('equal', adjustable='box')


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))
    panel_a_histogram(axes[0, 0])
    panel_b_esf(axes[0, 1])
    panel_c_mtf(axes[1, 0])
    panel_d_motion_blur(axes[1, 1])
    fig.suptitle('Visual definitions of the five image-quality metrics (Sec 3.3)',
                 fontsize=12, y=0.995)
    fig.subplots_adjust(top=0.94, bottom=0.10, hspace=0.45, wspace=0.28)

    for ext in ('png', 'pdf'):
        out_src = ROOT / f'F12_iq_metrics_definitions.{ext}'
        out_paper = PAPER_FIG / f'F12_iq_metrics_definitions.{ext}'
        fig.savefig(out_src, dpi=200 if ext == 'png' else None, bbox_inches='tight')
        fig.savefig(out_paper, dpi=200 if ext == 'png' else None, bbox_inches='tight')
        print(f'  saved: {out_src}')
        print(f'  saved: {out_paper}')


if __name__ == '__main__':
    main()
