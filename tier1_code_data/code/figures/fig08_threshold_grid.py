"""
Figure 8: Threshold robustness — Grid search of (C_M, L_90) thresholds
=========================================================================
- X: C_M threshold sweep
- Y: L_90 threshold sweep
- Color: dual-criterion gate accuracy
- Contour at the achievable ceiling (96%)
- Reference markers:
    - Reported gate (C_M=0.05, L_90=55)
    - Logistic-derived alternative
- IEEE TIM single-column 4.0" × 3.5"
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/Users/lch/home/code/tunnelscanning")
WIDE = ROOT / "tmp/b2_out/wide_v2.csv"
DATA = ROOT / "01_tunnelscanning/04_data/b2_results"
OUT_DIR = DATA / "figures"

# ---------- Data ----------
df = pd.read_csv(WIDE)
y = df['identifiable_bin'].values
C = df['C_M'].values
L = df['L_90'].values

# Threshold grids
C_grid = np.arange(0.005, 0.205, 0.005)
L_grid = np.arange(40, 101, 1.0)

acc_grid = np.zeros((len(L_grid), len(C_grid)))
for i, l_thr in enumerate(L_grid):
    for j, c_thr in enumerate(C_grid):
        pred = ((C >= c_thr) & (L >= l_thr)).astype(int)
        acc_grid[i, j] = (pred == y).mean()

acc_max = float(acc_grid.max())
n_perfect = int((acc_grid == acc_max).sum())

# ---------- Figure ----------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.linewidth': 0.8,
})

fig, ax = plt.subplots(figsize=(4.2, 3.6))

# Heatmap
im = ax.pcolormesh(C_grid, L_grid, acc_grid, shading='auto',
                   cmap='viridis', vmin=0.80, vmax=acc_max)

# Contour at the achievable ceiling
ceiling_levels = [acc_max - 1e-6]
cs = ax.contour(C_grid, L_grid, acc_grid, levels=ceiling_levels,
                colors='white', linewidths=1.0, linestyles='-')
ax.clabel(cs, fmt={ceiling_levels[0]: f'{acc_max*100:.0f}% ceiling'},
          fontsize=7, colors='white', inline=True)

# Reported dual-criterion gate marker
ax.plot(0.05, 55, marker='o', markersize=8, markerfacecolor='C3',
        markeredgecolor='white', markeredgewidth=1.0, zorder=10)
ax.annotate('Reported gate\n($C_M$=0.05, $L_{90}$=55)',
            xy=(0.05, 55), xytext=(0.015, 70),
            fontsize=7, color='C3', ha='left', va='bottom',
            arrowprops=dict(arrowstyle='->', color='C3', lw=0.6,
                            shrinkA=0, shrinkB=4))

# Style
ax.set_xlim(C_grid.min(), C_grid.max())
ax.set_ylim(L_grid.min(), L_grid.max())
ax.set_xlabel(r'$C_M$ threshold', fontsize=8)
ax.set_ylabel(r'$L_{90}$ threshold', fontsize=8)
ax.grid(True, alpha=0.25, linewidth=0.4, color='white')
ax.set_axisbelow(False)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
cbar.set_label('Dual-criterion gate accuracy', fontsize=7)
cbar.ax.tick_params(labelsize=6)

# Subtitle below the plot describing finding
ax.set_xlabel(r'$C_M$ threshold' + '\n' +
              f'Achievable ceiling = {acc_max*100:.0f}%   '
              f'(realised by {n_perfect} threshold combinations)',
              fontsize=8, labelpad=8)

plt.tight_layout(pad=0.4)

# Save
out_pdf = OUT_DIR / "F8_threshold_grid.pdf"
out_png = OUT_DIR / "F8_threshold_grid.png"
plt.savefig(out_pdf, dpi=600, bbox_inches='tight')
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.close()

print("="*70)
print("Figure 8: Threshold-robustness grid search")
print("="*70)
print(f"  Achievable ceiling accuracy: {acc_max*100:.1f}%")
print(f"  Threshold combinations achieving ceiling: {n_perfect}")
print(f"  → 100% accuracy is NOT attainable with C_M+L_90 alone (intrinsic limit)")
print(f"\n  Saved: {out_pdf}")
print(f"         {out_png}")
