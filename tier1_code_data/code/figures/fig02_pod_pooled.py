"""
Figure 2: POD pooled curve with 95% CI band (data-range restricted)
====================================================================
- POD curve (logistic, monotonic detection assumption)
- Wald 95% CI band (parametric, Berens-style)
- Empirical detection rate per width (markers)
- Reference lines: POD=0.5 (a50)
- Plateau line ~88% (ceiling from 6 fail conditions)
- a50 only (a90 unreachable due to plateau)
- IEEE TIM single-column
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/Users/lch/home/code/tunnelscanning")
DATA = ROOT / "01_tunnelscanning/04_data/b2_results"
LONG = ROOT / "tmp/b2_out/long_v2.csv"
OUT_DIR = DATA / "figures"

# ---------- Data ----------
curve = pd.read_csv(DATA / "pod_curve_pooled.csv")
df_l = pd.read_csv(LONG)
with open(DATA / "pod_ci_band.json") as f:
    ci = json.load(f)
ci_p = ci['pooled']
a50 = ci_p['a50_point_mm']

# Restrict to data range (0.1 to 1.0 mm chart widths)
W_MIN, W_MAX = 0.05, 1.05  # plot range (slightly outside data for visual breathing)
curve_in = curve[(curve['width_mm'] >= W_MIN) & (curve['width_mm'] <= W_MAX)].copy()

# Empirical detection rate per width
emp = df_l.groupby('width_mm')['detected'].agg(['sum','count','mean']).reset_index()
emp['se'] = np.sqrt(emp['mean'] * (1-emp['mean']) / emp['count'])

# POD plateau (ceiling): largest empirical detection rate (≈ proportion of gate-passing conditions)
plateau = emp['mean'].max()
n_fail = 50 - emp['mean'].max() * 50  # rough estimate

# ---------- Figure ----------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

fig, ax = plt.subplots(figsize=(3.5, 3.5))

# Wald 95% CI band (within data range only)
ax.fill_between(curve_in['width_mm'], curve_in['pod_wald_lo'], curve_in['pod_wald_hi'],
                color='C3', alpha=0.18, linewidth=0, label='95% CI (Wald)')

# Point estimate (within data range only)
ax.plot(curve_in['width_mm'], curve_in['pod_point'], color='C3', linestyle='-', linewidth=1.6,
        label='Logistic POD fit')

# Empirical detection rate
ax.errorbar(emp['width_mm'], emp['mean'],
            yerr=1.96*emp['se'], fmt='o', color='0.25',
            markersize=4, markeredgewidth=0.7, capsize=2, capthick=0.7,
            elinewidth=0.7, label='Empirical (per-width)')

# POD plateau (ceiling) — horizontal dashed line at ~0.88
ax.axhline(plateau, color='C0', linestyle='--', linewidth=0.9, alpha=0.85)
# Plateau annotation inside plot, upper-left empty area, 2-line text + arrow to line
ax.annotate(f'Plateau $\\approx$ {plateau*100:.0f}%\n(6/50 exposure-fail)',
            xy=(0.20, plateau), xycoords='data',
            xytext=(0.06, 0.96), textcoords='data',
            fontsize=7, color='C0', ha='left', va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='C0', linewidth=0.6, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='C0', lw=0.6, shrinkA=0, shrinkB=2))

# Reference horizontal line POD=0.5 (label outside plot box on the right)
ax.axhline(0.5, color='0.6', linestyle=':', linewidth=0.7)
ax.text(1.015, 0.5, 'POD = 0.50', va='center', ha='left', fontsize=7, color='0.4',
        transform=ax.get_yaxis_transform(), clip_on=False)

# Vertical reference at a50 (within data range)
ax.axvline(a50, color='0.5', linestyle='--', linewidth=0.6)
ax.annotate(f'$a_{{50}}$ = {a50:.2f} mm',
            xy=(a50, 0.5), xytext=(a50, 0.10),
            fontsize=7, ha='center', color='0.3',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='0.5', linewidth=0.5, alpha=0.95))

# Data range shaded background
ax.axvspan(0.1, 1.0, color='gray', alpha=0.05, zorder=0)
ax.text(0.1, 0.02, '$\\leftarrow$ Tested chart range (0.1–1.0 mm) $\\rightarrow$',
        fontsize=6.5, color='0.4', ha='left', va='bottom')

# Style
ax.set_xscale('log')
ax.set_xlim(W_MIN, W_MAX*1.05)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel('Crack width (mm)', fontsize=8)
ax.set_ylabel('Probability of detection (POD)', fontsize=8)
ax.grid(True, which='both', alpha=0.3, linewidth=0.4)

xticks = [0.05, 0.1, 0.2, 0.5, 1.0]
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks])

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
          ncol=3, fontsize=7, framealpha=0.95, edgecolor='0.7',
          columnspacing=0.8, handlelength=1.8, handletextpad=0.5)

plt.tight_layout(pad=0.3)
plt.subplots_adjust(bottom=0.22, right=0.85)

# Save
out_pdf = OUT_DIR / "F2_pod_pooled.pdf"
out_png = OUT_DIR / "F2_pod_pooled.png"
plt.savefig(out_pdf, dpi=600, bbox_inches='tight')
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.close()

print("="*70)
print("Figure 2: POD pooled curve (data-range restricted)")
print("="*70)
print(f"\n  a50 (point) = {a50:.3f} mm  (within data range, valid)")
print(f"  POD plateau ≈ {plateau*100:.0f}% (6 of 50 conditions exposure-fail)")
print(f"  → a90 not reachable in pooled analysis; use gate-stratified (F3)")
print(f"\n  Saved: {out_pdf}")
print(f"         {out_png}")
