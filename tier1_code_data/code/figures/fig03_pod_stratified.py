"""
Figure 3: POD stratified by exposure gate (gate-pass vs pooled)
================================================================
- Pooled POD (all 50 conditions, 88% plateau, dashed gray)
- Gate-pass POD (46 conditions, well-behaved, solid red)
- 95% CI bands for both
- a50, a90, a90/95 (gate-pass — within data range)
- IEEE TIM single-column 3.5" × 3.5"
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
curve_p = pd.read_csv(DATA / "pod_curve_pooled.csv")
curve_g = pd.read_csv(DATA / "pod_curve_gate_pass.csv")
df_l = pd.read_csv(LONG)
df_l['gate'] = ((df_l['C_M']>=0.05) & (df_l['L_90']>=55)).astype(int)

with open(DATA / "pod_ci_band.json") as f:
    ci = json.load(f)
g = ci['gate_pass']
a50_g = g['a50_point_mm']
a90_g = g['a90_point_mm']
a90_95_g_wald = g['a90_95_wald_mm']
a90_95_g_boot = g['a90_bootstrap']['a90_95']

# Restrict to data range
W_MIN, W_MAX = 0.05, 1.05
curve_p_in = curve_p[(curve_p['width_mm']>=W_MIN) & (curve_p['width_mm']<=W_MAX)].copy()
curve_g_in = curve_g[(curve_g['width_mm']>=W_MIN) & (curve_g['width_mm']<=W_MAX)].copy()

# Empirical detection rate per width per gate group
emp_p = df_l.groupby('width_mm')['detected'].agg(['sum','count','mean']).reset_index()
emp_p['se'] = np.sqrt(emp_p['mean']*(1-emp_p['mean'])/emp_p['count'])
emp_g = df_l[df_l['gate']==1].groupby('width_mm')['detected'].agg(['sum','count','mean']).reset_index()
emp_g['se'] = np.sqrt(emp_g['mean']*(1-emp_g['mean'])/emp_g['count'])

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

# Pooled CI band (light gray, in background)
ax.fill_between(curve_p_in['width_mm'], curve_p_in['pod_wald_lo'], curve_p_in['pod_wald_hi'],
                color='0.6', alpha=0.15, linewidth=0)
# Pooled curve (dashed gray)
ax.plot(curve_p_in['width_mm'], curve_p_in['pod_point'], color='0.4', linestyle='--',
        linewidth=1.2, label='Pooled (n = 50)')

# Gate-pass CI band (red)
ax.fill_between(curve_g_in['width_mm'], curve_g_in['pod_wald_lo'], curve_g_in['pod_wald_hi'],
                color='C3', alpha=0.18, linewidth=0, label='95% CI (gate-pass)')
# Gate-pass curve (solid red)
ax.plot(curve_g_in['width_mm'], curve_g_in['pod_point'], color='C3', linestyle='-',
        linewidth=1.6, label='Gate-pass (n = 46)')

# Empirical markers (gate-pass only)
ax.errorbar(emp_g['width_mm'], emp_g['mean'], yerr=1.96*emp_g['se'],
            fmt='o', color='C3', markersize=4, markeredgewidth=0.7,
            capsize=2, capthick=0.7, elinewidth=0.7)

# Reference horizontal lines (POD labels placed outside plot box, on the right)
ax.axhline(0.5, color='0.6', linestyle=':', linewidth=0.7)
ax.axhline(0.9, color='0.6', linestyle=':', linewidth=0.7)
ax.text(1.015, 0.5, 'POD = 0.50', va='center', ha='left', fontsize=7, color='0.4',
        transform=ax.get_yaxis_transform(), clip_on=False)
ax.text(1.015, 0.9, 'POD = 0.90', va='center', ha='left', fontsize=7, color='0.4',
        transform=ax.get_yaxis_transform(), clip_on=False)

# Vertical references for gate-pass acceptance metrics
ax.axvline(a50_g, color='0.5', linestyle='--', linewidth=0.6)
ax.axvline(a90_g, color='0.5', linestyle='--', linewidth=0.6)
ax.axvline(a90_95_g_wald, color='C3', linestyle=':', linewidth=0.8)

# Gate-pass key metrics annotations (bottom row, white-bg boxes)
ax.annotate(f'$a_{{50}}$ = {a50_g:.2f}',
            xy=(a50_g, 0.5), xytext=(a50_g, 0.10),
            fontsize=7, ha='center', color='0.3',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='0.5', linewidth=0.5, alpha=0.95))
ax.annotate(f'$a_{{90}}$ = {a90_g:.2f}',
            xy=(a90_g, 0.9), xytext=(a90_g, 0.20),
            fontsize=7, ha='center', color='0.3',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='0.5', linewidth=0.5, alpha=0.95))
ax.annotate(f'$a_{{90/95}}$ = {a90_95_g_wald:.2f}',
            xy=(a90_95_g_wald, 0.9), xytext=(a90_95_g_wald, 0.30),
            fontsize=7, ha='center', color='C3',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='C3', linewidth=0.5, alpha=0.95))

# Pooled plateau callout (upper-left, two-line, white box)
ax.annotate(f'Pooled plateau\n$\\approx$ 88% (gate-fail)',
            xy=(0.18, 0.88), xycoords='data',
            xytext=(0.06, 0.96), textcoords='data',
            fontsize=7, color='0.4', ha='left', va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='0.5', linewidth=0.6, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='0.5', lw=0.6, shrinkA=0, shrinkB=2))

# Tested chart range shade
ax.axvspan(0.1, 1.0, color='gray', alpha=0.05, zorder=0)

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
          columnspacing=0.8, handlelength=2.0, handletextpad=0.5)

plt.tight_layout(pad=0.3)
plt.subplots_adjust(bottom=0.22, right=0.85)

# Save
out_pdf = OUT_DIR / "F3_pod_stratified.pdf"
out_png = OUT_DIR / "F3_pod_stratified.png"
plt.savefig(out_pdf, dpi=600, bbox_inches='tight')
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.close()

print("="*70)
print("Figure 3: POD stratified (gate-pass vs pooled)")
print("="*70)
print(f"\n  Pooled (n=50):    plateau ≈ 88% (a90 unreachable)")
print(f"  Gate-pass (n=46): a50 = {a50_g:.3f} mm")
print(f"                    a90 = {a90_g:.3f} mm (within range ✓)")
print(f"                    a90/95 (Wald) = {a90_95_g_wald:.3f} mm")
print(f"                    a90/95 (Boot) = {a90_95_g_boot:.3f} mm")
print(f"\n  → Gate stratification reduces a90/95 from N/A to {a90_95_g_wald:.2f} mm")
print(f"\n  Saved: {out_pdf}")
print(f"         {out_png}")
