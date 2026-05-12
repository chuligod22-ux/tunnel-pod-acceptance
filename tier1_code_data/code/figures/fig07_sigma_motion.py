"""
Figure 7: η distribution per group — 2×2 grid of strip plots
================================================================
- 4 panels (cam1×60, cam1×80, cam2×60, cam2×80) in 2×2 grid
- Each panel: vertical strip plot (jittered points) + group mean line
- Common y-axis (η = 0 ~ 1.2)
- Common reference: kinematic prediction (η=1) on every panel
- IEEE TIM single-column 4.5" × 4.0"
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/Users/lch/home/code/tunnelscanning")
DATA = ROOT / "01_tunnelscanning/04_data/b2_results"
OUT_DIR = DATA / "figures"

# ---------- Data ----------
df = pd.read_csv(DATA / "sigma_motion_data.csv")
with open(DATA / "sigma_motion_model.json") as f:
    mdl = json.load(f)

eta_overall = mdl['overall']['eta_mean']

group_order = [('cam1', 60), ('cam1', 80), ('cam2', 60), ('cam2', 80)]
spd_colors = {60: 'C0', 80: 'C1'}

# ---------- Figure ----------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

fig, axes = plt.subplots(2, 2, figsize=(4.8, 4.8),
                          sharey=True, sharex=True)
RNG = np.random.default_rng(42)

panel_letters = ['(a)', '(b)', '(c)', '(d)']

for ax, letter, (cam, spd) in zip(axes.flat, panel_letters, group_order):
    sub = df[(df['cam']==cam) & (df['speed']==spd)]
    eta_vals = sub['eta'].values
    n = len(eta_vals)
    grp_mean = float(np.mean(eta_vals))
    grp_std = float(np.std(eta_vals, ddof=1)) if n>1 else float('nan')

    # Reference: kinematic prediction (η=1)
    ax.axhline(1.0, color='0.5', linestyle=':', linewidth=0.9)

    # Strip plot: jittered points
    jitter = RNG.uniform(-0.25, 0.25, n)
    ax.scatter(jitter, eta_vals, color=spd_colors[spd], s=24, alpha=0.75,
               edgecolor='white', linewidth=0.5, zorder=5)

    # Group mean line + ±1 SD shaded band
    ax.axhline(grp_mean, color='C3', linewidth=1.4)
    if not np.isnan(grp_std):
        ax.axhspan(grp_mean - grp_std, grp_mean + grp_std,
                   color='C3', alpha=0.10, zorder=0)

    # Per-panel style
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.0, 1.20)
    ax.set_xticks([])
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)

    # Sub-caption below each panel (centered)
    ax.set_xlabel(f'{letter} {cam}, {spd} km/h  (n = {n})\n'
                  f'η = {grp_mean:.3f} ± {grp_std:.3f}',
                  fontsize=8, fontweight='bold', labelpad=6)

# Shared y-axis label (only on left panels)
for ax in axes[:, 0]:
    ax.set_ylabel(r'η = $\sigma_{motion}/\sigma_{theory}$', fontsize=8)

# Suptitle / overall stats
fig.suptitle(f'Overall: η = {eta_overall:.3f}, n = {len(df)} '
             f'(system-invariant by AIC, ΔAIC < 2 across H1/H2/H3 hypotheses)',
             fontsize=8, y=0.99)

plt.tight_layout(pad=0.3)
plt.subplots_adjust(top=0.88)

# Save
out_pdf = OUT_DIR / "F7_sigma_motion.pdf"
out_png = OUT_DIR / "F7_sigma_motion.png"
plt.savefig(out_pdf, dpi=600, bbox_inches='tight')
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.close()

print("="*70)
print("Figure 7: η distribution by group (2x2 grid)")
print("="*70)
print(f"  Overall η = {eta_overall:.3f}  n = {len(df)}")
for cam, spd in group_order:
    sub = df[(df['cam']==cam) & (df['speed']==spd)]
    print(f"  {cam}, {spd} km/h: n={len(sub)}, mean={sub['eta'].mean():.3f}, std={sub['eta'].std():.3f}")
print(f"\n  Saved: {out_pdf}")
print(f"         {out_png}")
