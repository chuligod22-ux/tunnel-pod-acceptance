"""
Figure 5: 2D C_M–L_90 detection map with logistic decision contours
====================================================================
- X: C_M (Michelson contrast)
- Y: L_90 (90th percentile luminance)
- Background: logistic regression predicted probability contour
- Gate boundary: C_M ≥ 0.05, L_90 ≥ 55 (dashed lines)
- Markers: identifiable (Y, green ●) vs non-identifiable (N, red ✕)
- 50 conditions (cam1) overlaid
- IEEE TIM single-column 4.0" × 3.5"
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

ROOT = Path("/Users/lch/home/code/tunnelscanning")
WIDE = ROOT / "tmp/b2_out/wide_v2.csv"
OUT_DIR = ROOT / "01_tunnelscanning/04_data/b2_results/figures"

# ---------- Data ----------
df = pd.read_csv(WIDE)
y = df['identifiable_bin'].values

# Fit logistic model
X_obs = sm.add_constant(df[['C_M', 'L_90']].astype(float))
res = sm.Logit(y, X_obs).fit(disp=False, maxiter=200)

# Grid for background contour
cm_grid = np.linspace(0.0, 0.42, 200)
l90_grid = np.linspace(40, 270, 200)
CM, L90 = np.meshgrid(cm_grid, l90_grid)
X_grid = np.column_stack([np.ones(CM.size), CM.ravel(), L90.ravel()])
PROB = res.predict(X_grid).reshape(CM.shape)

# Detection categories
df_y = df[df['identifiable_bin']==1]
df_n = df[df['identifiable_bin']==0]

# Detection prediction with dual-criterion gate
df['gate_pred'] = ((df['C_M']>=0.05) & (df['L_90']>=55)).astype(int)
df['cls'] = ''
df.loc[(df['gate_pred']==1) & (df['identifiable_bin']==1), 'cls'] = 'TP'
df.loc[(df['gate_pred']==0) & (df['identifiable_bin']==0), 'cls'] = 'TN'
df.loc[(df['gate_pred']==1) & (df['identifiable_bin']==0), 'cls'] = 'FP'
df.loc[(df['gate_pred']==0) & (df['identifiable_bin']==1), 'cls'] = 'FN'

# ---------- Figure ----------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

fig, ax = plt.subplots(figsize=(4.0, 3.6))

# Background: logistic predicted probability (subtle, smooth)
levels = np.linspace(0, 1, 21)
cf = ax.contourf(CM, L90, PROB, levels=levels, cmap='RdYlGn',
                 alpha=0.35, antialiased=True)
# Decision boundary p=0.5
cs = ax.contour(CM, L90, PROB, levels=[0.5], colors='black', linewidths=1.0,
                linestyles='-', alpha=0.6)
ax.clabel(cs, inline=True, fmt={0.5:'p=0.5'}, fontsize=7)

# Gate boundary (dashed)
ax.axvline(0.05, color='0.2', linestyle='--', linewidth=1.0, alpha=0.85)
ax.axhline(55, color='0.2', linestyle='--', linewidth=1.0, alpha=0.85)

# Shade gate-fail regions (very light)
ax.axvspan(0.0, 0.05, color='red', alpha=0.06, zorder=0)
ax.axhspan(40, 55, color='red', alpha=0.06, zorder=0)

# Apply small jitter to avoid marker overlap (60 km/h ≈ 80 km/h cases coincide)
RNG = np.random.default_rng(42)
df['C_M_j'] = df['C_M'] + RNG.uniform(-0.005, 0.005, len(df))
df['L_90_j'] = df['L_90'] + RNG.uniform(-3, 3, len(df))

# Markers per class
markers = {
    'TP': dict(marker='o', color='C2', label=f'TP (n={(df["cls"]=="TP").sum()})', s=30),
    'TN': dict(marker='s', color='C0', label=f'TN (n={(df["cls"]=="TN").sum()})', s=35),
    'FP': dict(marker='X', color='C3', label=f'FP (n={(df["cls"]=="FP").sum()})', s=55),
    'FN': dict(marker='P', color='C1', label=f'FN (n={(df["cls"]=="FN").sum()})', s=55),
}
for cls, props in markers.items():
    sub = df[df['cls']==cls]
    if len(sub)==0: continue
    ax.scatter(sub['C_M_j'], sub['L_90_j'],
               marker=props['marker'], facecolor=props['color'],
               edgecolor='white', s=props['s'], linewidth=0.8, alpha=0.9,
               label=props['label'], zorder=5)

# Annotate the 2 FP cases — combined into a single label since both nearly coincide
fps = df[df['cls']=='FP'].sort_values('speed').reset_index(drop=True)
xy_anchor = (fps['C_M_j'].mean(), fps['L_90_j'].mean())
ax.annotate('FP: 60 & 80 km/h\nISO 200, d = 6.5 m',
            xy=xy_anchor,
            xytext=(0.18, 100), textcoords='data',
            fontsize=6.5, color='C3', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.20', facecolor='white',
                      edgecolor='C3', linewidth=0.6, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='C3', lw=0.5,
                            shrinkA=0, shrinkB=4))

# Gate boundary labels — inline with white bbox (v6 style)
ax.text(0.05, 150, '$C_M = 0.05$', fontsize=7, color='0.2',
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  edgecolor='0.4', linewidth=0.5, alpha=0.95))
ax.text(0.20, 55, '$L_{90} = 55$', fontsize=7, color='0.2',
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  edgecolor='0.4', linewidth=0.5, alpha=0.95))

# Style
ax.set_xlim(0, 0.42)
ax.set_ylim(40, 270)
ax.set_xlabel('Michelson contrast $C_M$', fontsize=8)
ax.set_ylabel('Luminance 90th percentile $L_{90}$', fontsize=8)
ax.grid(True, alpha=0.3, linewidth=0.4)

# Colorbar (small, on right)
cbar = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.04)
cbar.set_label('Logistic $P($identifiable$)$', fontsize=7)
cbar.ax.tick_params(labelsize=6)

# Legend below plot, horizontal
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
          ncol=4, fontsize=7, framealpha=0.95, edgecolor='0.7',
          columnspacing=0.8, handletextpad=0.4)

plt.tight_layout(pad=0.3)
plt.subplots_adjust(bottom=0.22)

# Save
out_pdf = OUT_DIR / "F5_cm_l90_heatmap.pdf"
out_png = OUT_DIR / "F5_cm_l90_heatmap.png"
plt.savefig(out_pdf, dpi=600, bbox_inches='tight')
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.close()

print("="*70)
print("Figure 5: C_M-L_90 detection map with logistic contours")
print("="*70)
print(f"\n  Class counts: TP={(df['cls']=='TP').sum()}, TN={(df['cls']=='TN').sum()}, "
      f"FP={(df['cls']=='FP').sum()}, FN={(df['cls']=='FN').sum()}")
print(f"\n  Saved: {out_pdf}")
print(f"         {out_png}")
