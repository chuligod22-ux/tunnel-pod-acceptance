"""
Figure 4: Calibration plot (2 panels)
======================================
- Panel A: Logistic regression (identifiable ~ C_M+L_90, n=50, 5 bins)
- Panel B: POD model (detected ~ log_width, gate-pass n=460, 10 bins)
- Predicted vs observed binned proportions
- Diagonal reference (perfect calibration)
- Brier score, MCE, HL test p-value annotations
- IEEE TIM double-column 7" × 3.5"
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
import statsmodels.api as sm
from scipy import stats

ROOT = Path("/Users/lch/home/code/tunnelscanning")
DATA = ROOT / "01_tunnelscanning/04_data/b2_results"
WIDE = ROOT / "tmp/b2_out/wide_v2.csv"
LONG = ROOT / "tmp/b2_out/long_v2.csv"
OUT_DIR = DATA / "figures"

# ---------- Helpers ----------
def hosmer_lemeshow(y, p, n_bins=10):
    df = pd.DataFrame({'y': y, 'p': p})
    df['bin'] = pd.qcut(df['p'], q=n_bins, duplicates='drop')
    g = df.groupby('bin', observed=True).agg(
        n=('y','count'), o_pos=('y','sum'), e_pos=('p','sum')).reset_index(drop=True)
    g['o_neg'] = g['n'] - g['o_pos']
    g['e_neg'] = g['n'] - g['e_pos']
    chi2 = ((g['o_pos']-g['e_pos'])**2/g['e_pos'].clip(lower=1e-9) +
            (g['o_neg']-g['e_neg'])**2/g['e_neg'].clip(lower=1e-9)).sum()
    df_chi = max(len(g) - 2, 1)
    return chi2, df_chi, 1 - stats.chi2.cdf(chi2, df_chi)

def calibration_bins(y, p, n_bins=10):
    df = pd.DataFrame({'y': y, 'p': p})
    df['bin'] = pd.qcut(df['p'], q=n_bins, duplicates='drop')
    g = df.groupby('bin', observed=True).agg(
        n=('y','count'), mean_pred=('p','mean'), obs_freq=('y','mean')).reset_index(drop=True)
    g['se'] = np.sqrt(g['obs_freq']*(1-g['obs_freq'])/g['n'])
    return g

# ---------- Panel A data: Logistic ----------
df_w = pd.read_csv(WIDE)
y_w = df_w['identifiable_bin'].values
X_w = sm.add_constant(df_w[['C_M','L_90']].astype(float))
res_w = sm.Logit(y_w, X_w).fit(disp=False, maxiter=200)
prob_w = res_w.predict(X_w)

bins_log = calibration_bins(y_w, prob_w, n_bins=5)
brier_log = brier_score_loss(y_w, prob_w)
mce_log = np.mean(np.abs(bins_log['mean_pred'] - bins_log['obs_freq']))
chi2_log, df_log, p_log = hosmer_lemeshow(y_w, prob_w, n_bins=5)

# ---------- Panel B data: POD (gate-pass) ----------
df_l = pd.read_csv(LONG)
df_l['gate'] = ((df_l['C_M']>=0.05) & (df_l['L_90']>=55)).astype(int)
df_pass = df_l[df_l['gate']==1]
y_p = df_pass['detected'].values
X_p = sm.add_constant(np.log(df_pass['width_mm'].values))
res_p = sm.Logit(y_p, X_p).fit(disp=False, maxiter=200)
prob_p = res_p.predict(X_p)

bins_pod = calibration_bins(y_p, prob_p, n_bins=10)
brier_pod = brier_score_loss(y_p, prob_p)
mce_pod = np.mean(np.abs(bins_pod['mean_pred'] - bins_pod['obs_freq']))
chi2_pod, df_pod, p_pod = hosmer_lemeshow(y_p, prob_p, n_bins=10)

# ---------- Figure ----------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

for ax, bins, brier, mce, chi2, df_chi, p_val, label, panel, n_total in [
    (axes[0], bins_log, brier_log, mce_log, chi2_log, df_log, p_log,
     'Logistic gate model', '(a)', len(df_w)),
    (axes[1], bins_pod, brier_pod, mce_pod, chi2_pod, df_pod, p_pod,
     'POD model (gate-pass)', '(b)', len(df_pass)),
]:
    # Diagonal reference (perfect calibration)
    ax.plot([0,1], [0,1], color='0.5', linestyle=':', linewidth=0.8, label='Perfect calibration')

    # Calibration markers + error bars (binomial CI)
    # alpha for visibility when markers overlap (logistic model in panel A)
    ax.errorbar(bins['mean_pred'], bins['obs_freq'],
                yerr=1.96*bins['se'], fmt='o', color='C3',
                markersize=5, markeredgewidth=0.8, markerfacecolor='C3',
                markeredgecolor='white', alpha=0.85,
                capsize=2, capthick=0.7, elinewidth=0.7,
                zorder=5, label='Binned (predicted vs observed)')

    # Single sample-size annotation (all bins equal-sized due to qcut)
    n_per_bin = int(bins['n'].iloc[0])
    ax.text(0.03, 0.97, f'{n_per_bin} samples per bin',
            transform=ax.transAxes, fontsize=7, color='0.35',
            ha='left', va='top', style='italic')

    # Style
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_xlabel('Mean predicted probability', fontsize=8)
    ax.set_ylabel('Observed frequency', fontsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.4)

    # Panel caption placed below the x-axis label
    ax.text(0.5, -0.28, f'{panel} {label}  (n = {n_total})',
            transform=ax.transAxes, fontsize=8, ha='center', va='top',
            fontweight='bold')

    # Stats annotation in lower-right area
    p_str = f'{p_val:.3f}' if p_val>=0.001 else f'{p_val:.1e}'
    stats_txt = (f'Brier = {brier:.3f}\n'
                 f'MCE  = {mce:.3f}\n'
                 f'HL: $\\chi^2$={chi2:.2f}, df={df_chi}, p={p_str}')
    ax.text(0.97, 0.05, stats_txt, transform=ax.transAxes,
            fontsize=7, ha='right', va='bottom', color='0.2',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='0.6', linewidth=0.6, alpha=0.95))

# Single legend for both panels (top, horizontal — original layout)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
           ncol=2, fontsize=7, framealpha=0.95, edgecolor='0.7',
           columnspacing=1.5, handletextpad=0.5)

plt.tight_layout(pad=0.3)
plt.subplots_adjust(top=0.88, bottom=0.20)

# Save
out_pdf = OUT_DIR / "F4_calibration.pdf"
out_png = OUT_DIR / "F4_calibration.png"
plt.savefig(out_pdf, dpi=600, bbox_inches='tight')
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.close()

print("="*70)
print("Figure 4: Calibration plot (2 panels)")
print("="*70)
print(f"\n  Panel A: Logistic (n={len(df_w)})")
print(f"    Brier = {brier_log:.4f}  MCE = {mce_log:.4f}  HL p = {p_log:.4f}")
print(f"\n  Panel B: POD gate-pass (n={len(df_pass)})")
print(f"    Brier = {brier_pod:.4f}  MCE = {mce_pod:.4f}  HL p = {p_pod:.4f}")
print(f"\n  Saved: {out_pdf}")
print(f"         {out_png}")
