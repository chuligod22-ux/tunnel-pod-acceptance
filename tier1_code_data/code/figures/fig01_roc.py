"""
Figure 1: ROC curves
====================
- Univariate: C_M, L_90
- Multivariable: Logistic regression (C_M + L_90)
- Each with AUC + optimal operating point (Youden J)
- IEEE TIM style: 3.5" single-column, 600 DPI
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm

ROOT = Path("/Users/lch/home/code/tunnelscanning")
WIDE = ROOT / "tmp/b2_out/wide_v2.csv"
OUT_DIR = ROOT / "01_tunnelscanning/04_data/b2_results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Data ----------
df = pd.read_csv(WIDE)
y = df['identifiable_bin'].values

# ROC for univariate predictors
def get_roc(y, x):
    """Try both directions, pick larger AUC."""
    a1 = roc_auc_score(y, x)
    a2 = roc_auc_score(y, -x)
    if a2 > a1:
        fpr, tpr, thr = roc_curve(y, -x)
        return fpr, tpr, -thr, a2
    fpr, tpr, thr = roc_curve(y, x)
    return fpr, tpr, thr, a1

fpr_cm, tpr_cm, thr_cm, auc_cm = get_roc(y, df['C_M'].values)
fpr_l, tpr_l, thr_l, auc_l = get_roc(y, df['L_90'].values)

# Combined logistic
X = sm.add_constant(df[['C_M','L_90']].astype(float))
res = sm.Logit(y, X).fit(disp=False, maxiter=200)
prob = res.predict(X)
fpr_c, tpr_c, thr_c, auc_c = roc_curve(y, prob), None, None, roc_auc_score(y, prob)
fpr_c, tpr_c, thr_c = roc_curve(y, prob)

# Optimal points (Youden J = sens + spec - 1)
def youden(fpr, tpr, thr):
    j = tpr - fpr
    i = int(np.argmax(j))
    return fpr[i], tpr[i], thr[i]

cm_op = youden(fpr_cm, tpr_cm, thr_cm)
l_op = youden(fpr_l, tpr_l, thr_l)
c_op = youden(fpr_c, tpr_c, thr_c)

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

# Diagonal reference
ax.plot([0,1], [0,1], color='0.7', linestyle=':', linewidth=0.8, label='Chance (AUC = 0.50)')

# ROC curves
ax.plot(fpr_cm, tpr_cm, color='0.55', linestyle='--', linewidth=1.2,
        label=fr'$C_M$ alone (AUC = {auc_cm:.2f})')
ax.plot(fpr_l, tpr_l, color='0.25', linestyle='-.', linewidth=1.2,
        label=fr'$L_{{90}}$ alone (AUC = {auc_l:.2f})')
ax.plot(fpr_c, tpr_c, color='C3', linestyle='-', linewidth=1.6,
        label=fr'Logistic $C_M + L_{{90}}$ (AUC = {auc_c:.2f})')

# Optimal operating points
ax.plot(cm_op[0], cm_op[1], 'o', color='0.55', markersize=4, markeredgewidth=0.8)
ax.plot(l_op[0], l_op[1],  's', color='0.25', markersize=4, markeredgewidth=0.8)
ax.plot(c_op[0], c_op[1],  'D', color='C3', markersize=5, markeredgewidth=0.8)

# Annotation for combined optimal — placed in lower-right empty area with white background
ax.annotate(f'Logistic optimum\nSens = {c_op[1]:.2f}\nSpec = {1-c_op[0]:.2f}',
            xy=(c_op[0], c_op[1]), xytext=(0.55, 0.45),
            fontsize=7, color='C3', ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='C3', linewidth=0.6, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='C3', lw=0.6,
                            shrinkA=0, shrinkB=4))

# Style
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=8)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linewidth=0.4)

# Legend below the plot (horizontal, 2 columns) to avoid overlap
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
          ncol=2, fontsize=7, framealpha=0.95, edgecolor='0.7',
          columnspacing=0.8, handlelength=2.2, handletextpad=0.5)

# Tight layout (reserve space for legend below)
plt.tight_layout(pad=0.3)
plt.subplots_adjust(bottom=0.25)

# Save
out_pdf = OUT_DIR / "F1_roc_curves.pdf"
out_png = OUT_DIR / "F1_roc_curves.png"
plt.savefig(out_pdf, dpi=600, bbox_inches='tight')
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.close()

print("="*70)
print("Figure 1: ROC curves")
print("="*70)
print(f"\n  C_M alone:       AUC={auc_cm:.3f}, optimal thr={cm_op[2]:.3f}, J={cm_op[1]-cm_op[0]:.3f}")
print(f"  L_90 alone:      AUC={auc_l:.3f}, optimal thr={l_op[2]:.1f},   J={l_op[1]-l_op[0]:.3f}")
print(f"  Combined (logit): AUC={auc_c:.3f}, optimal p={c_op[2]:.3f}, J={c_op[1]-c_op[0]:.3f}")
print(f"\n  Saved: {out_pdf}")
print(f"         {out_png}")
