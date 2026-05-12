"""
Figure 6: Confusion matrix + performance metrics
=================================================
- 2×2 confusion matrix for dual-criterion gate
- Cell counts + percentages
- Performance metrics summary (Accuracy, Sens, Spec, NPV, PPV)
- Bootstrap 95% CI for each metric
- IEEE TIM single-column 4.5" × 3.5"
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy import stats

ROOT = Path("/Users/lch/home/code/tunnelscanning")
WIDE = ROOT / "tmp/b2_out/wide_v2.csv"
DATA = ROOT / "01_tunnelscanning/04_data/b2_results"
OUT_DIR = DATA / "figures"

# ---------- Data ----------
df = pd.read_csv(WIDE)
y = df['identifiable_bin'].values
pred = ((df['C_M']>=0.05) & (df['L_90']>=55)).astype(int).values

# Confusion matrix (rows = actual, cols = predicted)
# format: rows [Actual N=0, Actual Y=1], cols [Pred N=0, Pred Y=1]
cm = confusion_matrix(y, pred, labels=[0,1])
tn, fp = cm[0,0], cm[0,1]
fn, tp = cm[1,0], cm[1,1]
n = len(y)

# Metrics
acc = (tp+tn)/n
sens = tp/(tp+fn) if (tp+fn)>0 else float('nan')
spec = tn/(tn+fp) if (tn+fp)>0 else float('nan')
ppv = tp/(tp+fp) if (tp+fp)>0 else float('nan')   # precision
npv = tn/(tn+fn) if (tn+fn)>0 else float('nan')

# Bootstrap CI
N_BOOT = 2000
RNG = np.random.default_rng(42)
def boot_ci(metric_fn):
    samples = []
    for _ in range(N_BOOT):
        idx = RNG.choice(n, size=n, replace=True)
        if len(np.unique(y[idx])) < 2: continue
        m = metric_fn(y[idx], pred[idx])
        if not np.isnan(m): samples.append(m)
    return (float(np.median(samples)),
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)))

acc_med, acc_lo, acc_hi = boot_ci(lambda y_,p_: (y_==p_).mean())
sens_med, sens_lo, sens_hi = boot_ci(lambda y_,p_: ((y_==1)&(p_==1)).sum()/max((y_==1).sum(),1))
spec_med, spec_lo, spec_hi = boot_ci(lambda y_,p_: ((y_==0)&(p_==0)).sum()/max((y_==0).sum(),1))

# ---------- Figure ----------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.linewidth': 0.8,
})

fig, (ax_cm, ax_metrics) = plt.subplots(1, 2, figsize=(7.0, 3.6),
                                          gridspec_kw={'width_ratios': [1.0, 1.0]})

# ===== Panel A: Confusion matrix heatmap =====
# Display matrix
display = np.array([[tn, fp],
                    [fn, tp]])
labels = np.array([[f'TN\n{tn}\n({tn/n*100:.0f}%)', f'FP\n{fp}\n({fp/n*100:.0f}%)'],
                   [f'FN\n{fn}\n({fn/n*100:.0f}%)', f'TP\n{tp}\n({tp/n*100:.0f}%)']])

# Custom colors: TP/TN green, FP/FN red, intensity by count
color_map = np.array([[(0.85, 0.95, 0.85), (1.0, 0.85, 0.85)],
                      [(1.0, 0.85, 0.85), (0.85, 0.95, 0.85)]])

ax_cm.imshow(np.ones((2,2,3)), extent=(-0.5, 1.5, 1.5, -0.5))
for i in range(2):
    for j in range(2):
        ax_cm.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       facecolor=color_map[i,j],
                                       edgecolor='0.5', linewidth=0.8))
        ax_cm.text(j, i, labels[i,j], ha='center', va='center',
                   fontsize=10, fontweight='bold', color='0.15')

ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(['Predicted N\n(gate-fail)', 'Predicted Y\n(gate-pass)'], fontsize=8)
ax_cm.set_yticklabels(['Actual N\n(non-detectable)', 'Actual Y\n(detectable)'], fontsize=8)
ax_cm.set_xlim(-0.5, 1.5)
ax_cm.set_ylim(1.5, -0.5)
ax_cm.set_aspect('equal')

# Remove ticks
ax_cm.tick_params(axis='both', which='both', length=0)

# ===== Panel B: Performance metrics bar chart with CI =====
metrics = [
    ('Accuracy',    acc,  acc_lo,  acc_hi),
    ('Sensitivity', sens, sens_lo, sens_hi),
    ('Specificity', spec, spec_lo, spec_hi),
    ('PPV',         ppv,  None,    None),
    ('NPV',         npv,  None,    None),
]

names = [m[0] for m in metrics]
values = [m[1] for m in metrics]
err_lo = [(v - lo) if lo is not None else 0 for _, v, lo, _ in metrics]
err_hi = [(hi - v) if hi is not None else 0 for _, v, _, hi in metrics]

y_pos = np.arange(len(names))[::-1]
colors = ['C0','C2','C3','C4','C5']
bars = ax_metrics.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='0.3', linewidth=0.6,
                       xerr=[err_lo, err_hi], capsize=3, error_kw=dict(elinewidth=0.7))

# Annotate values — point estimate only (CI shown via error bar to avoid redundancy)
for yi, (name, val, lo, hi) in zip(y_pos, metrics):
    txt = f'{val:.3f}'
    x_text = (max(hi, val) if hi is not None else val) + 0.03
    ax_metrics.text(x_text, yi, txt, va='center', ha='left', fontsize=8,
                    color='0.2')

ax_metrics.set_yticks(y_pos)
ax_metrics.set_yticklabels(names, fontsize=8)
ax_metrics.set_xlim(0, 1.20)
ax_metrics.set_xlabel('Metric value (error bars: bootstrap 95% CI)', fontsize=8)
ax_metrics.axvline(1.0, color='0.6', linestyle=':', linewidth=0.7)
ax_metrics.grid(True, axis='x', alpha=0.3, linewidth=0.4)
ax_metrics.set_axisbelow(True)

# Unified panel captions placed at consistent figure-bottom position
fig.text(0.27, 0.02, '(a) Dual-criterion gate confusion matrix (n = 50)',
         ha='center', va='bottom', fontsize=8, fontweight='bold')
fig.text(0.74, 0.02, '(b) Performance metrics with bootstrap 95% CI',
         ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout(pad=0.5)
plt.subplots_adjust(bottom=0.18)

# Save
out_pdf = OUT_DIR / "F6_confusion_matrix.pdf"
out_png = OUT_DIR / "F6_confusion_matrix.png"
plt.savefig(out_pdf, dpi=600, bbox_inches='tight')
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.close()

print("="*70)
print("Figure 6: Confusion matrix + performance metrics")
print("="*70)
print(f"\n  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"  Accuracy    = {acc:.3f}  95% CI [{acc_lo:.3f}, {acc_hi:.3f}]")
print(f"  Sensitivity = {sens:.3f}  95% CI [{sens_lo:.3f}, {sens_hi:.3f}]")
print(f"  Specificity = {spec:.3f}  95% CI [{spec_lo:.3f}, {spec_hi:.3f}]")
print(f"  PPV (precision) = {ppv:.3f}")
print(f"  NPV             = {npv:.3f}")
print(f"\n  Saved: {out_pdf}")
print(f"         {out_png}")
