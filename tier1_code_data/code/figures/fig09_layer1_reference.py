"""
Layer 1 reference panel — standalone PNG of the gate scatter plot.

Used as a reference image for PaperBanana (or any AI image generator)
to recreate the Layer 1 mini-plot with the *correct* distribution of
50 inspection conditions.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/lch/home/code/tunnelscanning")
DATA_DIR = ROOT / "01_tunnelscanning/04_data/b2_results"
OUT_DIR = ROOT / "01_tunnelscanning/04_data/b2_results/figures"

C_PASS = "#2ca02c"
C_FAIL = "#d62728"
LAYER1_TINT = "#f4f9fd"
EDGE = "#222222"

BETA0, BETA_CM, BETA_L90 = -44.88, 104.59, 0.163

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

df = pd.read_csv(DATA_DIR / "wide_with_shading.csv")
mask_y = df["identifiable_bin"] == 1

fig, ax = plt.subplots(figsize=(4.5, 3.6))
ax.set_facecolor(LAYER1_TINT)

ax.scatter(df.loc[mask_y, "C_M"], df.loc[mask_y, "L_90"],
           c=C_PASS, s=70, alpha=0.85, edgecolor="k", linewidth=0.5,
           label=fr"$G = 1$  (n={mask_y.sum()})")
ax.scatter(df.loc[~mask_y, "C_M"], df.loc[~mask_y, "L_90"],
           c=C_FAIL, s=120, marker="X", edgecolor="k", linewidth=0.7,
           label=fr"$G = 0$  (n={(~mask_y).sum()})")

cm_grid = np.linspace(0, 0.55, 200)
l90_bound = (-BETA0 - BETA_CM * cm_grid) / BETA_L90
ax.plot(cm_grid, l90_bound, "k--", linewidth=1.6,
        label=r"Logistic boundary $\hat{s}(\mathbf{x}) = \tau^\star$")

ax.set_xlim(0, 0.55)
ax.set_ylim(40, 230)
ax.set_xlabel(r"$C_M$  (Michelson contrast)", fontsize=12)
ax.set_ylabel(r"$L_{90}$  (90th-percentile luminance, DN)", fontsize=12)
ax.set_title("Layer 1: Gate — Real Data Distribution\n(reference for AI re-rendering)",
             fontsize=11, fontweight="bold")
ax.tick_params(labelsize=10)
ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.95)
ax.grid(True, alpha=0.3, linewidth=0.5)

# Annotation: range guidance
ax.text(0.025, 50, "G=0 cluster:\nlow $C_M$ AND low $L_{90}$",
        fontsize=9, color=C_FAIL, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_FAIL, linewidth=0.6))

ax.text(0.36, 215, "G=1 region:\nupper-right",
        fontsize=9, color=C_PASS, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_PASS, linewidth=0.6))

out = OUT_DIR / "F9_layer1_reference.png"
fig.savefig(out)
plt.close(fig)
print(f"[OK] Saved: {out}")
