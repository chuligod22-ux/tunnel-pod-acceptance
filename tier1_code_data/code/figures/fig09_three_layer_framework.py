"""
Figure 9: Three-Layer Acceptance Framework — Methodology Diagram with Data Visualizations
==========================================================================================
Companion to Section 2.6 of the IEEE TIM manuscript.

Layout (top -> bottom):
  Row A: Layer header strip (4 panels, color-coded)
  Row B: Mini-plot illustrating each layer's empirical result
           Layer 1 (Gate)       : C_M vs L_90 scatter with logistic decision boundary
           Layer 2 (POD)        : Gate-pass POD curve with 95% CI band, a_50/a_90 marked
           Layer 3 (Acceptance) : a_{90/95} bar comparison (pooled vs gate-pass) + Delta
           Validation           : Calibration plot of the logistic gate model with HL p, Brier
  Row C: Process steps (3 sequential operations per layer, written as step list)
  Row D: Equation summary (key formula + Eq. references)

IEEE TIM style: 7.16" two-column wide, 600 DPI, sans-serif 8 pt.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

ROOT = Path("/Users/lch/home/code/tunnelscanning")
DATA_DIR = ROOT / "01_tunnelscanning/04_data/b2_results"
OUT_DIR = ROOT / "01_tunnelscanning/04_data/b2_results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "mathtext.fontset": "dejavusans",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# Layer color palette (header_bg / mini-plot tint / step_bg / dark text)
LAYER1_BG = "#cfe2f3"; LAYER1_TINT = "#f4f9fd"; LAYER1_STEP = "#e3eef9"; LAYER1_DARK = "#2c5d87"
LAYER2_BG = "#d9ead3"; LAYER2_TINT = "#f5faf2"; LAYER2_STEP = "#e8f3e2"; LAYER2_DARK = "#316b1e"
LAYER3_BG = "#fce5cd"; LAYER3_TINT = "#fdf6ed"; LAYER3_STEP = "#fcedd9"; LAYER3_DARK = "#a35a18"
VALID_BG  = "#ead1dc"; VALID_TINT  = "#faf2f6"; VALID_STEP  = "#f1ddea"; VALID_DARK  = "#7a2750"

# Data accents
C_PASS = "#2ca02c"
C_FAIL = "#d62728"
C_POD = "#1f77b4"
C_VALID = "#9467bd"
EDGE = "#222222"
ARROW_COLOR = "#444444"

# Phase 1 results (results_summary.md §3-§7)
BETA0, BETA_CM, BETA_L90 = -44.88, 104.59, 0.163
A50, A90, A9095_GP = 0.122, 0.657, 0.922
A9095_POOL = 2.186
DELTA_PCT = (1.0 - A9095_GP / A9095_POOL) * 100
HL_P = 0.83
BRIER = 0.056

# ---------- Data ----------
df_wide = pd.read_csv(DATA_DIR / "wide_with_shading.csv")
df_pod_gp = pd.read_csv(DATA_DIR / "pod_curve_gate_pass.csv")
df_logp = pd.read_csv(DATA_DIR / "logistic_predictions.csv")

# ---------- Figure ----------
fig = plt.figure(figsize=(7.16, 7.0))
fig.suptitle("Three-Layer Acceptance Framework", fontsize=12, fontweight="bold", y=0.975)
fig.add_artist(plt.Line2D([0.30, 0.70], [0.953, 0.953], color=EDGE, linewidth=0.6,
                          transform=fig.transFigure))

# Geometry
left_margin, right_margin = 0.055, 0.985
gap = 0.022
total_w = right_margin - left_margin
panel_w = (total_w - 3 * gap) / 4
xs = [left_margin + i * (panel_w + gap) for i in range(4)]

# Row vertical layout (top -> bottom)
HEADER_BOTTOM = 0.885
HEADER_TOP    = 0.940

PLOT_BOTTOM   = 0.610
PLOT_TOP      = 0.875

STEPS_BOTTOM  = 0.345
STEPS_TOP     = 0.585

EQ_BOTTOM     = 0.085
EQ_TOP        = 0.320

layer_bgs = [LAYER1_BG, LAYER2_BG, LAYER3_BG, VALID_BG]
layer_tints = [LAYER1_TINT, LAYER2_TINT, LAYER3_TINT, VALID_TINT]
layer_steps = [LAYER1_STEP, LAYER2_STEP, LAYER3_STEP, VALID_STEP]
layer_darks = [LAYER1_DARK, LAYER2_DARK, LAYER3_DARK, VALID_DARK]
labels = ["Layer 1: Gate", r"Layer 2: POD$_{G=1}$", "Layer 3: Acceptance", "Validation"]

# ---------- Row A: Headers ----------
for x, lbl, bg, dk in zip(xs, labels, layer_bgs, layer_darks):
    ax_h = fig.add_axes([x, HEADER_BOTTOM, panel_w, HEADER_TOP - HEADER_BOTTOM])
    ax_h.set_xticks([]); ax_h.set_yticks([])
    for s in ax_h.spines.values():
        s.set_visible(False)
    ax_h.set_facecolor(bg)
    ax_h.text(0.5, 0.5, lbl, ha="center", va="center",
              fontsize=10, fontweight="bold", color=dk, transform=ax_h.transAxes)

# ---------- Row B: Mini-plots ----------
ax1 = fig.add_axes([xs[0], PLOT_BOTTOM, panel_w, PLOT_TOP - PLOT_BOTTOM])
ax2 = fig.add_axes([xs[1], PLOT_BOTTOM, panel_w, PLOT_TOP - PLOT_BOTTOM])
ax3 = fig.add_axes([xs[2], PLOT_BOTTOM, panel_w, PLOT_TOP - PLOT_BOTTOM])
ax4 = fig.add_axes([xs[3], PLOT_BOTTOM, panel_w, PLOT_TOP - PLOT_BOTTOM])
for ax, tint in zip([ax1, ax2, ax3, ax4], layer_tints):
    ax.set_facecolor(tint)

# --- Panel 1: Layer 1 ---
mask_y = df_wide["identifiable_bin"] == 1
ax1.scatter(df_wide.loc[mask_y, "C_M"], df_wide.loc[mask_y, "L_90"],
            c=C_PASS, s=22, alpha=0.85, edgecolor="k", linewidth=0.4, label="$G = 1$")
ax1.scatter(df_wide.loc[~mask_y, "C_M"], df_wide.loc[~mask_y, "L_90"],
            c=C_FAIL, s=42, marker="X", edgecolor="k", linewidth=0.5, label="$G = 0$")
cm_grid = np.linspace(0, 0.55, 200)
l90_bound = (-BETA0 - BETA_CM * cm_grid) / BETA_L90
ax1.plot(cm_grid, l90_bound, "k--", linewidth=1.1, label=r"$\hat{s} = \tau^\star$")
ax1.set_xlim(0, 0.55); ax1.set_ylim(40, 220)
ax1.set_xlabel(r"$C_M$", fontsize=9)
ax1.set_ylabel(r"$L_{90}$  (DN)", fontsize=9)
ax1.tick_params(labelsize=7.5)
ax1.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.9,
           handletextpad=0.4, borderpad=0.4, labelspacing=0.3)
ax1.grid(True, alpha=0.25, linewidth=0.4)

# --- Panel 2: Layer 2 ---
ax2.fill_between(df_pod_gp["width_mm"], df_pod_gp["pod_wald_lo"], df_pod_gp["pod_wald_hi"],
                 color=C_POD, alpha=0.20, label="95% CI")
ax2.plot(df_pod_gp["width_mm"], df_pod_gp["pod_point"], color=C_POD, linewidth=2.0,
         label=r"POD$_{G=1}(a)$")
ax2.axhline(0.5, color="gray", linestyle=":", linewidth=0.6)
ax2.axhline(0.9, color="gray", linestyle=":", linewidth=0.6)
ax2.axvline(A50, color=C_PASS, linestyle="--", linewidth=1.2)
ax2.axvline(A90, color=C_FAIL, linestyle="--", linewidth=1.2)
ax2.text(0.04, 0.96,
         fr"$a_{{50}} = {A50:.2f}$ mm" + "\n" + fr"$a_{{90}} = {A90:.2f}$ mm",
         transform=ax2.transAxes, ha="left", va="top",
         fontsize=7.5, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=EDGE, linewidth=0.5))
ax2.set_xscale("log")
ax2.set_xlim(0.05, 3); ax2.set_ylim(0, 1.05)
ax2.set_xlabel(r"crack width $a$ (mm)", fontsize=9)
ax2.set_ylabel(r"POD$_{G=1}(a)$", fontsize=9)
ax2.tick_params(labelsize=7.5)
ax2.grid(True, alpha=0.25, linewidth=0.4)

# --- Panel 3: Layer 3 ---
labels_a = ["Pooled", "Gate-pass"]
values_a = [A9095_POOL, A9095_GP]
colors_a = [C_FAIL, C_PASS]
ypos = np.arange(len(labels_a))
bars = ax3.barh(ypos, values_a, color=colors_a, edgecolor="k", linewidth=0.6, height=0.55)
for bar, val in zip(bars, values_a):
    if val > 1.5:
        ax3.text(val - 0.06, bar.get_y() + bar.get_height() / 2, f"{val:.2f} mm",
                 va="center", ha="right", fontsize=8.5, fontweight="bold", color="white")
    else:
        ax3.text(val + 0.06, bar.get_y() + bar.get_height() / 2, f"{val:.2f} mm",
                 va="center", ha="left", fontsize=8.5, fontweight="bold")
ax3.set_yticks(ypos); ax3.set_yticklabels(labels_a, fontsize=8)
ax3.set_xlim(0, 2.5)
ax3.set_xlabel(r"$a_{90/95}$ (mm)", fontsize=9)
ax3.tick_params(labelsize=7.5); ax3.invert_yaxis()
ax3.grid(True, alpha=0.25, linewidth=0.4, axis="x")
ax3.annotate("", xy=(A9095_GP + 0.05, 0.85), xytext=(A9095_POOL - 0.05, 0.15),
             arrowprops=dict(arrowstyle="-|>", color="#0b6e2f",
                             connectionstyle="arc3,rad=-0.45", linewidth=1.6))
ax3.text(1.30, 0.5, fr"$\Delta = {DELTA_PCT:.0f}\%$",
         ha="center", va="center", fontsize=11, fontweight="bold", color="#0b6e2f",
         bbox=dict(boxstyle="round,pad=0.32", fc="#e8f5e9", ec="#0b6e2f", linewidth=0.9))

# --- Panel 4: Validation ---
pred = df_logp["logistic_prob"].values
obs = df_logp["identifiable_bin"].values
n_bins = 5
order = np.argsort(pred)
pred_s, obs_s = pred[order], obs[order]
edges = np.linspace(0, len(pred_s), n_bins + 1, dtype=int)
xs_calib, ys_calib = [], []
for i in range(n_bins):
    if edges[i + 1] > edges[i]:
        xs_calib.append(pred_s[edges[i]:edges[i + 1]].mean())
        ys_calib.append(obs_s[edges[i]:edges[i + 1]].mean())
ax4.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="ideal")
ax4.plot(xs_calib, ys_calib, "-", color=C_VALID, linewidth=1.4)
ax4.scatter(xs_calib, ys_calib, color=C_VALID, s=42, edgecolor="k", linewidth=0.6, zorder=3,
            label="binned")
ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
ax4.set_xlabel(r"Predicted $\hat{\pi}$", fontsize=9)
ax4.set_ylabel(r"Observed $\bar{O}$", fontsize=9)
ax4.tick_params(labelsize=7.5)
ax4.grid(True, alpha=0.25, linewidth=0.4)
ax4.text(0.96, 0.04,
         f"HL $p = {HL_P}$\n" + fr"Brier $= {BRIER:.3f}$" + "\nAUC $= 0.96$",
         transform=ax4.transAxes, ha="right", va="bottom",
         fontsize=8, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=VALID_DARK, linewidth=0.7))

# ---------- Row C: Process Steps ----------
step_texts = [
    "① Compute $(C_M, L_{90})$\n   per condition\n\n"
    "② Fit multivariable\n   logistic regression\n\n"
    "③ Optimise $\\tau^\\star$ via\n   Youden's $J$\n\n"
    "④ Apply binary gate\n   $G \\in \\{0, 1\\}$",
    "① Filter to $G = 1$\n   subset ($n = 46$)\n\n"
    "② Fit log-logistic POD\n   on $(a, D)$ pairs\n\n"
    "③ Derive $a_{50}$, $a_{90}$\n   from sigmoid\n\n"
    "④ Bootstrap 95% CI\n   band on POD$(a)$",
    "① Compute $a^{G=1}_{90/95}$\n   via log-scale Wald\n\n"
    "② Compute pooled\n   $a^{\\mathrm{pool}}_{90/95}$ for comparison\n\n"
    "③ Quantify gate effect\n   $\\Delta = 1 - a^{G=1}/a^{\\mathrm{pool}}$\n\n"
    "④ Report acceptance\n   metric",
    "① Cluster bootstrap\n   ($B = 2000$, condition\n   level)\n\n"
    "② Hosmer-Lemeshow\n   GOF test (HL $p > 0.05$)\n\n"
    "③ Calibration plot,\n   Brier score, MCE\n\n"
    "④ Threshold robustness:\n   108 $(C_M, L_{90})$ cutoffs\n   yield 96% accuracy",
]
for x, txt, fc in zip(xs, step_texts, layer_steps):
    ax_s = fig.add_axes([x, STEPS_BOTTOM, panel_w, STEPS_TOP - STEPS_BOTTOM])
    ax_s.set_xticks([]); ax_s.set_yticks([])
    for s in ax_s.spines.values():
        s.set_linewidth(0.7); s.set_edgecolor(EDGE)
    ax_s.set_facecolor(fc)
    ax_s.text(0.06, 0.97, txt, ha="left", va="top", fontsize=7,
              transform=ax_s.transAxes, linespacing=1.25)

# Process row title (left side label)
fig.text(left_margin - 0.012, (STEPS_TOP + STEPS_BOTTOM) / 2,
         "Procedure", ha="right", va="center",
         fontsize=8, fontweight="bold", color=EDGE, rotation=90)

# ---------- Row D: Equations ----------
eq_texts = [
    r"$G = \mathbb{1}\!\left[\sigma\!\left(\hat{\beta}_0 + \hat{\beta}_1 C_M + \hat{\beta}_2 L_{90}\right) \geq \tau^\star\right]$"
    + "\n\nEq. (5), (13), (14)",
    r"$\mathrm{logit}\,\mathrm{POD}_{G=1}(a) = \beta_0' + \beta_1' \log a$"
    + "\n\nEq. (2), (15)-(18)",
    r"$a^{G=1}_{90/95} = \hat{a}^{G=1}_{90}\,e^{\,1.645\sqrt{\mathrm{Var}(\log\hat{a}^{G=1}_{90})}}$"
    + "\n" + r"with $\Delta$ (Eq. 20)" + "\nEq. (3), (4), (19), (20)",
    "Cluster bootstrap ($B = 2000$)\n+ Hosmer-Lemeshow GOF\n\nEq. (21)-(23)",
]
for x, eq, fc in zip(xs, eq_texts, layer_bgs):
    ax_eq = fig.add_axes([x, EQ_BOTTOM, panel_w, EQ_TOP - EQ_BOTTOM])
    ax_eq.set_xticks([]); ax_eq.set_yticks([])
    for s in ax_eq.spines.values():
        s.set_linewidth(0.8); s.set_edgecolor(EDGE)
    ax_eq.set_facecolor(fc)
    ax_eq.text(0.5, 0.5, eq, ha="center", va="center", fontsize=7.5,
               transform=ax_eq.transAxes)

# Eq row title
fig.text(left_margin - 0.012, (EQ_TOP + EQ_BOTTOM) / 2,
         "Mathematical specification", ha="right", va="center",
         fontsize=8, fontweight="bold", color=EDGE, rotation=90)

# ---------- Horizontal arrows between layers (3 rows) ----------
def add_h_arrows(y_center):
    for i in range(3):
        x_start = xs[i] + panel_w
        x_end = xs[i + 1]
        arr = FancyArrowPatch(
            (x_start, y_center), (x_end, y_center),
            arrowstyle="-|>", mutation_scale=18, linewidth=1.6, color=ARROW_COLOR,
            transform=fig.transFigure,
        )
        fig.patches.append(arr)

add_h_arrows((PLOT_TOP + PLOT_BOTTOM) / 2)
add_h_arrows((STEPS_TOP + STEPS_BOTTOM) / 2)
add_h_arrows((EQ_TOP + EQ_BOTTOM) / 2)

# G=1 label above first arrow
fig.text((xs[0] + panel_w + xs[1]) / 2, (PLOT_TOP + PLOT_BOTTOM) / 2 + 0.030,
         r"$G = 1$", ha="center", va="bottom", fontsize=8, color=C_PASS,
         fontweight="bold", style="italic")

# Output annotation
fig.text(0.5, 0.030,
         r"Output: smallest crack width $a^{G=1}_{90/95}$ for which $\mathrm{POD} \geq 0.90$ at 95% confidence, conditional on $G = 1$.",
         ha="center", fontsize=8, style="italic")

# ---------- Save ----------
out_pdf = OUT_DIR / "F9_three_layer_framework.pdf"
out_png = OUT_DIR / "F9_three_layer_framework.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
plt.close(fig)

print(f"[OK] Saved: {out_pdf}")
print(f"[OK] Saved: {out_png}")
