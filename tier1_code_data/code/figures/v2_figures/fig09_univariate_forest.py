"""Sec 4.3 — Univariate logistic forest plot.

Standardised slope coefficients ($\\hat{\\beta}_1$, per 1 SD of predictor) for
the crack width and the five IQ metrics, with 95 % Wald confidence intervals.
Each predictor is z-standardised so coefficients are directly comparable
across heterogeneous units (mm, DN, cy/px, px). The right margin annotates
AUC, sample size, and the LRT $p$ value.

Outputs: ../F9_univariate_forest.png and ../F9_univariate_forest.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score


# ---- Modern academic style (matched to F9) ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#2A2A2A",
    "xtick.color": "#2A2A2A",
    "ytick.color": "#2A2A2A",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
})

NAVY = "#1F4E79"
ACCENT = "#C75B12"
GRID = "#D7DDE3"
TXT = "#1F2933"
TXT_MUTED = "#5C6770"


PREDICTORS = [
    # (column, display label)
    ("width_mm", "Width $w$ (mm)"),
    ("C_M", "Michelson contrast $C_M$"),
    ("L_90", "$L_{90}$ (DN)"),
    ("mtf_h", "MTF50 (cy/px)"),
    ("bew_h", "$\\mathrm{BEW}_H$ (px)"),
    ("sigma_motion", "$\\sigma_{\\mathrm{motion}}$ (px)"),
]


def fit_standardised(df: pd.DataFrame, col: str) -> dict:
    sub = df.dropna(subset=[col]).reset_index(drop=True)
    x_raw = sub[col].astype(float).values
    x_std = (x_raw - np.mean(x_raw)) / np.std(x_raw, ddof=1)
    X = sm.add_constant(x_std)
    y = sub["detected"].astype(int).values
    res = sm.Logit(y, X).fit(disp=False, maxiter=200)
    b1 = float(res.params[1])
    se = float(res.bse[1])
    proba = res.predict(X)
    return {
        "b1": b1,
        "ci_lo": b1 - 1.96 * se,
        "ci_hi": b1 + 1.96 * se,
        "p": float(res.pvalues[1]),
        "auc": float(roc_auc_score(y, proba)),
        "n": int(len(y)),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent
    src = Path("/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/long_v3.csv")
    df = pd.read_csv(src)

    rows = []
    for col, label in PREDICTORS:
        f = fit_standardised(df, col)
        f["label"] = label
        rows.append(f)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y_positions = np.arange(len(rows))[::-1]
    for yp, row in zip(y_positions, rows):
        colour = NAVY if row["b1"] >= 0 else ACCENT
        ax.errorbar(row["b1"], yp,
                    xerr=[[row["b1"] - row["ci_lo"]],
                          [row["ci_hi"] - row["b1"]]],
                    fmt="o", color=colour, ecolor=colour,
                    markersize=7, markerfacecolor="white",
                    markeredgewidth=1.5, elinewidth=1.4,
                    capsize=3.5, capthick=1.0)

    ax.axvline(0.0, color="#444", linewidth=0.9, linestyle=":")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=10)
    ax.set_xlabel(r"Standardised $\hat{\beta}_1$ (per 1 SD of predictor)",
                  fontsize=10.5)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", labelsize=9.7)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color=GRID, alpha=0.7)
    ax.set_axisbelow(True)

    x_right = ax.get_xlim()[1]
    for yp, row in zip(y_positions, rows):
        ax.text(x_right * 1.02, yp,
                f"AUC = {row['auc']:.3f}   n = {row['n']}   p = {row['p']:.2g}",
                fontsize=8.5, va="center", ha="left", color=TXT_MUTED)
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 1.55)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "F9_univariate_forest.png",
                dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "F9_univariate_forest.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote: {out_dir / 'F9_univariate_forest.png'}")
    print(f"Wrote: {out_dir / 'F9_univariate_forest.pdf'}")


if __name__ == "__main__":
    main()
