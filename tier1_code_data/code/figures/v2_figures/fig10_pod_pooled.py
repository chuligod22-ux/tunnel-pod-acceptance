"""Sec 4.4 — Pooled POD curve (M_0, n=470 σ_motion-complete subsample).

Plots the linear-in-w logistic POD curve, augmented with:
  - per-reference-width binned detection rate (47 trials per width on the
    σ_motion-complete subsample) + Wilson 95% CI error bars
  - 0.5 / 0.9 reference levels marking a_50 / a_90
  - 0.880 plateau ceiling imposed by 6 expert-flagged-fail conditions
  - a_{90/95} (Wald) upper bound

Outputs: ../F10_pod_pooled.png and ../F10_pod_pooled.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


def wilson_ci(n_hits: int, n_trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n_trials == 0:
        return 0.0, 0.0
    from math import sqrt
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = n_hits / n_trials
    denom = 1 + z**2 / n_trials
    centre = (p + z**2 / (2 * n_trials)) / denom
    margin = (z * sqrt(p * (1 - p) / n_trials + z**2 / (4 * n_trials**2))) / denom
    return centre - margin, centre + margin


def main(out_dir: Path) -> None:
    # ---- Modern academic style ----
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
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.visible": True,
        "xtick.minor.size": 2.5,
        "xtick.minor.width": 0.6,
    })

    NAVY = "#1F4E79"
    NAVY_DEEP = "#143A5C"
    ACCENT = "#C75B12"  # warmer orange than #B05A1A
    GRID = "#D7DDE3"
    TXT = "#1F2933"
    TXT_MUTED = "#5C6770"

    json_path = Path("/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/A5b_extended_logistic_with_sigma.json")
    with open(json_path) as f:
        d = json.load(f)

    m0 = d["model0_basic"]
    b0 = m0["coef"]["intercept"]["value"]
    b1 = m0["coef"]["width_mm"]["value"]
    pod = d["M0_pod_summary"]
    a50 = pod["a50_mm"]
    a90 = pod["a90_mm"]
    a9095 = pod["a90_95_wald_mm"]

    df = pd.read_csv("/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/long_v3.csv")
    df_cc = df.dropna(subset=["sigma_motion"]).reset_index(drop=True)
    grouped = df_cc.groupby("width_mm")["detected"].agg(["sum", "count"]).reset_index()
    grouped["rate"] = grouped["sum"] / grouped["count"]
    grouped["ci_lo"], grouped["ci_hi"] = zip(*grouped.apply(
        lambda r: wilson_ci(int(r["sum"]), int(r["count"])), axis=1
    ))

    # Smooth logistic + 95 % delta-method confidence band on POD(w)
    w = np.linspace(0.0, 1.5, 600)
    eta = b0 + b1 * w
    pod_curve = 1.0 / (1.0 + np.exp(-eta))
    se_b0 = m0["coef"]["intercept"]["se"]
    se_b1 = m0["coef"]["width_mm"]["se"]
    var_eta = se_b0**2 + (w * se_b1) ** 2  # (no covariance term — best-effort)
    se_pod = pod_curve * (1.0 - pod_curve) * np.sqrt(var_eta)
    pod_lo = np.clip(pod_curve - 1.96 * se_pod, 0.0, 1.0)
    pod_hi = np.clip(pod_curve + 1.96 * se_pod, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    # ---- 1. Subtle horizontal reference lines (very light) ----
    for y in (0.5, 0.9):
        ax.axhline(y, color=GRID, linewidth=0.8, zorder=0)

    # 0.880 ceiling — short dashed accent
    ax.axhline(0.880, color=ACCENT, linewidth=1.0, linestyle=(0, (4, 3)),
               alpha=0.55, zorder=1)
    ax.text(1.46, 0.892, "0.880 ceiling", ha="right", va="bottom",
            fontsize=8.0, color=ACCENT, alpha=0.85, style="italic")

    # ---- 2. POD curve + CI band ----
    ax.fill_between(w, pod_lo, pod_hi, color=NAVY, alpha=0.10, linewidth=0, zorder=2)
    ax.plot(w, pod_curve, color=NAVY, linewidth=2.4, zorder=3,
            label=r"$M_0$ POD : logit $= \beta_0 + \beta_1 w$")

    # ---- 3. Per-width binned data points ----
    err_low = grouped["rate"] - grouped["ci_lo"]
    err_high = grouped["ci_hi"] - grouped["rate"]
    ax.errorbar(grouped["width_mm"], grouped["rate"],
                yerr=[err_low, err_high],
                fmt="o",
                color=NAVY_DEEP,
                ecolor=NAVY_DEEP,
                markersize=6.0,
                markerfacecolor="white",
                markeredgewidth=1.4,
                elinewidth=1.0,
                capsize=2.5,
                capthick=0.9,
                zorder=4,
                label=r"per-width rate ($n=47$ per width, Wilson 95% CI)")

    # ---- 4. Acceptance metric markers (short verticals) ----
    for x_val, anchor_y in [(a50, 0.5), (a90, 0.9)]:
        ax.plot([x_val, x_val], [0, anchor_y], color=NAVY, linewidth=0.8,
                linestyle=":", alpha=0.55, zorder=2)

    # a_{90/95} accent line — clean dashed vertical from the axis up to the 0.880 ceiling
    ax.plot([a9095, a9095], [0, 0.880], color=ACCENT, linewidth=1.4,
            linestyle="--", alpha=0.85, zorder=3,
            label=rf"$a_{{90/95}}\,(\mathrm{{Wald}})={a9095:.3f}\,$mm")
    # Subtle x-axis tick mark below the plot to anchor the value visually without intruding on the data area
    ax.scatter([a9095], [-0.018], marker="^", s=24, color=ACCENT,
               clip_on=False, zorder=5)

    # ---- 5. Inline acceptance-metric labels with subtle white background ----
    label_bbox = dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.85)
    ax.text(a50 + 0.025, 0.27, rf"$a_{{50}}={a50:.3f}\,$mm",
            fontsize=8.8, color=NAVY, va="center", ha="left",
            bbox=label_bbox, zorder=5)
    ax.text(a90 + 0.025, 0.67, rf"$a_{{90}}={a90:.3f}\,$mm",
            fontsize=8.8, color=NAVY, va="center", ha="left",
            bbox=label_bbox, zorder=5)

    # ---- 6. Axes cosmetics ----
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(np.arange(0.0, 1.51, 0.25))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel(r"Reference crack width  $w$  (mm)", fontsize=10.5, color=TXT)
    ax.set_ylabel(r"Probability of detection,  $\widehat{\mathrm{POD}}(w)$",
                  fontsize=10.5, color=TXT)
    ax.tick_params(axis="both", labelsize=9.5, colors=TXT_MUTED)

    # (Title removed — figure caption is provided in the manuscript body.)

    # Grid: subtle horizontal only
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color=GRID, alpha=0.7)
    ax.set_axisbelow(True)

    # ---- 7. Legend in lower-right (now clear of the shortened a_{90/95} accent) ----
    leg = ax.legend(fontsize=8.4, loc="lower right",
                    frameon=True, framealpha=0.95,
                    edgecolor="#D7DDE3",
                    handlelength=1.9, handletextpad=0.6,
                    borderaxespad=0.8)
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "F10_pod_pooled.png", dpi=240, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out_dir / "F10_pod_pooled.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"Wrote: {out_dir / 'F10_pod_pooled.png'}")
    print(f"Wrote: {out_dir / 'F10_pod_pooled.pdf'}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent
    main(out_dir)
