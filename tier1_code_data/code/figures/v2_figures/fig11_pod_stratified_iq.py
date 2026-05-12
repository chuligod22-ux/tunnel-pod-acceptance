"""Sec 4.5 — Stratified POD curves under median-split on L_90 and BEW_H.

Two-panel figure:
  Top: L_90 high (solid) vs low (dashed) — directionality matches Eq. (5)
  Bottom: BEW_H low (sharper, solid) vs high (blurrier, dashed)

Outputs: ../F11_pod_stratified_iq.png and ../F11_pod_stratified_iq.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(out_dir: Path) -> None:
    # ---- Modern academic style (matched to F8) ----
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
    label_bbox = dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.85)

    curves = pd.read_csv("/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/stratified_pod_curves.csv")

    with open("/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/stratified_pod_iq.json") as f:
        strat = json.load(f)["results"]

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 6.4), sharex=True)

    panels = [
        ("L_90", "$L_{90}$", "high", "low",
         "high $L_{90}$ (better exposure)", "low $L_{90}$ (under-exposed)"),
        ("bew_h", "$\\mathrm{BEW}_H$", "low", "high",
         "low $\\mathrm{BEW}_H$ (sharper)", "high $\\mathrm{BEW}_H$ (blurrier)"),
    ]

    for ax, (metric, label, better, worse, lbl_better, lbl_worse) in zip(axes, panels):
        df_better = curves[(curves["metric"] == metric) & (curves["group"] == better)]
        df_worse = curves[(curves["metric"] == metric) & (curves["group"] == worse)]
        ax.plot(df_better["width_mm"], df_better["pod"], color=NAVY, linewidth=2.2,
                linestyle="-", label=lbl_better, zorder=3)
        ax.plot(df_worse["width_mm"], df_worse["pod"], color=ACCENT, linewidth=2.0,
                linestyle="--", label=lbl_worse, zorder=3)

        s = strat[metric]
        a9095_better = s[better]["a90_95"]
        a9095_worse = s[worse]["a90_95"]
        ax.plot([a9095_better, a9095_better], [0, 0.9], color=NAVY,
                linestyle=":", linewidth=0.9, alpha=0.55, zorder=2)
        ax.plot([a9095_worse, a9095_worse], [0, 0.9], color=ACCENT,
                linestyle=":", linewidth=0.9, alpha=0.55, zorder=2)
        ax.scatter([a9095_better], [-0.018], marker="^", s=22, color=NAVY,
                   clip_on=False, zorder=5)
        ax.scatter([a9095_worse], [-0.018], marker="^", s=22, color=ACCENT,
                   clip_on=False, zorder=5)

        # Subtle reference horizontals
        ax.axhline(0.5, color=GRID, linewidth=0.8, zorder=1)
        ax.axhline(0.9, color=GRID, linewidth=0.8, zorder=1)

        # White-boxed a_{90/95} labels placed at each vertical guide, aligned at y = 0.4
        ax.text(a9095_better - 0.025, 0.40,
                rf"$a_{{90/95}}^{{\rm high}} = {a9095_better:.3f}\,$mm",
                fontsize=8.6, color=NAVY, va="center", ha="right",
                bbox=label_bbox, zorder=5)
        ax.text(a9095_worse + 0.025, 0.40,
                rf"$a_{{90/95}}^{{\rm low}} = {a9095_worse:.3f}\,$mm",
                fontsize=8.6, color=ACCENT, va="center", ha="left",
                bbox=label_bbox, zorder=5)

        ax.set_ylabel(r"$\widehat{\mathrm{POD}}(w)$", fontsize=10.5, color=TXT)
        ax.set_xlim(0.0, 1.5)
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.tick_params(axis="both", labelsize=9.3, colors=TXT_MUTED)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color=GRID, alpha=0.7)
        ax.set_axisbelow(True)
        leg = ax.legend(fontsize=8.4, loc="upper left",
                        frameon=True, framealpha=0.95, edgecolor="#D7DDE3",
                        handlelength=1.9, handletextpad=0.6, borderaxespad=0.8)
        leg.get_frame().set_linewidth(0.6)

    # (Per-panel and overall titles removed — figure caption is provided in the manuscript body.)

    axes[1].set_xticks(np.arange(0.0, 1.51, 0.25))
    axes[1].set_xlabel(r"Reference crack width  $w$  (mm)", fontsize=10.5, color=TXT)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "F11_pod_stratified_iq.png", dpi=240, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out_dir / "F11_pod_stratified_iq.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"Wrote: {out_dir / 'F11_pod_stratified_iq.png'}")
    print(f"Wrote: {out_dir / 'F11_pod_stratified_iq.pdf'}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent
    main(out_dir)
