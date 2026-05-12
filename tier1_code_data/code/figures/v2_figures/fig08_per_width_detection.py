"""Sec 4.2 — Per-reference-width detection rate with Wilson 95% CIs.

Bar plot of per-width detection rate (n=50 trials per width) with Wilson 95%
confidence intervals as error bars. Horizontal dashed line marks the 0.880
plateau imposed by the 6 expert-flagged-fail conditions.

Outputs: ../F8_per_width_detection.png and ../F8_per_width_detection.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

    src = Path("/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/per_width_detection.json")
    with open(src) as f:
        data = json.load(f)

    rows = data["per_width"]
    widths = np.array([r["width_mm"] for r in rows])
    rates = np.array([r["rate"] for r in rows])
    cis = np.array([r["ci95_wilson"] for r in rows])
    err_low = rates - cis[:, 0]
    err_high = cis[:, 1] - rates

    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    ax.bar(widths, rates, width=0.075, color=NAVY, alpha=0.82,
           edgecolor="#0F1F33", linewidth=0.6, zorder=2)
    ax.errorbar(widths, rates, yerr=[err_low, err_high],
                fmt="none", ecolor="#0F1F33", linewidth=0.9, capsize=2.8, zorder=3)

    ax.axhline(0.880, color=ACCENT, linestyle="--", linewidth=1.1, alpha=0.85,
               label="0.880 plateau (44/50, 6 expert-flagged-fail)", zorder=1)

    # Place numeric labels inside the bars (white, bold) to avoid overlap with error bars.
    for x, y in zip(widths, rates):
        ax.text(x, y - 0.04, f"{y:.2f}", ha="center", va="top",
                fontsize=8.2, color="white", fontweight="bold", zorder=4)

    ax.set_xlim(0.0, 1.1)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(widths)
    ax.set_xlabel(r"Reference crack width  $w$  (mm)", fontsize=10.5, color=TXT)
    ax.set_ylabel("Detection rate (Wilson 95 % CI)", fontsize=10.5, color=TXT)
    ax.tick_params(axis="both", labelsize=9.5, colors=TXT_MUTED)

    # (Title removed — figure caption is provided in the manuscript body.)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color=GRID, alpha=0.7)
    ax.set_axisbelow(True)

    leg = ax.legend(fontsize=8.4, loc="lower right",
                    frameon=True, framealpha=0.95, edgecolor="#D7DDE3",
                    handlelength=1.9, handletextpad=0.6, borderaxespad=0.8)
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "F8_per_width_detection.png", dpi=240, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out_dir / "F8_per_width_detection.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"Wrote: {out_dir / 'F8_per_width_detection.png'}")
    print(f"Wrote: {out_dir / 'F8_per_width_detection.pdf'}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent
    main(out_dir)
