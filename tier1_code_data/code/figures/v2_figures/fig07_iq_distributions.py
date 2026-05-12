"""Sec 4.1 — Distributions of the five image-quality metrics across 50 conditions
and two cameras.

Box plots per IQ metric with cam1 (orange) vs cam2 (blue) reported side-by-side.
σ_motion is restricted to the 91 entries where Eq. (8) is defined.

Outputs: ../F7_iq_distributions.png and ../F7_iq_distributions.pdf
"""

from __future__ import annotations

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
    CAM1_COLOR = "#E69F00"  # cam1 amber
    CAM2_COLOR = NAVY        # cam2 navy
    GRID = "#D7DDE3"
    TXT = "#1F2933"
    TXT_MUTED = "#5C6770"

    df = pd.read_csv("/Users/lch/home/code/tunnelscanning/tmp/long_iq_100.csv")

    metrics = [
        ("C_M", "$C_M$ (–)", None),
        ("L_90", "$L_{90}$ (DN)", None),
        ("mtf_h", "MTF50 (cy/px)", None),
        ("bew_h", "$\\mathrm{BEW}_H$ (px)", None),
        ("sigma_motion", "$\\sigma_{\\mathrm{motion}}$ (px)", "bew_hgv"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.4))

    for ax, (col, label, gate) in zip(axes, metrics):
        df_local = df.dropna(subset=[col])
        if gate is not None:
            df_local = df_local[df_local[gate] == "Y"]
        cam1_vals = df_local.loc[df_local["camera"] == "cam1", col].values
        cam2_vals = df_local.loc[df_local["camera"] == "cam2", col].values

        positions = [1, 2]
        bp = ax.boxplot([cam1_vals, cam2_vals], positions=positions, widths=0.55,
                        patch_artist=True, showfliers=True,
                        medianprops={"color": "#0F1F33", "linewidth": 1.3},
                        whiskerprops={"color": "#0F1F33", "linewidth": 0.9},
                        capprops={"color": "#0F1F33", "linewidth": 0.9},
                        flierprops={"marker": "o", "markersize": 3.0,
                                    "markerfacecolor": "#888", "markeredgecolor": "none",
                                    "alpha": 0.55})
        for patch, color in zip(bp["boxes"], [CAM1_COLOR, CAM2_COLOR]):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("#0F1F33")
            patch.set_linewidth(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels([f"cam1\n($n={len(cam1_vals)}$)", f"cam2\n($n={len(cam2_vals)}$)"],
                           fontsize=9.0, color=TXT_MUTED)
        ax.set_ylabel(label, fontsize=10.0, color=TXT)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color=GRID, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8.8, colors=TXT_MUTED)

    # (Title removed — figure caption is provided in the manuscript body.)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "F7_iq_distributions.png", dpi=240, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out_dir / "F7_iq_distributions.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"Wrote: {out_dir / 'F7_iq_distributions.png'}")
    print(f"Wrote: {out_dir / 'F7_iq_distributions.pdf'}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent
    main(out_dir)
