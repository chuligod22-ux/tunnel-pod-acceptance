"""Sec 3.6 — Numerical inversion flowchart for the IQ acceptance criterion.

Six-step procedure:
  1. Calibrated multivariable POD (M_2 from Sec 3.5)
  2. Dense 5-axis IQ grid
  3. a_{90/95}(x) at every grid point (delta-method + cluster bootstrap)
  4. Boundary identification
  5. Per-axis thresholds extraction
  6. Operational application (real-time + offline)

Outputs:
  ../F5_inversion_flowchart_matplotlib.png
  ../F5_inversion_flowchart_matplotlib.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, x, y, w, h, title, body, *, face="#F5F7FA", edge="#1F4E79"):
    """Render a rounded rectangle with a bold title and body text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y + h / 2 - 0.30,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=edge,
    )
    ax.text(
        x,
        y - 0.10,
        body,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#1F2933",
    )


def _arrow(ax, y_from, y_to, x=5.0, *, color="#1F4E79"):
    arrow = FancyArrowPatch(
        (x, y_from),
        (x, y_to),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.4,
        color=color,
    )
    ax.add_patch(arrow)


def main(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 11.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Step boxes (top → bottom) ----
    box_w, box_h = 7.6, 1.30
    centres_y = [12.40, 10.65, 8.90, 7.15, 5.40, 3.65]

    _box(
        ax,
        5,
        centres_y[0],
        box_w,
        box_h,
        "Step 1 — Calibrated multivariable POD",
        r"$M_2$: 5-predictor logistic POD $(C_M, L_{90}, \mathrm{MTF50}, \mathrm{BEW}_H, \sigma_{\mathrm{motion}};\ w)$  [Sec 3.5]",
        face="#E7EEF7",
    )

    _box(
        ax,
        5,
        centres_y[1],
        box_w,
        box_h,
        "Step 2 — Dense 5-axis IQ grid",
        "200 points along each of the five IQ axes\nbetween 5th and 95th condition-level percentiles",
    )

    _box(
        ax,
        5,
        centres_y[2],
        box_w,
        box_h,
        r"Step 3 — $a_{90/95}(\mathbf{x})$ at every grid point",
        r"Delta-method SE on $\hat{a}_{90}$  +  cluster-bootstrap upper percentile",
    )

    _box(
        ax,
        5,
        centres_y[3],
        box_w,
        box_h,
        r"Step 4 — Boundary identification $\partial \mathcal{A}(a^{\star})$",
        r"Contour where $a_{90/95}(\mathbf{x}) = a^{\star}$ (target detectable width)",
    )

    _box(
        ax,
        5,
        centres_y[4],
        box_w,
        box_h + 0.15,
        "Step 5 — Per-axis IQ thresholds",
        r"$C_{M,\min},\ L_{90,\min},\ \mathrm{MTF50}_{\min}\ \ |\ \ \mathrm{BEW}_{H,\max},\ \sigma_{\mathrm{motion},\max}$"
        + "\n(remaining axes held at condition-level medians)",
    )

    # Step 6: split into two outputs (real-time / offline)
    _box(
        ax,
        2.55,
        centres_y[5],
        4.20,
        1.30,
        "Step 6a — Real-time (per-frame)",
        "Frame admitted iff IQ vector lies in $\\mathcal{A}(a^{\\star})$;\n"
        "otherwise re-acquisition request",
        face="#FFF3E0",
        edge="#B05A1A",
    )
    _box(
        ax,
        7.45,
        centres_y[5],
        4.20,
        1.30,
        "Step 6b — Offline (per-condition)",
        "Campaign-level acceptance test on cam1+cam2\nmean IQ vector",
        face="#FFF3E0",
        edge="#B05A1A",
    )

    # ---- Arrows between boxes ----
    for i in range(len(centres_y) - 2):
        y_from = centres_y[i] - box_h / 2
        y_to = centres_y[i + 1] + box_h / 2 + 0.05
        _arrow(ax, y_from, y_to)

    # Step 5 → Step 6a / 6b (split arrows)
    y_split_top = centres_y[4] - (box_h + 0.15) / 2
    y_split_mid = y_split_top - 0.45
    ax.plot(
        [5.0, 5.0], [y_split_top, y_split_mid], color="#1F4E79", linewidth=1.4
    )
    ax.plot(
        [2.55, 5.0, 7.45], [y_split_mid, y_split_mid, y_split_mid],
        color="#1F4E79", linewidth=1.4,
    )
    for x_target in (2.55, 7.45):
        _arrow(ax, y_split_mid, centres_y[5] + 1.30 / 2 + 0.05, x=x_target)

    # ---- Title ----
    ax.text(
        5,
        13.15,
        "Numerical inversion of $M_2$ for the IQ acceptance criterion",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#1F2933",
    )

    # ---- Footer ----
    ax.text(
        5,
        2.45,
        r"Output: admissible region $\mathcal{A}(a^{\star})$ + five per-axis thresholds",
        ha="center",
        va="center",
        fontsize=10.5,
        style="italic",
        color="#1F4E79",
    )

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "F5_inversion_flowchart_matplotlib.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "F5_inversion_flowchart_matplotlib.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent
    main(out_dir)
    print(f"Wrote: {out_dir / 'F5_inversion_flowchart_matplotlib.png'}")
    print(f"Wrote: {out_dir / 'F5_inversion_flowchart_matplotlib.pdf'}")
