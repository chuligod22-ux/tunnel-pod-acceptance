"""Sec 5.2 — Three-layer per-frame acceptance cascade diagram.

Layers:
  Layer 1 — Exposure gate     (L_90 admit/reject)
  Layer 2 — Sharpness gate    (BEW_H admit/reject)
  Layer 3 — Reliability gate  (a_{90/95}(x) ≤ a* via M_2)

Each layer admits the frame on pass and emits a re-acquisition request
on fail. Frames passing all three layers are accepted into the inspection
pipeline.

Outputs: ../F12_acceptance_cascade_matplotlib.png and ../F12_acceptance_cascade_matplotlib.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, x, y, w, h, title, body, *, face="#F5F7FA", edge="#1F4E79", title_size=11):
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
        fontsize=title_size,
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


def _arrow(ax, x_from, y_from, x_to, y_to, *, color="#1F4E79", style="-|>"):
    arrow = FancyArrowPatch(
        (x_from, y_from),
        (x_to, y_to),
        arrowstyle=style,
        mutation_scale=18,
        linewidth=1.4,
        color=color,
    )
    ax.add_patch(arrow)


def main(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 11.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.5)
    ax.set_aspect("equal")
    ax.axis("off")

    box_w_main, box_h = 6.4, 1.30
    cx_main = 3.6  # main cascade column (left side)
    cx_reject = 8.4  # reject column (right side)

    # Y positions
    y_input = 12.30
    y_l1 = 10.40
    y_l2 = 8.50
    y_l3 = 6.60
    y_admit = 4.70
    y_reject = (y_l1 + y_l2 + y_l3) / 3.0  # centered re-acquisition box on the right

    # Input box
    _box(
        ax,
        cx_main,
        y_input,
        box_w_main,
        box_h,
        "Input — acquired frame $f$",
        r"IQ vector $\mathbf{x} = (C_M, L_{90}, \mathrm{MTF50}, \mathrm{BEW}_H, \sigma_{\mathrm{motion}})$",
        face="#E7EEF7",
    )

    # Layer 1
    _box(
        ax,
        cx_main,
        y_l1,
        box_w_main,
        box_h,
        "Layer 1 — Exposure gate",
        r"Admit if $L_{90}$ within $[L_{90,\min},\ L_{90,\max}]$;" + "\n"
        + "reject otherwise (under-exposure or saturation)",
    )

    # Layer 2
    _box(
        ax,
        cx_main,
        y_l2,
        box_w_main,
        box_h,
        "Layer 2 — Sharpness gate",
        r"Admit if $\mathrm{BEW}_H \leq \mathrm{BEW}_{H,\max}$;" + "\n"
        + r"reject otherwise (insufficient resolution, Eq. 8 subsumes $\sigma_{\mathrm{motion}}$)",
    )

    # Layer 3
    _box(
        ax,
        cx_main,
        y_l3,
        box_w_main,
        box_h + 0.10,
        "Layer 3 — Reliability gate",
        r"Admit if $a_{90/95}(\mathbf{x}) \leq a^{\star}$ from calibrated $M_2$;" + "\n"
        + r"i.e. $\mathbf{x} \in \mathcal{A}(a^{\star})$ (Eq. 6)",
    )

    # Admit terminal
    _box(
        ax,
        cx_main,
        y_admit,
        box_w_main,
        box_h,
        "Admit — frame accepted",
        "Frame enters the inspection pipeline\n"
        "(downstream automated detection / archival)",
        face="#E5F2E5",
        edge="#2D6A2D",
    )

    # Reject terminal (right column)
    _box(
        ax,
        cx_reject,
        y_reject,
        2.6,
        1.40,
        "Re-acquisition\nrequest",
        r"Speed / ISO /" + "\n"
        + "stand-off retuned;\n"
        + "inspection vehicle\n"
        + "control system",
        face="#FFF3E0",
        edge="#B05A1A",
        title_size=10.5,
    )

    # ---- Cascade arrows (top → bottom on the main column) ----
    main_box_top = lambda y: y + box_h / 2 + 0.05
    main_box_bot = lambda y: y - box_h / 2 - 0.05
    for y_from, y_to in [
        (y_input, y_l1),
        (y_l1, y_l2),
        (y_l2, y_l3),
        (y_l3, y_admit),
    ]:
        _arrow(ax, cx_main, main_box_bot(y_from), cx_main, main_box_top(y_to))

    # ---- Reject branches (each layer → re-acquisition box) ----
    # Right edge of layer boxes
    layer_right_x = cx_main + box_w_main / 2 + 0.05
    reject_left_x = cx_reject - 2.6 / 2 - 0.05

    for y in (y_l1, y_l2, y_l3):
        # horizontal arrow from layer right edge to reject box
        ax.plot([layer_right_x, reject_left_x], [y, y], color="#B05A1A",
                linewidth=1.2, linestyle=":")

    # Single arrow head into the reject box at its centre
    _arrow(ax, reject_left_x - 0.15, y_reject, reject_left_x, y_reject,
           color="#B05A1A")

    # Label on the reject branch
    ax.text(
        (layer_right_x + reject_left_x) / 2,
        y_l3 - 0.4,
        "fail any layer\n(re-acquire)",
        ha="center", va="top",
        fontsize=8.5, style="italic", color="#B05A1A",
    )

    # ---- Title ----
    ax.text(
        5,
        13.10,
        "Three-layer per-frame acceptance cascade",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#1F2933",
    )

    # ---- Footer ----
    ax.text(
        5,
        3.40,
        "Real-time at sensor (per-frame) and offline auditing (per-condition cam1+cam2 mean IQ)",
        ha="center",
        va="center",
        fontsize=10.0,
        style="italic",
        color="#1F4E79",
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "F12_acceptance_cascade_matplotlib.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "F12_acceptance_cascade_matplotlib.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_dir / 'F12_acceptance_cascade_matplotlib.png'}")
    print(f"Wrote: {out_dir / 'F12_acceptance_cascade_matplotlib.pdf'}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent
    main(out_dir)
