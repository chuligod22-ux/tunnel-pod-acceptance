"""Fig. R1 — External validation composite (publication asset).

(a) Representative CNN segmentation overlays under the SAME three exposure
    regimes as Fig. R5 (chart side), one frame per condition:
      a1 clear          60 km/h, ISO 200,  2.5 m  (frame_000043, 58k crack px)
      a2 over-exposed   60 km/h, ISO 1600, 3.5 m  (frame_000033, fragmentary)
      a3 under-exposed  60 km/h, ISO 100,  6.5 m  (frame_000021, zero px)
    Bands extracted from the per-condition diagnostic montages produced by
    `cnn_detect.py --viz-cond`, debug annotations cropped out.
(b) Condition-level CNN detection-rate heatmap over ISO x stand-off distance
    (mean over the two survey speeds), rebuilt from
    cam2_realcrack_condition_summary.csv.
(c) Mean CNN detection rate versus stand-off distance.

Sub-captions are placed centred below each panel (paper convention).

Outputs:
  viz/fig_R1_external_validation.png / .pdf
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
VIZ = HERE / "viz"

DISTS = [2.5, 3.5, 4.5, 5.5, 6.5]
ISOS = [100, 200, 400, 800, 1600]

TEXT_CROP_PX = 40  # remove debug annotation band at each strip top

# (montage, n_strips, strip_index, crack x-centre in montage px, in-panel tag)
# Crack x-centres: median of red overlay pixels in the strip (2.5/3.5 m) or, for
# the no-detection 6.5 m band, from the detected ISO 800 condition at the same
# stand-off (cnn_crack_d65_ISO800_V60.png, strip 5). The imaged wall section
# shifts with stand-off distance, so each band is windowed about the crack to
# keep the three panels visually comparable.
# display_gain: multiplicative display-only gain (annotated in the panel tag);
# the under-exposed band (mean ~28 DN) is otherwise near-black in print.
BANDS = [
    ("cnn_crack_d25_ISO200_V60.png", 16, 10, 758, 1.0,
     "clear: ISO 200, 2.5 m"),
    ("cnn_crack_d35_ISO1600_V60.png", 16, 8, 178, 1.0,
     "over-exposed: ISO 1600, 3.5 m"),
    ("cnn_crack_d65_ISO100_V60.png", 21, 10, 590, 4.0,
     "under-exposed: ISO 100, 6.5 m (no detection; ×4 display gain)"),
]
WINDOW_FRAC = 0.60      # crack-centred window width as fraction of band width
MAX_ASPECT = 0.42       # centre-crop taller windows to this height/width ratio


def load_rates() -> dict[tuple[int, float], float]:
    agg: dict[tuple[int, float], list[float]] = defaultdict(list)
    with open(HERE / "cam2_realcrack_condition_summary.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            agg[(int(r["iso"]), float(r["distance_m"]))].append(
                float(r["cnn_det_rate"])
            )
    return {k: sum(v) / len(v) for k, v in agg.items()}


def band_image(
    fname: str, n_strips: int, idx: int, cx: int, gain: float
) -> Image.Image:
    img = Image.open(VIZ / fname)
    w, h = img.size
    strip_h = h // n_strips
    band = img.crop((0, idx * strip_h + TEXT_CROP_PX, w, (idx + 1) * strip_h))
    win = int(band.width * WINDOW_FRAC)
    x0 = min(max(cx - win // 2, 0), band.width - win)
    band = band.crop((x0, 0, x0 + win, band.height))
    if band.height / band.width > MAX_ASPECT:  # tall far-distance band
        target = int(band.width * MAX_ASPECT)
        top = int((band.height - target) * 0.5)
        band = band.crop((0, top, band.width, top + target))
    if gain != 1.0:  # display-only gain, annotated in the panel tag
        arr = np.asarray(band).astype(np.float32) * gain
        band = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return band


def main() -> None:
    rates = load_rates()
    grid = np.array([[rates[(iso, d)] for d in DISTS] for iso in ISOS])
    dist_mean = grid.mean(axis=0)

    bands = [(band_image(f, n, i, cx, g), tag) for f, n, i, cx, g, tag in BANDS]

    fig = plt.figure(figsize=(9.6, 9.2))
    ratios = [b.height / b.width * 3.2 for b, _ in bands] + [1.55]
    gs = fig.add_gridspec(
        4, 2, height_ratios=ratios, width_ratios=[1.25, 1.0],
        hspace=0.18, wspace=0.30,
    )

    # (a) three overlay bands, full width
    for k, (band, tag) in enumerate(bands):
        ax = fig.add_subplot(gs[k, :])
        ax.imshow(band)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#555555")
            s.set_linewidth(0.8)
        ax.text(
            0.008, 0.94, tag, transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left", color="#1F2933",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
        )
        if k == len(bands) - 1:
            ax.set_xlabel(
                "(a) Representative CNN segmentation overlays under the three "
                "exposure regimes of Fig. R5 (60 km/h)",
                fontsize=10, labelpad=6,
            )

    # (b) heatmap
    ax_b = fig.add_subplot(gs[3, 0])
    im = ax_b.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax_b.set_xticks(range(len(DISTS)), [f"{d}" for d in DISTS])
    ax_b.set_yticks(range(len(ISOS)), [str(i) for i in ISOS])
    ax_b.set_ylabel("ISO", fontsize=10)
    for i in range(len(ISOS)):
        for j in range(len(DISTS)):
            ax_b.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.03)
    ax_b.set_xlabel(
        "Distance (m)\n(b) Condition-level detection rate, ISO × stand-off",
        fontsize=10,
    )

    # (c) distance curve
    ax_c = fig.add_subplot(gs[3, 1])
    ax_c.plot(DISTS, dist_mean, "o-", color="#1F4E79", linewidth=1.8, markersize=6)
    ax_c.set_ylim(0, 1.0)
    ax_c.set_ylabel("Mean CNN detection rate", fontsize=10)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(alpha=0.3)
    ax_c.set_xlabel(
        "Distance (m)\n(c) Mean detection rate vs stand-off", fontsize=10
    )

    VIZ.mkdir(exist_ok=True)
    fig.savefig(VIZ / "fig_R1_external_validation.png", dpi=200, bbox_inches="tight")
    fig.savefig(VIZ / "fig_R1_external_validation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("dist means:", [round(v, 3) for v in dist_mean])
    print("Wrote:", VIZ / "fig_R1_external_validation.png")


if __name__ == "__main__":
    main()
