"""Fig. 6 — Representative cam1 raw inspection frames.

Four panels covering the parameter space and visibility classes:
  (a) Best:           v=60 km/h, ISO 200, d=2.5 m  -- algorithm-confirmed (MDW = 0.1 mm)
  (b) Typical mid:    v=80 km/h, ISO 200, d=4.5 m  -- algorithm-confirmed (MDW = 0.4 mm)
  (c) Boundary:       v=60 km/h, ISO 200, d=6.5 m  -- expert-flagged-fail (gate-pass, Y=0)
  (d) Gate-fail sat:  v=60 km/h, ISO 1600, d=2.5 m -- expert-flagged-fail (gate-fail, saturated)

Panels (a)-(c) form a distance ladder (2.5/4.5/6.5 m) at fixed v=60 km/h, ISO 200,
isolating the standoff-distance degradation toward the gate-pass boundary; (d)
breaks the pattern as a gate-fail saturation case.

Output: 01_tunnelscanning/01_paper/output/ieee_tim_v2/figures/F6_representative_imagery.{png,pdf}
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.image import imread

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "03_src" / "data" / "raw" / "crack" / "cam1"
OUT_DIR = ROOT / "01_paper" / "output" / "ieee_tim_v2" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANELS = [
    {
        "tag": "(a)",
        "file": "cam1_v60_iso200_d25.png",
        "cond": "v = 60 km/h, ISO 200, d = 2.5 m",
        "metrics": "C_M = 0.364,  L_90 = 191",
        "class_": "algorithm-confirmed (Y = 1, MDW = 0.1 mm)",
        "highlight": "best",
    },
    {
        "tag": "(b)",
        "file": "cam1_v80_iso200_d45.png",
        "cond": "v = 80 km/h, ISO 200, d = 4.5 m",
        "metrics": "C_M = 0.290,  L_90 = 109",
        "class_": "algorithm-confirmed (Y = 1, MDW = 0.4 mm)",
        "highlight": "typical",
    },
    {
        "tag": "(c)",
        "file": "cam1_v60_iso200_d65_x.png",
        "cond": "v = 60 km/h, ISO 200, d = 6.5 m",
        "metrics": "C_M = 0.315,  L_90 = 73",
        "class_": "expert-flagged-fail (Y = 0, gate-pass boundary)",
        "highlight": "boundary",
    },
    {
        "tag": "(d)",
        "file": "cam1_v60_iso1600_d25_x.png",
        "cond": "v = 60 km/h, ISO 1600, d = 2.5 m",
        "metrics": "C_M = 0.018,  L_90 = 255",
        "class_": "expert-flagged-fail (Y = 0, gate-fail, saturated)",
        "highlight": "fail",
    },
]


def fill_metrics_from_csv():
    """Optional: re-read wide csv to update C_M / L_90 strings precisely."""
    import pandas as pd
    csv = ROOT.parent / "tmp" / "b2_out" / "wide_50conditions.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df = df.rename(columns={"L_max": "L_90"})
    keys = [(60, 200, 2.5), (80, 200, 4.5), (60, 200, 6.5), (60, 1600, 2.5)]
    for p, (sp, iso, d) in zip(PANELS, keys):
        row = df[(df["speed"] == sp) & (df["iso"] == iso) & (df["dist"] == d)]
        if not row.empty:
            cm = float(row["C_M"].iloc[0])
            l90 = float(row["L_90"].iloc[0])
            p["metrics"] = f"$C_M$ = {cm:.3f},  $L_{{90}}$ = {l90:.0f}"


def main():
    fill_metrics_from_csv()

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), constrained_layout=False)
    axes_flat = axes.flatten()

    for ax, panel in zip(axes_flat, PANELS):
        img_path = RAW_DIR / panel["file"]
        img = imread(img_path)
        ax.imshow(img, cmap="gray", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Two-line sub-caption centred BELOW the panel
        ax.set_xlabel(
            f"{panel['tag']}  {panel['cond']}\n{panel['metrics']}",
            fontsize=9.5,
            loc="center",
            labelpad=6,
        )
        # Class label inside panel, lower-left (visual cue, complements caption)
        ax.text(
            0.015,
            0.04,
            panel["class_"],
            transform=ax.transAxes,
            fontsize=8.5,
            color="white",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2.5),
        )

    plt.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.06, wspace=0.05, hspace=0.32)

    out_png = OUT_DIR / "F6_representative_imagery.png"
    out_pdf = OUT_DIR / "F6_representative_imagery.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
