#!/usr/bin/env python3
"""Fig. R3(a)/R3(b) — publication version, split into two standalone figures
(replaces draft viz/wp5_psf_uncertainty.png and the combined
viz/fig_R3_psf_uncertainty.png; user decision 2026-07-19: panel (a) is cited
in Sec 4.5 and panel (b) in Sec 4.6, so each panel is placed in its own
section as a separate figure).

R3(a): Condition-level detection rate vs measured edge-spread width, for the
    optics-only BEW_V (circles) and motion-affected BEW_H (triangles).
    Visual support for the weaker BEW_V-detection correlation cited in
    Sec 4.5 (r = -0.29 vs -0.36).
R3(b): Sign-stability of the M1 coefficients across the measurement-error
    Monte-Carlo refits of Sec 4.6 (dashed line: 95 %). Values read from
    wp5_psf_uncertainty.json (no re-run); proper symbol labels replace the
    draft's code variable names.

In-image panel sub-captions are dropped: each figure now carries its own
text caption in the manuscript.

Outputs: viz/fig_R3a_bew_detection.{png,pdf}, viz/fig_R3b_sign_stability.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "03_src/data"
CAM2 = DATA / "cam2_analysis.csv"
LONG = HERE.parent.parent.parent / "tmp/v2_analysis/long_v3.csv"
if not LONG.exists():
    LONG = (HERE.parent.parent / "01_paper/output/ieee_tim_v2/supplementary"
            "/tier1_code_data/data/long_v3.csv")
FIG = HERE / "viz"

LABELS = {  # code name -> display label (order preserved for the bar chart)
    "bew_h": r"$\mathrm{BEW}_H$",
    "mtf_h": r"$\mathrm{MTF50}_H$",
    "L_90": r"$L_{90}$",
    "C_M": r"$C_M$",
    "width_mm": "Crack width $w$",
    "const": "Intercept",
}


def main() -> None:
    cam2 = pd.read_csv(CAM2, encoding="utf-8-sig")
    long_df = pd.read_csv(LONG)
    det = long_df.groupby(["speed", "iso", "dist"])["detected"].mean().reset_index()
    det.columns = ["speed", "iso", "dist", "det_rate"]
    merged = det.merge(
        cam2[["speed", "iso", "dist", "bew_v_mean", "bew_h_mean"]],
        on=["speed", "iso", "dist"], how="inner",
    )
    sens = json.load(open(HERE / "wp5_psf_uncertainty.json", encoding="utf-8"))
    cs = sens["pod_sensitivity"]["coef_stability"]

    FIG.mkdir(exist_ok=True)

    # R3(a) — BEW vs detection (Sec 4.5)
    fig_a, ax_a = plt.subplots(figsize=(5.4, 4.2))
    ax_a.scatter(merged["bew_v_mean"], merged["det_rate"], s=32, c="#2166ac",
                 label=r"$\mathrm{BEW}_V$ (optics only)")
    ax_a.scatter(merged["bew_h_mean"], merged["det_rate"], s=32, c="#b2182b",
                 marker="^", alpha=0.65, label=r"$\mathrm{BEW}_H$ (optics + motion)")
    ax_a.set_xlabel("Edge-spread width (px)", fontsize=10)
    ax_a.set_ylabel("Condition detection rate", fontsize=10)
    ax_a.legend(fontsize=8.5, loc="upper right")
    ax_a.grid(alpha=0.3)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    fig_a.tight_layout()
    fig_a.savefig(FIG / "fig_R3a_bew_detection.png", dpi=200, bbox_inches="tight")
    fig_a.savefig(FIG / "fig_R3a_bew_detection.pdf", bbox_inches="tight")

    # R3(b) — coefficient sign-stability (Sec 4.6)
    names = [n for n in LABELS if n in cs]
    vals = [cs[n]["sign_stable_pct"] for n in names]
    disp = [LABELS[n] for n in names]
    fig_b, ax_b = plt.subplots(figsize=(5.4, 4.2))
    ax_b.barh(disp[::-1], vals[::-1], color="#4393c3")
    ax_b.axvline(95, ls="--", color="gray", lw=1)
    ax_b.set_xlim(0, 105)
    ax_b.set_xlabel("Coefficient sign-stability (%)", fontsize=10)
    ax_b.grid(alpha=0.3, axis="x")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    fig_b.tight_layout()
    fig_b.savefig(FIG / "fig_R3b_sign_stability.png", dpi=200, bbox_inches="tight")
    fig_b.savefig(FIG / "fig_R3b_sign_stability.pdf", bbox_inches="tight")

    print("n conditions:", len(merged), "| sign-stability:", dict(zip(names, vals)))
    print("Wrote:", FIG / "fig_R3a_bew_detection.png")
    print("Wrote:", FIG / "fig_R3b_sign_stability.png")


if __name__ == "__main__":
    main()
