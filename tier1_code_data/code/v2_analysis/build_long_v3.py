"""
Build long_v3.csv:
  - Source IQ: tmp/long_iq_100.csv (cam1+cam2, 100 entries) -> condition-level mean (50 conditions)
  - Source detection: tmp/b2_out/long_v2.csv (cam1, 50 conditions x 10 widths = 500 rows)
  - Output: tmp/v2_analysis/long_v3.csv (500 rows = 50 cond x 10 widths)
    Columns: speed, iso, dist, C_M, L_90, mtf_h, mtf_v, bew_h, bew_v, sigma_motion,
             width_mm, detected
  - IQ values are cam1+cam2 mean per condition (sigma_motion uses nanmean to keep
    conditions where one camera has NaN).
"""
import pandas as pd
import numpy as np

IQ_PATH = "/Users/lch/home/code/tunnelscanning/tmp/long_iq_100.csv"
DET_PATH = "/Users/lch/home/code/tunnelscanning/tmp/b2_out/long_v2.csv"
OUT_PATH = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/long_v3.csv"

METRICS = ["C_M", "L_90", "mtf_h", "mtf_v", "bew_h", "bew_v", "sigma_motion"]


def main():
    iq = pd.read_csv(IQ_PATH)
    det = pd.read_csv(DET_PATH)[["speed", "iso", "dist", "width_mm", "detected"]]

    # Aggregate IQ by condition (cam1+cam2 mean, NaN-safe)
    agg = (
        iq.groupby(["speed", "iso", "dist"])[METRICS]
        .agg(lambda s: float(np.nanmean(s)) if s.notna().any() else np.nan)
        .reset_index()
    )

    # Sanity: 50 conditions
    assert len(agg) == 50, f"expected 50 conditions, got {len(agg)}"

    # Merge with detection
    v3 = det.merge(agg, on=["speed", "iso", "dist"], how="left")
    assert len(v3) == 500, f"expected 500 rows, got {len(v3)}"

    # Reorder columns
    cols = ["speed", "iso", "dist"] + METRICS + ["width_mm", "detected"]
    v3 = v3[cols]

    v3.to_csv(OUT_PATH, index=False)
    print(f"[build_v3] wrote {OUT_PATH}")
    print(f"shape: {v3.shape}")
    print()
    print("NaN counts per column:")
    print(v3.isna().sum())
    print()
    print("Per-condition IQ summary (cam1+cam2 mean):")
    cond = v3.drop_duplicates(["speed", "iso", "dist"])
    for m in METRICS:
        valid = cond[m].notna().sum()
        if valid > 0:
            print(f"  {m:14s}  n={valid:2d}  mean={cond[m].mean():.4f}  median={cond[m].median():.4f}  range=[{cond[m].min():.4f}, {cond[m].max():.4f}]")
    print()
    # Confirm detection unchanged
    assert v3["detected"].sum() == 374, f"detection changed: {v3['detected'].sum()}"
    print("[OK] Detection counts preserved (374/500 = 0.748).")


if __name__ == "__main__":
    main()
