"""
A4-mod: Per-width detection rate (visibility-based hit/miss aggregation)
Input:  tmp/b2_out/long_v2.csv  (500 rows: 50 conditions x 10 widths)
Output: tmp/v2_analysis/per_width_detection.json

For each width in {0.1, 0.2, ..., 1.0} mm:
  count detected=1, total=50, rate, Wilson 95% CI.
Plus overall detection rate.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

INPUT = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/long_v3.csv"
OUTPUT = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/per_width_detection.json"


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    df = pd.read_csv(INPUT)
    rows = []
    for w in sorted(df["width_mm"].unique()):
        sub = df[df["width_mm"] == w]
        k = int(sub["detected"].sum())
        n = int(len(sub))
        rate = k / n
        lo, hi = wilson_ci(k, n)
        rows.append({
            "width_mm": float(w), "n": n, "n_detected": k,
            "rate": float(rate),
            "ci95_wilson": [float(lo), float(hi)],
        })
    overall = {
        "n": int(len(df)),
        "n_detected": int(df["detected"].sum()),
        "rate": float(df["detected"].mean()),
        "n_conditions": int(df.drop_duplicates(["speed", "iso", "dist"]).shape[0]),
    }
    out = {"per_width": rows, "overall": overall}
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[A4-mod] wrote {OUTPUT}")
    for r in rows:
        ci = r["ci95_wilson"]
        print(f"  w={r['width_mm']:.1f} mm  {r['n_detected']:2d}/{r['n']}  rate={r['rate']:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  overall: {overall['n_detected']}/{overall['n']} = {overall['rate']:.3f}")


if __name__ == "__main__":
    main()
