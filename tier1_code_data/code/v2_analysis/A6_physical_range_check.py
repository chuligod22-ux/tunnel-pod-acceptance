"""A6 follow-up: fraction of per-axis threshold bootstrap replicates outside
the physically attainable range.

Re-executes the A6_perAxis_thresholds.py condition-clustered bootstrap with the
identical fixed seed (20260511), resampling unit (47 condition clusters), and
B = 2000, and reports the fraction of threshold replicates falling outside each
axis's hard physical range (L_90: [0, 255] DN for 8-bit data; C_M: [0, 1]
Michelson contrast). This is the single source for the statement in Sec 5.2 /
Table XI of the revised manuscript that the upper limb of the unconstrained
L_90 interval nominally exceeds the 255 DN ceiling in 3 % of resamples
(97 % of resamples fall at or below the ceiling).

Note: the percentile CI endpoints reproduce the archived
A6_perAxis_thresholds.json to < 0.1 % relative deviation; the exact endpoint
digits depend mildly on the scikit-learn version (lbfgs refits), which does not
affect the reported fractions at the quoted precision.

Output: results_json/A6_physical_range_check.json
"""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).parent
LONG = HERE.parent.parent / "data" / "long_v3.csv"
if not LONG.exists():
    LONG = Path("/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/long_v3.csv")
OUT = HERE.parent.parent / "results_json" / "A6_physical_range_check.json"

PREDICTORS = ["width_mm", "C_M", "L_90", "mtf_h", "bew_h"]
IQ_AXES = ["C_M", "L_90", "mtf_h", "bew_h"]
PHYS_BOUNDS = {"C_M": (0.0, 1.0), "L_90": (0.0, 255.0)}
A_STAR = 0.5
LOGIT_TARGET = float(np.log(0.9 / 0.1))
B = 2000
RNG_SEED = 20260511


def fit_m1(df):
    X = df[PREDICTORS].astype(float).values
    y = df["detected"].astype(int).values
    m = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
    m.fit(X, y)
    return dict(zip(["intercept"] + PREDICTORS,
                    np.concatenate([m.intercept_, m.coef_[0]])))


def invert_axis(coefs, axis_name, medians):
    other = sum(coefs[ax] * medians[ax] for ax in IQ_AXES if ax != axis_name)
    b_j = coefs[axis_name]
    if abs(b_j) < 1e-12:
        return float("nan")
    return (LOGIT_TARGET - coefs["intercept"] - coefs["width_mm"] * A_STAR - other) / b_j


def main():
    df_all = pd.read_csv(LONG)
    needed = PREDICTORS + ["detected", "sigma_motion", "speed", "iso", "dist"]
    df = df_all.dropna(subset=needed).copy()
    df["cond_id"] = (df["speed"].astype(str) + "_" + df["iso"].astype(str)
                     + "_" + df["dist"].astype(str))
    cond_ids = df["cond_id"].unique().tolist()
    medians = {ax: float(df[ax].median()) for ax in IQ_AXES}

    rng = np.random.default_rng(RNG_SEED)
    boot = {ax: [] for ax in IQ_AXES}
    by_cond = {cid: df[df["cond_id"] == cid] for cid in cond_ids}
    for _ in range(B):
        picks = rng.choice(cond_ids, size=len(cond_ids), replace=True)
        df_b = pd.concat([by_cond[cid] for cid in picks], ignore_index=True)
        coefs_b = fit_m1(df_b)
        for ax in IQ_AXES:
            boot[ax].append(invert_axis(coefs_b, ax, medians))

    out = {"method": ("re-execution of A6_perAxis_thresholds.py cluster bootstrap "
                      "(seed 20260511, B = 2000, 47 condition clusters, n = 470) "
                      "reporting fractions of threshold replicates outside hard "
                      "physical ranges"),
           "axes": {}}
    for ax in IQ_AXES:
        arr = np.asarray(boot[ax], float)
        arr = arr[np.isfinite(arr)]
        entry = {
            "n_finite_replicates": int(len(arr)),
            "ci_2p5": float(np.percentile(arr, 2.5)),
            "ci_97p5": float(np.percentile(arr, 97.5)),
        }
        if ax in PHYS_BOUNDS:
            lo, hi = PHYS_BOUNDS[ax]
            entry["physical_bounds"] = [lo, hi]
            entry["n_below_physical_min"] = int((arr < lo).sum())
            entry["n_above_physical_max"] = int((arr > hi).sum())
            entry["fraction_below_physical_min"] = float((arr < lo).mean())
            entry["fraction_above_physical_max"] = float((arr > hi).mean())
        out["axes"][ax] = entry

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    l90 = out["axes"]["L_90"]
    print(f"L_90: {l90['n_above_physical_max']}/{l90['n_finite_replicates']} "
          f"replicates > 255 DN ({100 * l90['fraction_above_physical_max']:.1f} %)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
