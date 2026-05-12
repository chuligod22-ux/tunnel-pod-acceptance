"""
A6: Per-axis IQ thresholds at a*=0.5 mm via M1 numerical inversion.

Inputs
------
  /Users/lch/home/code/tunnelscanning/tmp/v2_analysis/A5b_extended_logistic_with_sigma.json
      (M1 point estimates; for reference / cross-check)
  /Users/lch/home/code/tunnelscanning/tmp/v2_analysis/long_v3.csv
      (500-row hit/miss; A5b uses sigma_motion-complete subsample n=470)

Method
------
  M1 (4-IQ, no sigma_motion, parsimony-aligned):
    logit P = b0 + b_w * w + b_C * C_M + b_L * L_90 + b_M * MTF50 + b_B * BEW_H

  Per-axis threshold x_j* at a* = 0.5 mm and P_d^target = 0.90:
    x_j* = ( logit(0.9) - b0 - b_w * 0.5 - sum_{k != j} b_k * tilde{x}_k ) / b_j
    with tilde{x}_k = sample median of axis k on the n=470 fit set.

Uncertainty
-----------
  Cluster bootstrap (B = 2000) at the condition level
  (Cameron, Gelbach & Miller 2008). Each bootstrap resample draws 50 conditions
  with replacement; M1 is refit and x_j* is recomputed. 95% CI = percentile
  [2.5, 97.5]. Fallback to normal-approximation CI if percentile spread >
  10x the point estimate (divergence guard).

Output
------
  /Users/lch/home/code/tunnelscanning/tmp/v2_analysis/A6_perAxis_thresholds.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LONG_CSV = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/long_v3.csv"
A5B_JSON = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/A5b_extended_logistic_with_sigma.json"
OUTPUT = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/A6_perAxis_thresholds.json"

PREDICTORS = ["width_mm", "C_M", "L_90", "mtf_h", "bew_h"]
IQ_AXES = ["C_M", "L_90", "mtf_h", "bew_h"]
A_STAR = 0.5
P_D_TARGET = 0.9
LOGIT_TARGET = float(np.log(P_D_TARGET / (1.0 - P_D_TARGET)))
B = 2000
RNG_SEED = 20260511


def fit_m1(df):
    """Fit M1 (4-IQ no sigma) logistic POD on a dataframe. Returns dict of coefs."""
    X = df[PREDICTORS].astype(float).values
    y = df["detected"].astype(int).values
    m = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
    m.fit(X, y)
    coefs = np.concatenate([m.intercept_, m.coef_[0]])
    names = ["intercept"] + PREDICTORS
    return dict(zip(names, coefs))


def invert_axis(coefs, axis_name, medians, a_star=A_STAR, logit_t=LOGIT_TARGET):
    """Numerical inversion: solve for axis_name threshold holding others at medians."""
    b0 = coefs["intercept"]
    b_w = coefs["width_mm"]
    # Sum of other-axis contributions
    other = 0.0
    for ax in IQ_AXES:
        if ax == axis_name:
            continue
        other += coefs[ax] * medians[ax]
    b_j = coefs[axis_name]
    if abs(b_j) < 1e-12:
        return float("nan")
    numerator = logit_t - b0 - b_w * a_star - other
    return numerator / b_j


def main():
    # Load complete-case subsample matching A5b (n=470)
    df_all = pd.read_csv(LONG_CSV)
    needed = PREDICTORS + ["detected", "sigma_motion", "speed", "iso", "dist"]
    df = df_all.dropna(subset=needed).copy()
    # Condition cluster id
    df["cond_id"] = (
        df["speed"].astype(str) + "_" +
        df["iso"].astype(str) + "_" +
        df["dist"].astype(str)
    )
    n_rows = len(df)
    cond_ids = df["cond_id"].unique().tolist()
    n_cond = len(cond_ids)

    # Sample medians (fixed values for inversion)
    medians = {ax: float(df[ax].median()) for ax in IQ_AXES}

    # Point estimate via fresh M1 fit on the same subsample
    coefs_point = fit_m1(df)

    # Cross-check against A5b stored coefficients
    with open(A5B_JSON) as f:
        a5b = json.load(f)
    a5b_coef = a5b["model1_4iq_no_sigma"]["coef"]
    coefs_ref = {k: a5b_coef[k]["value"] for k in ["intercept"] + PREDICTORS}
    max_dev = max(abs(coefs_point[k] - coefs_ref[k]) / max(abs(coefs_ref[k]), 1e-9)
                  for k in coefs_point)

    # Point thresholds
    thresholds_point = {ax: invert_axis(coefs_point, ax, medians) for ax in IQ_AXES}

    # Cluster bootstrap
    rng = np.random.default_rng(RNG_SEED)
    boot_thresholds = {ax: [] for ax in IQ_AXES}
    by_cond = {cid: df[df["cond_id"] == cid] for cid in cond_ids}
    failures = 0
    for b in range(B):
        picks = rng.choice(cond_ids, size=n_cond, replace=True)
        parts = [by_cond[cid] for cid in picks]
        df_b = pd.concat(parts, ignore_index=True)
        try:
            coefs_b = fit_m1(df_b)
            for ax in IQ_AXES:
                boot_thresholds[ax].append(invert_axis(coefs_b, ax, medians))
        except Exception:
            failures += 1

    # CIs
    out_thresholds = {}
    direction_map = {
        "C_M": "minimum",
        "L_90": "minimum",
        "mtf_h": "minimum (with caveat)",
        "bew_h": "maximum",
    }
    interp_map = {
        "C_M": "minimum Michelson contrast for P_d = 0.9 at a* = 0.5 mm",
        "L_90": "minimum 90th-percentile luminance for the same target",
        "mtf_h": "sign-reversed inversion owing to MTF50 close-distance saturation confounding (Sec 4.3, Sec 5.4)",
        "bew_h": "maximum horizontal edge-spread width for the same target",
    }
    sample_range = {ax: (float(df[ax].min()), float(df[ax].max())) for ax in IQ_AXES}
    warnings_out = []
    for ax in IQ_AXES:
        arr = np.asarray(boot_thresholds[ax], dtype=float)
        arr = arr[np.isfinite(arr)]
        point = thresholds_point[ax]
        if len(arr) >= 100:
            lo = float(np.percentile(arr, 2.5))
            hi = float(np.percentile(arr, 97.5))
            ci_method = "cluster_bootstrap_percentile"
        else:
            lo = float("nan"); hi = float("nan")
            ci_method = "insufficient_bootstrap"
            warnings_out.append(f"{ax}: insufficient finite bootstrap replicates")
        spread = (hi - lo) if np.isfinite(hi - lo) else float("nan")
        if np.isfinite(spread) and abs(point) > 0 and spread > 10.0 * abs(point):
            warnings_out.append(
                f"{ax}: bootstrap spread ({spread:.3g}) exceeds 10x point estimate "
                f"({point:.3g}); CI should be interpreted with caution"
            )
        out_thresholds[ax] = {
            "point": float(point),
            "ci_lower": lo,
            "ci_upper": hi,
            "direction": direction_map[ax],
            "interpretation": interp_map[ax],
            "ci_method": ci_method,
            "sample_min": sample_range[ax][0],
            "sample_max": sample_range[ax][1],
            "in_sample_range": bool(sample_range[ax][0] <= point <= sample_range[ax][1]),
            "bootstrap_n_finite": int(len(arr)),
        }

    out = {
        "method": "M1 (4-IQ, no sigma) numerical inversion at a*=0.5mm, P_d=0.9",
        "model_source": "A5b_extended_logistic_with_sigma.json:model1_4iq_no_sigma",
        "fit_check_max_relative_deviation_vs_A5b": float(max_dev),
        "logit_target": LOGIT_TARGET,
        "a_star_mm": A_STAR,
        "fixed_axis_values_medians": medians,
        "M1_coefficients_refit": {k: float(v) for k, v in coefs_point.items()},
        "thresholds": out_thresholds,
        "bootstrap_B": B,
        "bootstrap_failures": int(failures),
        "n_clusters": int(n_cond),
        "n_rows": int(n_rows),
        "warnings": warnings_out,
    }
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    # Compact stdout summary
    print(f"n_rows={n_rows} n_clusters={n_cond} max_dev_vs_A5b={max_dev:.2e}")
    print(f"medians: {medians}")
    print(f"a*={A_STAR} mm  P_d_target={P_D_TARGET}  logit={LOGIT_TARGET:.4f}")
    for ax in IQ_AXES:
        t = out_thresholds[ax]
        print(f"  {ax}: point={t['point']:.4f}  CI=[{t['ci_lower']:.4f}, {t['ci_upper']:.4f}]  "
              f"sample=[{t['sample_min']:.4f}, {t['sample_max']:.4f}]  in_range={t['in_sample_range']}")
    if warnings_out:
        print("WARNINGS:")
        for w in warnings_out:
            print(f"  - {w}")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
