"""
A2: IQ-conditioned stratified POD via median split
Input:  tmp/b2_out/long_v2.csv (500 rows)
Output: tmp/v2_analysis/stratified_pod_iq.json + stratified_pod_curves.csv

Procedure (per metric in {C_M, L_90, mtf_h}):
  1. Compute median over 50 cond-level values (de-dup via condition key).
  2. Assign each condition to high/low group by metric.
  3. Restrict 500-row long table to that group (~250 rows each).
  4. Fit logit P(detected) = a + b * width_mm.
  5. Compute a50, a90, a90/95 (Wald, Berens 95%).
  6. Bootstrap (B=2000) on conditions (cluster bootstrap) for a90 CI.

Output: per metric x {high, low} -> {b0, b1, a50, a90, a90_95, ci_a90}
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy import stats

INPUT = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/long_v3.csv"
OUT_JSON = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/stratified_pod_iq.json"
OUT_CURVES = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/stratified_pod_curves.csv"
METRICS = ["C_M", "L_90", "mtf_h", "bew_h"]
B = 2000
RNG = np.random.default_rng(42)
A_GRID = np.linspace(0.05, 1.5, 146)


def fit_pod(width, det):
    width = np.asarray(width, dtype=float).reshape(-1, 1)
    det = np.asarray(det, dtype=int)
    if det.sum() == 0 or det.sum() == len(det):
        return None
    m = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
    m.fit(width, det)
    b0 = float(m.intercept_[0])
    b1 = float(m.coef_[0][0])
    if b1 <= 0:
        return None
    a50 = -b0 / b1
    a90 = (np.log(0.9 / 0.1) - b0) / b1
    p = m.predict_proba(width)[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = float(np.sum(det * np.log(p) + (1 - det) * np.log(1 - p)))
    return {"b0": b0, "b1": b1, "a50": float(a50), "a90": float(a90),
            "n": int(len(det)), "n_pos": int(det.sum()), "log_lik": ll}


def a90_95_berens(b0, b1, X, y):
    """95% upper bound on a90 via Wald on logit at p=0.9 -> back-solve."""
    X = np.asarray(X).reshape(-1, 1)
    Xd = np.hstack([np.ones_like(X), X])
    p_pred = 1 / (1 + np.exp(-(b0 + b1 * X.flatten())))
    W = p_pred * (1 - p_pred)
    XtWX = Xd.T @ (W[:, None] * Xd)
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return float("nan")
    a90 = (np.log(0.9 / 0.1) - b0) / b1
    grad_b0 = -1.0 / b1
    grad_b1 = -(np.log(0.9 / 0.1) - b0) / (b1 ** 2)
    var_a90 = grad_b0 ** 2 * cov[0, 0] + grad_b1 ** 2 * cov[1, 1] + 2 * grad_b0 * grad_b1 * cov[0, 1]
    se = float(np.sqrt(max(var_a90, 0.0)))
    return float(a90 + 1.645 * se)


def cluster_bootstrap_a90(df, B=2000):
    cond_keys = df.groupby(["speed", "iso", "dist"]).ngroup().values
    unique_cond = np.unique(cond_keys)
    a90s = []
    for _ in range(B):
        chosen = RNG.choice(unique_cond, size=len(unique_cond), replace=True)
        rows = []
        for c in chosen:
            rows.append(df.iloc[np.where(cond_keys == c)[0]])
        sub = pd.concat(rows, ignore_index=True)
        r = fit_pod(sub["width_mm"].values, sub["detected"].values)
        if r is not None and np.isfinite(r["a90"]) and r["a90"] > 0:
            a90s.append(r["a90"])
    if not a90s:
        return None
    arr = np.asarray(a90s)
    return {"B_completed": int(len(arr)),
            "mean": float(arr.mean()),
            "ci95_percentile": [float(np.percentile(arr, 2.5)),
                                 float(np.percentile(arr, 97.5))]}


def main():
    df = pd.read_csv(INPUT)
    out = {"medians": {}, "results": {}, "n_conditions_with_metric": {}}
    curves_rows = []

    for m in METRICS:
        # per-metric drop NaN at condition level (preserves all available data)
        df_m = df.dropna(subset=[m, "width_mm", "detected"])
        cond_m = df_m.drop_duplicates(["speed", "iso", "dist"])[
            ["speed", "iso", "dist", m]
        ]
        out["n_conditions_with_metric"][m] = int(len(cond_m))
        med = float(cond_m[m].median())
        out["medians"][m] = med
        # group assignment: condition -> {high, low}
        hi_keys = cond_m[cond_m[m] >= med][["speed", "iso", "dist"]]
        lo_keys = cond_m[cond_m[m] < med][["speed", "iso", "dist"]]
        hi = df_m.merge(hi_keys, on=["speed", "iso", "dist"], how="inner")
        lo = df_m.merge(lo_keys, on=["speed", "iso", "dist"], how="inner")
        out["results"][m] = {}
        for label, sub in [("high", hi), ("low", lo)]:
            r = fit_pod(sub["width_mm"].values, sub["detected"].values)
            if r is None:
                out["results"][m][label] = {"error": "fit failed", "n": int(len(sub))}
                continue
            a90_95 = a90_95_berens(r["b0"], r["b1"], sub["width_mm"].values, sub["detected"].values)
            ci = cluster_bootstrap_a90(sub, B)
            out["results"][m][label] = {
                **r, "a90_95": a90_95, "bootstrap_a90": ci,
                "n_conditions": int(sub.drop_duplicates(["speed", "iso", "dist"]).shape[0]),
            }
            # POD curve points
            for a in A_GRID:
                pod = 1.0 / (1.0 + np.exp(-(r["b0"] + r["b1"] * a)))
                curves_rows.append({"metric": m, "group": label, "width_mm": float(a), "pod": float(pod)})

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    pd.DataFrame(curves_rows).to_csv(OUT_CURVES, index=False)
    print(f"[A2] wrote {OUT_JSON}")
    print(f"[A2] wrote {OUT_CURVES}")
    for m in METRICS:
        for g in ["high", "low"]:
            r = out["results"][m].get(g, {})
            if "a90" in r:
                ci = r.get("bootstrap_a90", {}).get("ci95_percentile", [float("nan"), float("nan")])
                print(f"  {m:6s} {g:4s}  a50={r['a50']:.3f}  a90={r['a90']:.3f}  a90/95={r['a90_95']:.3f}  ci=[{ci[0]:.3f}, {ci[1]:.3f}]  n_cond={r['n_conditions']}")


if __name__ == "__main__":
    main()
