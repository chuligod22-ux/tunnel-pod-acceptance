"""
A1: Per-metric hit/miss univariate logistic regression
Input:  tmp/b2_out/long_v2.csv  (500 rows: 50 cond x 10 widths)
Output: tmp/v2_analysis/per_metric_hitmiss.json

For each predictor x in {width_mm, C_M, L_90, mtf_h, sigma_motion}:
  fit logit P(detected) = b0 + b1*x
  report b0, b1, Wald CI, p-value, McFadden R^2, AUC, n
Bootstrap B=2000 for slope CI.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

INPUT = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/long_v3.csv"
OUTPUT = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/per_metric_hitmiss.json"
B = 2000
RNG = np.random.default_rng(42)


def fit_logit(x, y):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    model = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
    model.fit(x, y)
    b0 = float(model.intercept_[0])
    b1 = float(model.coef_[0][0])
    p = model.predict_proba(x)[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    p_null = float(np.mean(y))
    p_null_clip = np.clip(p_null, 1e-12, 1 - 1e-12)
    ll_null = float(np.sum(y * np.log(p_null_clip) + (1 - y) * np.log(1 - p_null_clip)))
    mcfadden = 1 - ll / ll_null if ll_null != 0 else float("nan")
    auc = float(roc_auc_score(y, p))

    # Wald SE via Fisher information
    p_pred = model.predict_proba(x)[:, 1]
    W = p_pred * (1 - p_pred)
    X = np.hstack([np.ones_like(x), x])
    XtWX = X.T @ (W[:, None] * X)
    try:
        cov = np.linalg.inv(XtWX)
        se_b0 = float(np.sqrt(cov[0, 0]))
        se_b1 = float(np.sqrt(cov[1, 1]))
        z = b1 / se_b1
        p_b1 = float(2 * (1 - stats.norm.cdf(abs(z))))
        ci_b1 = (b1 - 1.96 * se_b1, b1 + 1.96 * se_b1)
    except np.linalg.LinAlgError:
        se_b0 = se_b1 = p_b1 = float("nan")
        ci_b1 = (float("nan"), float("nan"))

    return {
        "n": int(len(y)),
        "b0": b0, "b1": b1,
        "se_b0": se_b0, "se_b1": se_b1,
        "p_value_b1": p_b1,
        "ci95_b1": [float(ci_b1[0]), float(ci_b1[1])],
        "mcfadden_r2": mcfadden,
        "auc": auc,
        "log_lik": ll,
    }


def bootstrap_slope(x, y, B):
    n = len(y)
    slopes = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        try:
            r = fit_logit(x[idx], y[idx])
            if r is not None:
                slopes.append(r["b1"])
        except Exception:
            continue
    if not slopes:
        return None
    arr = np.asarray(slopes)
    return {
        "B_completed": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ci95_percentile": [float(np.percentile(arr, 2.5)),
                            float(np.percentile(arr, 97.5))],
    }


def main():
    df = pd.read_csv(INPUT)
    predictors = ["width_mm", "C_M", "L_90", "mtf_h", "bew_h", "sigma_motion"]
    out = {"input_rows": int(len(df)),
           "n_pos_total": int(df["detected"].sum()),
           "n_neg_total": int(len(df) - df["detected"].sum()),
           "results": {}}
    for p in predictors:
        sub = df.dropna(subset=[p, "detected"])
        x = sub[p].astype(float).values
        y = sub["detected"].astype(int).values
        fit = fit_logit(x, y)
        boot = bootstrap_slope(x, y, B) if fit is not None else None
        out["results"][p] = {"n_used": int(len(sub)),
                              "n_pos": int(y.sum()),
                              "n_neg": int(len(y) - y.sum()),
                              "wald": fit, "bootstrap": boot}
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[A1] wrote {OUTPUT}")
    for p in predictors:
        rec = out["results"][p]
        r = rec["wald"]
        if r is not None:
            print(f"  {p:14s}  n={rec['n_used']:3d}  b1={r['b1']:+.4f}  p={r['p_value_b1']:.3g}  AUC={r['auc']:.3f}  R2={r['mcfadden_r2']:.3f}")


if __name__ == "__main__":
    main()
