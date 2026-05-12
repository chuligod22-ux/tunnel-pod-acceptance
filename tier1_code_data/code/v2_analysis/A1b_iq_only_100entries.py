"""
A1b: IQ-only single-metric AUC on 100 condition-camera entries
Input:  tmp/long_iq_100.csv
Output: tmp/v2_analysis/iq_only_100.json

Binary outcome: identifiable Y/N at the condition-camera level (cam1 + cam2 = 100).
Predictors: C_M, L_90, mtf_h, bew_h, sigma_motion.

Per metric, fit logit P(identifiable=Y) = b0 + b1*x and report b1 with Wald CI,
p-value, McFadden R^2, AUC. NaN rows are dropped per metric so all available data
is used (sigma_motion 91 entries; others 100).
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

INPUT = "/Users/lch/home/code/tunnelscanning/tmp/long_iq_100.csv"
OUTPUT = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/iq_only_100.json"


def fit_logit(x, y):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    m = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
    m.fit(x, y)
    b0 = float(m.intercept_[0])
    b1 = float(m.coef_[0][0])
    p = m.predict_proba(x)[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    p0 = float(np.mean(y)); p0c = np.clip(p0, 1e-12, 1 - 1e-12)
    ll0 = float(np.sum(y * np.log(p0c) + (1 - y) * np.log(1 - p0c)))
    r2 = 1 - ll / ll0
    auc = float(roc_auc_score(y, p))
    Xd = np.hstack([np.ones_like(x), x])
    W = (p * (1 - p))
    XtWX = Xd.T @ (W[:, None] * Xd)
    try:
        cov = np.linalg.inv(XtWX)
        se_b1 = float(np.sqrt(cov[1, 1]))
        z = b1 / se_b1
        p_b1 = float(2 * (1 - stats.norm.cdf(abs(z))))
        ci = [b1 - 1.96 * se_b1, b1 + 1.96 * se_b1]
    except np.linalg.LinAlgError:
        se_b1, p_b1 = float("nan"), float("nan")
        ci = [float("nan"), float("nan")]
    return {"n": int(len(y)), "n_pos": int(y.sum()),
            "b0": b0, "b1": b1, "se_b1": se_b1, "p_value": p_b1,
            "ci95_b1": [float(ci[0]), float(ci[1])],
            "mcfadden_r2": float(r2), "auc": auc}


def main():
    df = pd.read_csv(INPUT)
    # Outcome: identifiable column has 'Y'/'N' (per condition-camera)
    y_all = (df["identifiable"].astype(str).str.upper() == "Y").astype(int)
    out = {
        "input_rows": int(len(df)),
        "n_pos": int(y_all.sum()),
        "n_neg": int(len(y_all) - y_all.sum()),
        "by_camera": df.groupby("camera").size().to_dict(),
        "results": {},
    }
    metrics = ["C_M", "L_90", "mtf_h", "bew_h", "sigma_motion"]
    for m in metrics:
        sub = df[[m]].copy()
        sub["y"] = y_all.values
        sub = sub.dropna(subset=[m])
        r = fit_logit(sub[m].values, sub["y"].values)
        out["results"][m] = r
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[A1b] wrote {OUTPUT}")
    print(f"  outcome: identifiable Y/N (n_pos={out['n_pos']}/{len(df)})")
    for m in metrics:
        r = out["results"][m]
        if r:
            print(f"  {m:14s}  n={r['n']:3d}  AUC={r['auc']:.3f}  b1={r['b1']:+.4f}  p={r['p_value']:.3g}  R2={r['mcfadden_r2']:.3f}")


if __name__ == "__main__":
    main()
