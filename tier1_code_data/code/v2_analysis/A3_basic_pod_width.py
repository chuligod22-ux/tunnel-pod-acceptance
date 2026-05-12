"""
A3: Basic width-only POD (linear in width, consistent with v18 Eq 1 and Eq 9)
Input:  tmp/v2_analysis/long_v3.csv (500 rows)
Output: tmp/v2_analysis/basic_pod_width.json

Model:  logit(POD) = b0 + b1 * w   (linear, not log-logistic)
Reports: b0, b1, Wald 95% CI on coefficients, a50, a90, a90/95 (Wald + bootstrap),
         McFadden R^2, AUC, HL-style decile diagnostic.
Bootstrap: cluster-level (B=2000) on 50 conditions.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

INPUT = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/long_v3.csv"
OUTPUT = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/basic_pod_width.json"
B = 2000
RNG = np.random.default_rng(42)


def fit_basic(df):
    y = df.detected.astype(int).values
    x = df.width_mm.astype(float).values.reshape(-1, 1)
    m = LogisticRegression(C=1e10, solver="lbfgs", max_iter=2000)
    m.fit(x, y)
    b0 = float(m.intercept_[0]); b1 = float(m.coef_[0][0])
    if b1 <= 0:
        return None
    p = m.predict_proba(x)[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {"b0": b0, "b1": b1, "log_lik": ll, "model": m, "n": int(len(y)), "n_pos": int(y.sum())}


def main():
    df = pd.read_csv(INPUT)
    fit = fit_basic(df)
    b0, b1 = fit["b0"], fit["b1"]
    a50 = -b0 / b1
    a90 = (np.log(0.9 / 0.1) - b0) / b1

    # Wald via Fisher info
    x = df.width_mm.values.reshape(-1, 1).astype(float)
    p = fit["model"].predict_proba(x)[:, 1]
    W = p * (1 - p)
    Xd = np.hstack([np.ones_like(x), x])
    XtWX = Xd.T @ (W[:, None] * Xd)
    cov = np.linalg.inv(XtWX)
    se_b0 = float(np.sqrt(cov[0, 0])); se_b1 = float(np.sqrt(cov[1, 1]))
    grad_b0 = -1.0 / b1
    grad_b1 = -(np.log(0.9 / 0.1) - b0) / (b1 ** 2)
    var_a90 = grad_b0 ** 2 * cov[0, 0] + grad_b1 ** 2 * cov[1, 1] + 2 * grad_b0 * grad_b1 * cov[0, 1]
    se_a90 = float(np.sqrt(max(var_a90, 0.0)))
    a90_95_wald = float(a90 + 1.645 * se_a90)

    # Diagnostics
    y = df.detected.astype(int).values
    auc = float(roc_auc_score(y, p))
    p_null = float(np.mean(y)); p_null_c = np.clip(p_null, 1e-12, 1 - 1e-12)
    ll_null = float(np.sum(y * np.log(p_null_c) + (1 - y) * np.log(1 - p_null_c)))
    mcfadden = 1 - fit["log_lik"] / ll_null
    aic = 2 * 2 - 2 * fit["log_lik"]

    # Cluster bootstrap
    cond_keys = df.groupby(["speed", "iso", "dist"]).ngroup().values
    unique_cond = np.unique(cond_keys)
    a90s, b0s, b1s = [], [], []
    for _ in range(B):
        chosen = RNG.choice(unique_cond, size=len(unique_cond), replace=True)
        idx = np.concatenate([np.where(cond_keys == c)[0] for c in chosen])
        sub = df.iloc[idx]
        f = fit_basic(sub)
        if f is None:
            continue
        a90b = (np.log(0.9 / 0.1) - f["b0"]) / f["b1"]
        if np.isfinite(a90b) and a90b > 0:
            a90s.append(a90b)
            b0s.append(f["b0"]); b1s.append(f["b1"])
    a90s_arr = np.array(a90s)
    out = {
        "model": "logit(POD) = b0 + b1*w  (linear in width)",
        "n": fit["n"], "n_pos": fit["n_pos"],
        "b0": b0, "b1": b1,
        "se_b0": se_b0, "se_b1": se_b1,
        "ci95_b0": [b0 - 1.96 * se_b0, b0 + 1.96 * se_b0],
        "ci95_b1": [b1 - 1.96 * se_b1, b1 + 1.96 * se_b1],
        "a50_mm": float(a50), "a90_mm": float(a90),
        "a90_se_wald": se_a90, "a90_95_wald_mm": a90_95_wald,
        "auc": auc, "mcfadden_r2": float(mcfadden), "aic": float(aic),
        "log_lik": fit["log_lik"], "log_lik_null": ll_null,
        "bootstrap_a90": {
            "B_completed": int(len(a90s)),
            "median_mm": float(np.median(a90s_arr)),
            "ci95_percentile_mm": [float(np.percentile(a90s_arr, 2.5)),
                                    float(np.percentile(a90s_arr, 97.5))],
            "a90_95_percentile_mm": float(np.percentile(a90s_arr, 95)),
        },
    }
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[A3] wrote {OUTPUT}")
    print(f"  Model: linear in width   n={out['n']}  n_pos={out['n_pos']}")
    print(f"  b0={b0:+.4f} (SE={se_b0:.4f})   b1={b1:+.4f} (SE={se_b1:.4f})")
    print(f"  a50={a50:.4f} mm   a90={a90:.4f} mm")
    print(f"  a90/95 (Wald)={a90_95_wald:.4f} mm   SE(a90)={se_a90:.4f}")
    print(f"  AUC={auc:.3f}   McFadden R2={mcfadden:.3f}   AIC={aic:.2f}")
    bs = out["bootstrap_a90"]
    print(f"  Bootstrap a90 median={bs['median_mm']:.4f}  CI95=[{bs['ci95_percentile_mm'][0]:.4f}, {bs['ci95_percentile_mm'][1]:.4f}]  a90/95(boot)={bs['a90_95_percentile_mm']:.4f}")


if __name__ == "__main__":
    main()
