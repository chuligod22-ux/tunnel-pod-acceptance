"""
A5: Extended logistic POD (multivariable)
Input:  tmp/b2_out/long_v2.csv
Output: tmp/v2_analysis/extended_logistic_pod.json

Models:
  M0: logit(P) = b0 + b1*width
  M1: logit(P) = b0 + b1*width + b2*mtf_h + b3*C_M + b4*L_90

Reports:
  - coefficients (Wald 95% CI, p-values)
  - log-lik, AIC, McFadden R^2
  - LRT M0 vs M1 (chi^2 df=3)
  - AUC
  - inverse: minimum mtf_h / C_M / L_90 to achieve a90/95 <= target
    (numerical inversion at fixed width=0.3 mm)
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

INPUT = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/long_v3.csv"
OUTPUT = "/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/extended_logistic_pod.json"


def fit_multilogit(X_cols, y, df):
    X = df[X_cols].astype(float).values
    y = np.asarray(y, dtype=int)
    m = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
    m.fit(X, y)
    coefs = np.concatenate([m.intercept_, m.coef_[0]])
    p = m.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    k = len(coefs)
    aic = 2 * k - 2 * ll
    p_null = float(np.mean(y))
    p_null_c = np.clip(p_null, 1e-12, 1 - 1e-12)
    ll_null = float(np.sum(y * np.log(p_null_c) + (1 - y) * np.log(1 - p_null_c)))
    mcfadden = 1 - ll / ll_null
    auc = float(roc_auc_score(y, p))

    # Wald SEs
    Xd = np.hstack([np.ones((len(X), 1)), X])
    p_pred = m.predict_proba(X)[:, 1]
    W = p_pred * (1 - p_pred)
    XtWX = Xd.T @ (W[:, None] * Xd)
    try:
        cov = np.linalg.inv(XtWX)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full_like(coefs, np.nan)

    out = {
        "n": int(len(y)),
        "coef": {},
        "log_lik": ll, "aic": float(aic), "mcfadden_r2": float(mcfadden), "auc": auc,
        "n_params": int(k),
    }
    names = ["intercept"] + list(X_cols)
    for nm, c, s in zip(names, coefs, se):
        z = float(c / s) if s and not np.isnan(s) else float("nan")
        pv = float(2 * (1 - stats.norm.cdf(abs(z)))) if not np.isnan(z) else float("nan")
        ci = [float(c - 1.96 * s), float(c + 1.96 * s)] if not np.isnan(s) else [float("nan"), float("nan")]
        out["coef"][nm] = {"value": float(c), "se": float(s) if not np.isnan(s) else None,
                            "z": z, "p_value": pv, "ci95": ci}
    return out, m


def a90_at_iq(model, X_cols, iq_values, width_grid):
    """Given fixed IQ values, scan width_grid for first p>=0.9."""
    rec = {**iq_values, "width_mm": np.nan}
    pods = []
    for w in width_grid:
        x = []
        for c in X_cols:
            x.append(w if c == "width_mm" else iq_values[c])
        prob = float(model.predict_proba(np.array([x]))[0][1])
        pods.append(prob)
    pods = np.asarray(pods)
    above = np.where(pods >= 0.9)[0]
    a90 = float(width_grid[above[0]]) if len(above) else float("inf")
    return a90, pods.tolist()


def main():
    df_full = pd.read_csv(INPUT)
    # complete-case for 5-predictor extended model (drops only sigma_motion 30 NaN
    # condition-rows; bew_h, mtf_h, C_M, L_90 all 0 NaN in long_v3)
    df = df_full.dropna(subset=["width_mm", "mtf_h", "bew_h", "C_M", "L_90", "detected"]).reset_index(drop=True)
    y = df["detected"].astype(int).values

    # M0: width only
    m0, _ = fit_multilogit(["width_mm"], y, df)
    # M1: 4-predictor extended (user spec Sec 3.5: w + MTF + C_M + L_90)
    m1, model1 = fit_multilogit(["width_mm", "mtf_h", "C_M", "L_90"], y, df)
    # M2: 5-predictor full (5-metric IQ suite + width)
    m2, model2 = fit_multilogit(["width_mm", "mtf_h", "bew_h", "C_M", "L_90"], y, df)

    # LRTs
    def lrt(ma, mb):
        s = 2 * (mb["log_lik"] - ma["log_lik"])
        d = mb["n_params"] - ma["n_params"]
        return {"statistic": float(s), "df": int(d),
                "p_value": float(1 - stats.chi2.cdf(s, d)),
                "delta_AIC": float(ma["aic"] - mb["aic"])}

    # Multicollinearity: condition number of design matrix for M2
    X2 = df[["width_mm", "mtf_h", "bew_h", "C_M", "L_90"]].astype(float).values
    Xs = (X2 - X2.mean(axis=0)) / X2.std(axis=0)
    sv = np.linalg.svd(Xs, compute_uv=False)
    cond_number = float(sv.max() / sv.min())
    # Pearson correlations
    corr = pd.DataFrame(X2, columns=["width_mm", "mtf_h", "bew_h", "C_M", "L_90"]).corr().to_dict()

    # Inverse a90 at IQ quartiles (use M2 for 5-metric)
    iq_levels = {}
    for col in ["mtf_h", "bew_h", "C_M", "L_90"]:
        iq_levels[col] = {
            "p25": float(np.percentile(df[col], 25)),
            "p50": float(np.percentile(df[col], 50)),
            "p75": float(np.percentile(df[col], 75)),
        }
    width_grid = np.linspace(0.05, 2.0, 196)
    a90_at = {}
    for tag in ["p25", "p50", "p75"]:
        iqv = {col: iq_levels[col][tag] for col in iq_levels}
        a90, _ = a90_at_iq(model2, ["width_mm", "mtf_h", "bew_h", "C_M", "L_90"], iqv, width_grid)
        a90_at[tag] = {**iqv, "a90_predicted": a90}

    out = {
        "input_rows_full": int(len(df_full)),
        "n_complete_case": int(len(df)),
        "n_conditions_complete_case": int(df.drop_duplicates(["speed","iso","dist"]).shape[0]),
        "model0_basic": m0,
        "model1_extended_4pred": m1,
        "model2_full_5pred": m2,
        "LRT_M0_vs_M1": lrt(m0, m1),
        "LRT_M0_vs_M2": lrt(m0, m2),
        "LRT_M1_vs_M2": lrt(m1, m2),
        "M2_design_diagnostics": {
            "condition_number_standardised": cond_number,
            "pearson_correlations": corr,
        },
        "iq_levels_summary": iq_levels,
        "a90_at_iq_quartiles_M2": a90_at,
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[A5] wrote {OUTPUT}")
    print(f"  n complete-case: {len(df)} ({len(df)//10} conditions)")
    print(f"  M0 (w):       AIC={m0['aic']:.2f}  R2={m0['mcfadden_r2']:.3f}  AUC={m0['auc']:.3f}")
    print(f"  M1 (w+MCL):   AIC={m1['aic']:.2f}  R2={m1['mcfadden_r2']:.3f}  AUC={m1['auc']:.3f}")
    print(f"  M2 (w+MBCL):  AIC={m2['aic']:.2f}  R2={m2['mcfadden_r2']:.3f}  AUC={m2['auc']:.3f}")
    print(f"  LRT M0-M1: dAIC={out['LRT_M0_vs_M1']['delta_AIC']:+.2f}  p={out['LRT_M0_vs_M1']['p_value']:.3g}")
    print(f"  LRT M0-M2: dAIC={out['LRT_M0_vs_M2']['delta_AIC']:+.2f}  p={out['LRT_M0_vs_M2']['p_value']:.3g}")
    print(f"  LRT M1-M2: dAIC={out['LRT_M1_vs_M2']['delta_AIC']:+.2f}  p={out['LRT_M1_vs_M2']['p_value']:.3g}")
    print(f"  M2 design condition number (standardised): {cond_number:.2f}")
    print("  M2 coefficients:")
    for nm, c in m2["coef"].items():
        print(f"    {nm:14s}  beta={c['value']:+.4f}  p={c['p_value']:.3g}  ci={c['ci95']}")
    print("  M2 a90 at IQ quartiles (width that achieves p>=0.9):")
    for tag, r in a90_at.items():
        print(f"    {tag}: a90={r['a90_predicted']}")


if __name__ == "__main__":
    main()
