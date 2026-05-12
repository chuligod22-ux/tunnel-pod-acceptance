"""A5b — Nested logistic POD with σ_motion included.

Refits the M0/M1/M2 hierarchy on the σ_motion-complete subset of long_v3.csv:

  M0: width-only                        (intercept + w)            [2 params]
  M1: width + 4 IQ (no σ_motion)        + C_M + L_90 + MTF50 + BEW_H        [6 params]
  M2: M1 + σ_motion                     full 5-IQ specification             [7 params]

All three models are fitted on the same complete-case sample so AIC/LRT comparisons
are valid. Reports coefficients, Wald CIs, AIC, McFadden R², AUC, LRT p-values,
and a90 / a90/95 from the calibrated linear-in-w POD (M0).

Output: A5b_extended_logistic_with_sigma.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score


def fit_logistic(df: pd.DataFrame, predictors: list[str]) -> dict:
    X = sm.add_constant(df[predictors].astype(float).values)
    y = df["detected"].astype(int).values
    model = sm.Logit(y, X)
    res = model.fit(disp=False, maxiter=200)

    names = ["intercept", *predictors]
    coefs = res.params
    ses = res.bse
    z = coefs / ses
    p = 2 * (1 - sm.stats.stattools.stats.norm.cdf(np.abs(z))) if False else res.pvalues
    ci = res.conf_int()

    coef_dict = {}
    for i, n in enumerate(names):
        coef_dict[n] = {
            "value": float(coefs[i]),
            "se": float(ses[i]),
            "z": float(z[i]),
            "p_value": float(p[i]),
            "ci95": [float(ci[i, 0]), float(ci[i, 1])],
        }

    proba = res.predict(X)
    auc = float(roc_auc_score(y, proba))

    # McFadden pseudo-R² with respect to the intercept-only model fitted on the same y.
    null_model = sm.Logit(y, np.ones((len(y), 1)))
    null_res = null_model.fit(disp=False, maxiter=200)
    mcfadden = float(1.0 - res.llf / null_res.llf)

    return {
        "n": int(len(y)),
        "predictors": predictors,
        "coef": coef_dict,
        "log_lik": float(res.llf),
        "aic": float(res.aic),
        "mcfadden_r2": mcfadden,
        "auc": auc,
        "n_params": int(len(names)),
    }


def likelihood_ratio_test(m_small: dict, m_big: dict) -> dict:
    """Standard LRT for nested logistic models fitted on the same sample."""
    df_diff = m_big["n_params"] - m_small["n_params"]
    stat = 2.0 * (m_big["log_lik"] - m_small["log_lik"])
    from scipy import stats as sps

    p = float(sps.chi2.sf(stat, df_diff))
    return {
        "df": int(df_diff),
        "stat": float(stat),
        "p_value": p,
        "delta_aic": float(m_small["aic"] - m_big["aic"]),
    }


def width_only_pod_summary(m0: dict) -> dict:
    """a50 / a90 / a90/95 (Wald via delta method) from the linear-in-w POD."""
    from scipy import stats as sps

    b0 = m0["coef"]["intercept"]["value"]
    b1 = m0["coef"]["width_mm"]["value"]
    se_b0 = m0["coef"]["intercept"]["se"]
    se_b1 = m0["coef"]["width_mm"]["se"]
    a50 = -b0 / b1
    logit_90 = float(np.log(0.9 / 0.1))
    a90 = (logit_90 - b0) / b1
    # Delta method on a90 = (logit(p) - b0) / b1
    var_a90 = (1.0 / b1) ** 2 * se_b0 ** 2 + ((logit_90 - b0) / b1 ** 2) ** 2 * se_b1 ** 2
    se_a90 = float(np.sqrt(var_a90))
    a9095 = float(a90 + 1.645 * se_a90)
    return {
        "a50_mm": float(a50),
        "a90_mm": float(a90),
        "se_a90_mm": se_a90,
        "a90_95_wald_mm": a9095,
    }


def main() -> None:
    src = Path(__file__).resolve().parent / "long_v3.csv"
    df = pd.read_csv(src)

    # Complete-case on σ_motion (drops 3 conditions × 10 widths = 30 rows).
    df_full = df.copy()
    df_cc = df.dropna(subset=["sigma_motion"]).reset_index(drop=True)

    n_full_cond = df_full[["speed", "iso", "dist"]].drop_duplicates().shape[0]
    n_cc_cond = df_cc[["speed", "iso", "dist"]].drop_duplicates().shape[0]

    # Fit all three models on the SAME complete-case sample for valid AIC/LRT comparisons.
    m0 = fit_logistic(df_cc, ["width_mm"])
    m1 = fit_logistic(df_cc, ["width_mm", "C_M", "L_90", "mtf_h", "bew_h"])
    m2 = fit_logistic(df_cc, ["width_mm", "C_M", "L_90", "mtf_h", "bew_h", "sigma_motion"])

    lrt_m0_m1 = likelihood_ratio_test(m0, m1)
    lrt_m1_m2 = likelihood_ratio_test(m1, m2)
    lrt_m0_m2 = likelihood_ratio_test(m0, m2)

    pod = width_only_pod_summary(m0)

    out = {
        "input_rows_full": int(len(df_full)),
        "input_rows_complete_case": int(len(df_cc)),
        "n_conditions_full": int(n_full_cond),
        "n_conditions_complete_case": int(n_cc_cond),
        "sample_note": (
            "Complete-case on σ_motion: 3 conditions (30 rows) "
            "where BEW_H ≤ BEW_V on both cameras are excluded so that "
            "M0, M1, M2 share an identical sample for valid LRT/AIC comparisons."
        ),
        "model0_basic": m0,
        "model1_4iq_no_sigma": m1,
        "model2_5iq_with_sigma": m2,
        "LRT_M0_vs_M1": lrt_m0_m1,
        "LRT_M1_vs_M2": lrt_m1_m2,
        "LRT_M0_vs_M2": lrt_m0_m2,
        "M0_pod_summary": pod,
    }

    out_path = Path(__file__).resolve().parent / "A5b_extended_logistic_with_sigma.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote: {out_path}")

    # Brief console summary
    print()
    print(f"Complete-case n = {len(df_cc)} rows ({n_cc_cond} conditions)")
    print()
    print(
        f"{'Model':<10} {'n_params':>8} {'log-lik':>10} {'AIC':>10} "
        f"{'McFadden R²':>13} {'AUC':>8}"
    )
    for name, m in [("M0", m0), ("M1", m1), ("M2", m2)]:
        print(
            f"{name:<10} {m['n_params']:>8} {m['log_lik']:>10.2f} {m['aic']:>10.2f} "
            f"{m['mcfadden_r2']:>13.4f} {m['auc']:>8.4f}"
        )
    print()
    print(f"LRT M0→M1: ΔAIC = {lrt_m0_m1['delta_aic']:.2f}, p = {lrt_m0_m1['p_value']:.3e}")
    print(f"LRT M1→M2: ΔAIC = {lrt_m1_m2['delta_aic']:.2f}, p = {lrt_m1_m2['p_value']:.3e}")
    print(f"LRT M0→M2: ΔAIC = {lrt_m0_m2['delta_aic']:.2f}, p = {lrt_m0_m2['p_value']:.3e}")
    print()
    print(f"M0 width-only POD: a50={pod['a50_mm']:.3f}, a90={pod['a90_mm']:.3f}, "
          f"a90/95 (Wald)={pod['a90_95_wald_mm']:.3f} mm")


if __name__ == "__main__":
    main()
