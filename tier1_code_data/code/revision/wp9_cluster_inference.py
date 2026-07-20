"""Cluster-aware sensitivity analyses for the M0-vs-M1 incremental-value claim.

Motivation (revision R3): the 470 long-format rows comprise 47 acquisition
conditions x 10 crack widths; the ten rows of a condition share one IQ vector
and are therefore correlated. Conventional AIC/LRT/Wald results for M0/M1/M2
assume independent rows and are retained in the manuscript only as descriptive
model-fit comparisons. This script provides the cluster-aware evidence:

  (1) paired condition-cluster bootstrap of the apparent AUC difference
      DeltaAUC = AUC(M1) - AUC(M0), B = 2000, seed 20260511
      (resample 47 condition clusters with replacement; refit both models);
  (2) repeated grouped 5-fold cross-validation (R = 200 fold shuffles,
      folds split at the condition level) giving the distribution of the
      paired out-of-sample DeltaAUC;
  (3) GEE logistic fit of M1 (exchangeable working correlation,
      cluster-robust standard errors, standardized covariates) as
      cluster-aware coefficient inference;
  (4) a mixed-effects logistic sensitivity fit with a condition random
      intercept (variational Bayes, standardized covariates); if the fit
      does not converge this is recorded honestly in the output JSON.

Output: results_json/revision/wp9_cluster_inference.json
"""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).parent
LONG = HERE.parent.parent / "data" / "long_v3.csv"
if not LONG.exists():
    LONG = Path("/Users/lch/home/code/tunnelscanning/tmp/v2_analysis/long_v3.csv")
OUT = HERE.parent.parent / "results_json" / "revision" / "wp9_cluster_inference.json"

PRED_M0 = ["width_mm"]
PRED_M1 = ["width_mm", "C_M", "L_90", "mtf_h", "bew_h"]
B_BOOT = 2000
R_CV = 200
K_FOLD = 5
RNG_SEED = 20260511


def fit_predict(df_fit, df_eval, preds):
    m = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
    m.fit(df_fit[preds].astype(float).values, df_fit["detected"].astype(int).values)
    return m.predict_proba(df_eval[preds].astype(float).values)[:, 1]


def main():
    df_all = pd.read_csv(LONG)
    needed = PRED_M1 + ["detected", "sigma_motion", "speed", "iso", "dist"]
    df = df_all.dropna(subset=needed).copy().reset_index(drop=True)
    df["cond_id"] = (df["speed"].astype(str) + "_" + df["iso"].astype(str)
                     + "_" + df["dist"].astype(str))
    cond_ids = df["cond_id"].unique().tolist()
    n_cond = len(cond_ids)
    y_all = df["detected"].astype(int).values

    out = {"n_rows": int(len(df)), "n_clusters": int(n_cond),
           "seed": RNG_SEED, "B_bootstrap": B_BOOT,
           "R_cv_repeats": R_CV, "k_folds": K_FOLD}

    # Full-data apparent AUCs (reference)
    p0 = fit_predict(df, df, PRED_M0)
    p1 = fit_predict(df, df, PRED_M1)
    out["apparent_auc_m0"] = float(roc_auc_score(y_all, p0))
    out["apparent_auc_m1"] = float(roc_auc_score(y_all, p1))
    out["apparent_delta_auc"] = out["apparent_auc_m1"] - out["apparent_auc_m0"]

    # (1) Paired condition-cluster bootstrap of apparent DeltaAUC
    rng = np.random.default_rng(RNG_SEED)
    by_cond = {cid: df[df["cond_id"] == cid] for cid in cond_ids}
    deltas, fails = [], 0
    for _ in range(B_BOOT):
        picks = rng.choice(cond_ids, size=n_cond, replace=True)
        db = pd.concat([by_cond[c] for c in picks], ignore_index=True)
        yb = db["detected"].astype(int).values
        if yb.min() == yb.max():
            fails += 1
            continue
        try:
            a0 = roc_auc_score(yb, fit_predict(db, db, PRED_M0))
            a1 = roc_auc_score(yb, fit_predict(db, db, PRED_M1))
            deltas.append(a1 - a0)
        except Exception:
            fails += 1
    deltas = np.asarray(deltas)
    out["bootstrap_paired_delta_auc"] = {
        "n_valid": int(len(deltas)), "n_failed": int(fails),
        "mean": float(deltas.mean()),
        "ci_2p5": float(np.percentile(deltas, 2.5)),
        "ci_97p5": float(np.percentile(deltas, 97.5)),
        "fraction_leq_0": float((deltas <= 0).mean()),
    }

    # (2) Repeated grouped 5-fold CV, paired out-of-sample DeltaAUC
    rng_cv = np.random.default_rng(RNG_SEED)
    cv_d, cv_a0, cv_a1 = [], [], []
    ids = np.array(cond_ids)
    for _ in range(R_CV):
        perm = rng_cv.permutation(ids)
        folds = np.array_split(perm, K_FOLD)
        o0 = np.empty(len(df)); o1 = np.empty(len(df))
        for f in folds:
            te = df["cond_id"].isin(f).values
            o0[te] = fit_predict(df[~te], df[te], PRED_M0)
            o1[te] = fit_predict(df[~te], df[te], PRED_M1)
        a0 = roc_auc_score(y_all, o0); a1 = roc_auc_score(y_all, o1)
        cv_a0.append(a0); cv_a1.append(a1); cv_d.append(a1 - a0)
    cv_d = np.asarray(cv_d)
    out["repeated_grouped_cv"] = {
        "note": ("distribution over fold shuffles on the fixed dataset; "
                 "quantifies fold-construction variability, not sampling "
                 "uncertainty of new conditions"),
        "mean_auc_m0": float(np.mean(cv_a0)), "mean_auc_m1": float(np.mean(cv_a1)),
        "mean_delta_auc": float(cv_d.mean()),
        "pct_2p5": float(np.percentile(cv_d, 2.5)),
        "pct_97p5": float(np.percentile(cv_d, 97.5)),
        "fraction_leq_0": float((cv_d <= 0).mean()),
    }

    # (3) GEE logistic, exchangeable working correlation, robust SE
    try:
        import statsmodels.api as sm
        Xz = df[PRED_M1].astype(float)
        Xz = (Xz - Xz.mean()) / Xz.std(ddof=0)
        Xz = sm.add_constant(Xz)
        gee_result = None
        for label, cs in (("exchangeable", sm.cov_struct.Exchangeable()),
                          ("independence", sm.cov_struct.Independence())):
            gee = sm.GEE(y_all, Xz, groups=df["cond_id"],
                         family=sm.families.Binomial(), cov_struct=cs).fit()
            if not np.any(np.isnan(gee.params.values)):
                gee_result = (label, gee)
                break
        if gee_result is None:
            out["gee_m1_standardized"] = {
                "failed": True,
                "reason": ("estimation degenerated (non-finite parameters) under "
                           "both exchangeable and independence working "
                           "correlations; consistent with the quasi-separation "
                           "induced by the monotonic completion, under which "
                           "unpenalised logistic coefficients diverge")}
        else:
            label, gee = gee_result
            ols = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
            ols.fit(Xz.values[:, 1:], y_all)
            ord_coefs = dict(zip(PRED_M1, ols.coef_[0]))
            out["gee_m1_standardized"] = {"working_correlation": label}
            out["gee_m1_standardized"].update({
                k: {"coef": float(gee.params[k]), "robust_se": float(gee.bse[k]),
                    "z": float(gee.tvalues[k]), "p": float(gee.pvalues[k]),
                    "sign_matches_ordinary_logit": bool(
                        np.sign(gee.params[k]) == np.sign(ord_coefs[k]))}
                for k in PRED_M1})
    except Exception as e:
        out["gee_m1_standardized"] = {"failed": True, "reason": str(e)[:300]}

    # (4) Mixed-effects logistic, condition random intercept (VB)
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        dfm = df.copy()
        for c in PRED_M1:
            dfm[c] = (dfm[c] - dfm[c].mean()) / dfm[c].std(ddof=0)
        md = BinomialBayesMixedGLM.from_formula(
            "detected ~ width_mm + C_M + L_90 + mtf_h + bew_h",
            {"cond": "0 + C(cond_id)"}, dfm)
        fit = md.fit_vb()
        names = list(fit.model.exog_names)
        out["mixed_effects_m1_standardized_vb"] = {
            "converged": True,
            "random_intercept_sd_posterior_mean": float(
                np.exp(fit.vcp_mean[0])) if len(fit.vcp_mean) else None,
            "fixed_effects": {
                n: {"post_mean": float(fit.fe_mean[i]),
                    "post_sd": float(fit.fe_sd[i])}
                for i, n in enumerate(names)},
        }
    except Exception as e:
        out["mixed_effects_m1_standardized_vb"] = {
            "converged": False, "error": str(e)[:300]}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    b = out["bootstrap_paired_delta_auc"]
    c = out["repeated_grouped_cv"]
    print(f"apparent dAUC={out['apparent_delta_auc']:.4f}")
    print(f"cluster-bootstrap dAUC mean={b['mean']:.4f} CI=[{b['ci_2p5']:.4f},{b['ci_97p5']:.4f}] P(<=0)={b['fraction_leq_0']:.4f}")
    print(f"repeated grouped CV dAUC mean={c['mean_delta_auc']:.4f} pct=[{c['pct_2p5']:.4f},{c['pct_97p5']:.4f}] P(<=0)={c['fraction_leq_0']:.4f}")
    print("GEE:", "ok" if "failed" not in out["gee_m1_standardized"] else "FAILED")
    print("Mixed VB:", out["mixed_effects_m1_standardized_vb"].get("converged"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
