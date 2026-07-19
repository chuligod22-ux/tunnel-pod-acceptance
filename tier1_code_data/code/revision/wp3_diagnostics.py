#!/usr/bin/env python3
"""
wp3_diagnostics.py — Phase 4 리비전 WP3 통계 진단 (R1·R3 대응).

기존 M0/M1/M2 nested logistic (A5b, long_v3.csv σ_motion complete-case, n=470,
47 조건)을 재사용하여 리뷰어가 요구한 진단을 산출:

  WP3-1 다중공선성: Pearson 상관행렬 + VIF + 설계행렬 조건수(condition number)
  WP3-2 AUC 신뢰구간 + 낙관편향 보정: DeLong CI(독립가정) + cluster bootstrap CI
         + GroupKFold(조건 단위) 교차검증 AUC + optimism-corrected AUC
  WP3-3 M1 calibration: reliability curve + Brier + calibration intercept/slope
  WP3-4 비선형·상호작용: 후보 항 추가 LRT/ΔAIC

클러스터(조건=speed×iso×dist, 조건당 10 폭) 구조를 존중: 행 단위 CV는 누출되므로
조건 단위 GroupKFold / 조건 리샘플 cluster bootstrap 사용.

출력: wp3_diagnostics.json + viz/wp3_calibration.png + 콘솔 요약
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from sklearn.metrics import roc_auc_score, brier_score_loss
from statsmodels.stats.outliers_influence import variance_inflation_factor

HERE = Path(__file__).parent
SRC = HERE.parent.parent.parent / "tmp/v2_analysis/long_v3.csv"
if not SRC.exists():
    SRC = (HERE.parent.parent / "01_paper/output/ieee_tim_v2/supplementary"
           "/tier1_code_data/data/long_v3.csv")
OUT = HERE / "wp3_diagnostics.json"
FIG = HERE / "viz"

IQ4 = ["C_M", "L_90", "mtf_h", "bew_h"]
PRED_M1 = ["width_mm", *IQ4]
PRED_M2 = ["width_mm", *IQ4, "sigma_motion"]
SEED = 20260716  # 재현성(고정)


def load():
    df = pd.read_csv(SRC).dropna(subset=["sigma_motion"]).reset_index(drop=True)
    df["cond"] = list(zip(df["speed"], df["iso"], df["dist"]))
    return df


def fit(df, predictors):
    X = sm.add_constant(df[predictors].astype(float).values)
    y = df["detected"].astype(int).values
    res = sm.Logit(y, X).fit(disp=False, maxiter=200)
    return res, X, y


# ── WP3-1 다중공선성 ────────────────────────────────────────────
def multicollinearity(df, predictors):
    Xp = df[predictors].astype(float)
    corr = Xp.corr(method="pearson").round(3)
    # VIF (표준화 후 상수 포함)
    Z = (Xp - Xp.mean()) / Xp.std(ddof=0)
    Zc = sm.add_constant(Z.values)
    vif = {predictors[i]: round(float(variance_inflation_factor(Zc, i + 1)), 3)
           for i in range(len(predictors))}
    # 조건수: 표준화 설계행렬(상수 포함)의 특이값 비
    sv = np.linalg.svd(Zc, compute_uv=False)
    cond_number = float(sv.max() / sv.min())
    return {"pearson_corr": corr.to_dict(), "vif": vif,
            "condition_number": round(cond_number, 2),
            "note": "VIF>5(또는 10) 및 조건수>30이면 다중공선성 우려"}


# ── WP3-2 AUC CI + 교차검증 ─────────────────────────────────────
def delong_auc_var(y_true, y_score):
    """Fast DeLong: AUC와 분산 반환 (독립 가정)."""
    y_true = np.asarray(y_true)
    order = np.argsort(-y_score)
    y_score, y_true = y_score[order], y_true[order]
    pos = y_true == 1
    neg = ~pos
    m, n = pos.sum(), neg.sum()
    xp, xn = y_score[pos], y_score[neg]

    def midrank(x):
        s = np.argsort(x)
        xs = x[s]
        r = np.zeros(len(x))
        i = 0
        while i < len(x):
            j = i
            while j < len(x) and xs[j] == xs[i]:
                j += 1
            r[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        out = np.empty(len(x))
        out[s] = r
        return out

    tx = midrank(np.r_[xp, xn])
    tp_ = midrank(xp)
    tn_ = midrank(xn)
    auc = (tx[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tx[:m] - tp_) / n
    v10 = 1 - (tx[m:] - tn_) / m
    s01 = np.var(v01, ddof=1) / m
    s10 = np.var(v10, ddof=1) / n
    return float(auc), float(s01 + s10)


def delong_ci(y, p, alpha=0.05):
    auc, var = delong_auc_var(y, p)
    se = np.sqrt(var)
    z = sps.norm.ppf(1 - alpha / 2)
    lo, hi = auc - z * se, auc + z * se
    return {"auc": round(auc, 4), "se": round(se, 4),
            "ci95": [round(max(0, lo), 4), round(min(1, hi), 4)]}


def cluster_bootstrap_auc(df, predictors, B=2000):
    """조건 리샘플 cluster bootstrap AUC 분포."""
    rng = np.random.default_rng(SEED)
    conds = df["cond"].unique()
    aucs = []
    for _ in range(B):
        pick = rng.choice(len(conds), size=len(conds), replace=True)
        parts = [df[df["cond"] == conds[k]] for k in pick]
        bs = pd.concat(parts, ignore_index=True)
        if bs["detected"].nunique() < 2:
            continue
        try:
            res, X, y = fit(bs, predictors)
            aucs.append(roc_auc_score(y, res.predict(X)))
        except Exception:
            continue
    aucs = np.array(aucs)
    return {"B": len(aucs), "mean": round(float(aucs.mean()), 4),
            "ci95": [round(float(np.percentile(aucs, 2.5)), 4),
                     round(float(np.percentile(aucs, 97.5)), 4)]}


def oof_predictions(df, predictors, k=5):
    """조건 단위 GroupKFold out-of-fold 예측확률 (누출 방지)."""
    from sklearn.model_selection import GroupKFold
    groups = df["cond"].astype(str).values
    y_all = df["detected"].astype(int).values
    oof = np.full(len(df), np.nan)
    gkf = GroupKFold(n_splits=k)
    for tr, te in gkf.split(df, y_all, groups):
        res, _, _ = fit(df.iloc[tr], predictors)
        Xte = sm.add_constant(df.iloc[te][predictors].astype(float).values)
        oof[te] = res.predict(Xte)
    return oof


def group_kfold_auc(df, predictors, k=5):
    """조건 단위 GroupKFold 교차검증 AUC (out-of-sample, 누출 방지)."""
    oof = oof_predictions(df, predictors, k)
    return round(float(roc_auc_score(df["detected"].astype(int).values, oof)), 4)


def optimism_corrected_auc(df, predictors, B=500):
    """Harrell optimism 보정 (cluster bootstrap): corrected = apparent - optimism."""
    rng = np.random.default_rng(SEED + 1)
    res0, X0, y0 = fit(df, predictors)
    apparent = roc_auc_score(y0, res0.predict(X0))
    conds = df["cond"].unique()
    opt = []
    for _ in range(B):
        pick = rng.choice(len(conds), size=len(conds), replace=True)
        bs = pd.concat([df[df["cond"] == conds[k]] for k in pick], ignore_index=True)
        if bs["detected"].nunique() < 2:
            continue
        try:
            resb, Xb, yb = fit(bs, predictors)
            boot_auc = roc_auc_score(yb, resb.predict(Xb))
            Xorig = sm.add_constant(df[predictors].astype(float).values)
            orig_auc = roc_auc_score(y0, resb.predict(Xorig))
            opt.append(boot_auc - orig_auc)
        except Exception:
            continue
    optimism = float(np.mean(opt))
    return {"apparent_auc": round(float(apparent), 4),
            "optimism": round(optimism, 4),
            "corrected_auc": round(float(apparent - optimism), 4),
            "B": len(opt)}


# ── WP3-3 calibration ──────────────────────────────────────────
def calibration(df, predictors, nbins=10, label="M1", use_oof=True):
    """out-of-fold(기본) 예측으로 보정 진단. in-sample은 MLE라 slope=1로 무의미."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    y = df["detected"].astype(int).values
    if use_oof:
        p = np.asarray(oof_predictions(df, predictors, k=5))
    else:
        res, X, _ = fit(df, predictors)
        p = np.asarray(res.predict(X))
    brier = float(brier_score_loss(y, p))
    # Cox calibration: logit(y)~a+b*logit(p)
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    cal = sm.Logit(y, sm.add_constant(lp)).fit(disp=False)
    cal_int, cal_slope = float(cal.params[0]), float(cal.params[1])
    # binned reliability
    bins = np.quantile(p, np.linspace(0, 1, nbins + 1))
    bins[-1] += 1e-9
    idx = np.digitize(p, bins) - 1
    rel = []
    for b in range(nbins):
        m = idx == b
        if m.sum() > 0:
            rel.append({"pred_mean": round(float(p[m].mean()), 4),
                        "obs_freq": round(float(np.asarray(y)[m].mean()), 4),
                        "n": int(m.sum())})
    # it=0 필수: 기본 robustifying(it=3)은 이진 반응에서 붕괴(전부 1.0)
    lo = lowess(y.astype(float), p, frac=0.5, it=0, return_sorted=True)
    return {"label": label, "brier": round(brier, 4),
            "calibration_intercept": round(cal_int, 4),
            "calibration_slope": round(cal_slope, 4),
            "reliability_bins": rel,
            "lowess": [[round(float(a), 4), round(float(b), 4)] for a, b in lo[::max(1, len(lo)//60)]],
            "note": "완벽 보정: intercept≈0, slope≈1"}


# ── WP3-4 비선형·상호작용 ───────────────────────────────────────
def nonlinearity(df):
    """M1에 후보 비선형/상호작용 항 추가 → LRT/ΔAIC."""
    base = df.copy()
    base["width2"] = base["width_mm"] ** 2
    base["w_x_dist"] = base["width_mm"] * base["dist"]
    base["w_x_L90"] = base["width_mm"] * base["L_90"]
    base["dist_x_iso"] = base["dist"] * base["iso"]
    base["CM_x_L90"] = base["C_M"] * base["L_90"]
    m1, _, _ = fit(base, PRED_M1)
    tests = {}
    for name, extra in [("width^2", ["width2"]),
                        ("width×dist", ["w_x_dist"]),
                        ("width×L_90", ["w_x_L90"]),
                        ("dist×iso", ["dist_x_iso"]),
                        ("C_M×L_90", ["CM_x_L90"])]:
        try:
            mbig, _, _ = fit(base, PRED_M1 + extra)
            stat = 2 * (mbig.llf - m1.llf)
            p = float(sps.chi2.sf(stat, len(extra)))
            tests[name] = {"lrt_stat": round(float(stat), 3), "df": len(extra),
                           "p_value": round(p, 4),
                           "delta_aic": round(float(m1.aic - mbig.aic), 3),
                           "improves": bool(p < 0.05)}
        except Exception as e:
            tests[name] = {"error": str(e)}
    return {"m1_aic": round(float(m1.aic), 2), "candidate_terms": tests,
            "note": "p<0.05 & ΔAIC>2 이면 비선형/상호작용 유의 (없으면 선형-가법 적정)"}


def make_calib_fig(cal_m1):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    FIG.mkdir(exist_ok=True)
    rel = cal_m1["reliability_bins"]
    lo = np.array(cal_m1["lowess"])
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot([r["pred_mean"] for r in rel], [r["obs_freq"] for r in rel],
            "o-", color="#2166ac", label="decile bins")
    if lo.size:
        ax.plot(lo[:, 0], lo[:, 1], "-", color="#b2182b", alpha=0.8, label="lowess")
    ax.set_xlabel("Predicted probability (M1)")
    ax.set_ylabel("Observed detection frequency")
    ax.set_title(f"M1 calibration (Brier={cal_m1['brier']}, "
                 f"slope={cal_m1['calibration_slope']})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "wp3_calibration.png", dpi=140)
    print(f"figure: {FIG/'wp3_calibration.png'}")


def main():
    df = load()
    print(f"데이터: {len(df)} 행, {df['cond'].nunique()} 조건 "
          f"(detected {int(df['detected'].sum())}/{len(df)})\n")

    result = {"n_rows": int(len(df)), "n_conditions": int(df["cond"].nunique()),
              "source": str(SRC)}

    print("── WP3-1 다중공선성 ──")
    mc = multicollinearity(df, PRED_M2)
    result["multicollinearity"] = mc
    print("VIF:", mc["vif"])
    print("조건수(condition number):", mc["condition_number"])

    print("\n── WP3-2 AUC CI + 교차검증 ──")
    aucdiag = {}
    for label, preds in [("M1", PRED_M1), ("M2", PRED_M2)]:
        res, X, y = fit(df, preds)
        p = np.asarray(res.predict(X))
        d = {"delong": delong_ci(y, p),
             "cluster_bootstrap": cluster_bootstrap_auc(df, preds, B=2000),
             "groupkfold_cv_auc": group_kfold_auc(df, preds, k=5),
             "optimism": optimism_corrected_auc(df, preds, B=500)}
        aucdiag[label] = d
        print(f"  {label}: apparent={d['delong']['auc']} "
              f"DeLong CI={d['delong']['ci95']} | clusterBS CI={d['cluster_bootstrap']['ci95']} "
              f"| CV(cond)={d['groupkfold_cv_auc']} | optimism-corrected={d['optimism']['corrected_auc']}")
    result["auc_diagnostics"] = aucdiag

    print("\n── WP3-3 M1 calibration ──")
    cal = calibration(df, PRED_M1, label="M1")
    result["calibration_M1"] = cal
    print(f"  Brier={cal['brier']}  intercept={cal['calibration_intercept']}  "
          f"slope={cal['calibration_slope']}")

    print("\n── WP3-4 비선형·상호작용 ──")
    nl = nonlinearity(df)
    result["nonlinearity"] = nl
    for name, t in nl["candidate_terms"].items():
        if "p_value" in t:
            print(f"  {name}: LRT p={t['p_value']} ΔAIC={t['delta_aic']} "
                  f"{'개선' if t['improves'] else '무의미'}")

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")
    make_calib_fig(cal)


if __name__ == "__main__":
    main()
