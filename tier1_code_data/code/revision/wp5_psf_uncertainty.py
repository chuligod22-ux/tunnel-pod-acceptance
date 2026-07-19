#!/usr/bin/env python3
"""
wp5_psf_uncertainty.py — Phase 4 리비전 WP5: 등방 PSF 검증 + 측정 불확도 (R1·R3).

리뷰어 요구:
  WP5-1 측정 반복성·불확도: MTF/BEW 조건내 프레임간 반복성(CV%), 절대 불확도.
  WP5-2 등방 PSF 검증: BEW_V(광학-only, cross-track) 조건별 변동 정량화,
        속도/거리/ISO 의존성 분해(ANOVA), hv_ratio 속도의존, BEW_V-검출 상관.
        isotropic-optics 가정(σ_motion² = BEW_H² − BEW_V²의 전제) 방어 근거.
  WP5-3 POD 민감도: IQ 측정오차(측정 CV%)에 대한 M1 AUC/계수 Monte-Carlo 안정성.

데이터: cam2_analysis.csv(조건별 mean+std+n), long_v3.csv(모델링, n=470).
출력: wp5_psf_uncertainty.json + viz/wp5_*.png + 콘솔.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "03_src/data"
CAM2 = DATA / "cam2_analysis.csv"
LONG = HERE.parent.parent.parent / "tmp/v2_analysis/long_v3.csv"
if not LONG.exists():
    LONG = (HERE.parent.parent / "01_paper/output/ieee_tim_v2/supplementary"
            "/tier1_code_data/data/long_v3.csv")
OUT = HERE / "wp5_psf_uncertainty.json"
FIG = HERE / "viz"
SEED = 20260716

PRED_M1 = ["width_mm", "C_M", "L_90", "mtf_h", "bew_h"]


# ── WP5-1 측정 반복성·불확도 ────────────────────────────────────
def repeatability(cam2):
    """조건내 프레임간(≈24) 반복성: 절대 std + 상대 CV%."""
    out = {}
    for m in ["mtf_h", "mtf_v", "bew_h", "bew_v"]:
        mean_col, std_col = f"{m}_mean", f"{m}_std"
        if std_col not in cam2:
            continue
        cv = (cam2[std_col] / cam2[mean_col] * 100).replace([np.inf, -np.inf], np.nan).dropna()
        out[m] = {
            "grand_mean": round(float(cam2[mean_col].mean()), 4),
            "typical_abs_std": round(float(cam2[std_col].mean()), 4),
            "repeatability_cv_pct": round(float(cv.mean()), 2),
            "n_frames_typical": int(cam2.get(f"{m}_n", pd.Series([24])).median()),
        }
    return out


# ── WP5-2 등방 PSF 검증 ─────────────────────────────────────────
def isotropic_psf(cam2, long_df):
    res = {}
    bv = cam2["bew_v_mean"].dropna()
    bh = cam2["bew_h_mean"].dropna()
    # BEW_V 조건간 변동
    res["bew_v_across_conditions"] = {
        "mean": round(float(bv.mean()), 3), "std": round(float(bv.std()), 3),
        "cv_pct": round(float(bv.std() / bv.mean() * 100), 1),
        "range": [round(float(bv.min()), 2), round(float(bv.max()), 2)],
        "note": "조건간 CV — 조건내 반복성(WP5-1)과 비교: 초과분이 실제 광학변동",
    }
    # BEW_V ~ dist + iso + speed (광학이면 거리(초점)의존, 속도 무관해야 함)
    c = cam2.copy()
    c.columns = [x.strip() for x in c.columns]
    m = smf.ols("bew_v_mean ~ dist + iso + speed", data=c).fit()
    res["bew_v_anova"] = {
        "r2": round(float(m.rsquared), 3),
        "coef": {k: round(float(v), 5) for k, v in m.params.items()},
        "p_values": {k: round(float(v), 4) for k, v in m.pvalues.items()},
        "note": "광학기원이면 dist(초점/GSD) 유의, speed 무의미 기대",
    }
    # hv_ratio(BEW_H/BEW_V) 속도 의존 — 모션은 H에만 추가 → 속도↑시 비율↑
    c["hv_bew"] = c["bew_h_mean"] / c["bew_v_mean"]
    hv_by_speed = c.groupby("speed")["hv_bew"].mean().to_dict()
    m2 = smf.ols("hv_bew ~ speed + dist", data=c).fit()
    res["hv_ratio_vs_speed"] = {
        "mean_by_speed": {int(k): round(float(v), 3) for k, v in hv_by_speed.items()},
        "speed_coef": round(float(m2.params.get("speed", np.nan)), 5),
        "speed_p": round(float(m2.pvalues.get("speed", np.nan)), 4),
        "note": "isotropic optics + along-track motion이면 hv_ratio가 속도와 함께 증가",
    }
    # MTF isotropy: mtf_v(cross, 광학) vs mtf_h(along, 광학+모션)
    res["mtf_isotropy"] = {
        "mtf_v_mean": round(float(cam2["mtf_v_mean"].mean()), 4),
        "mtf_h_mean": round(float(cam2["mtf_h_mean"].mean()), 4),
        "v_sharper_frac": round(float((cam2["mtf_v_mean"] > cam2["mtf_h_mean"]).mean()), 3),
        "note": "V가 대체로 sharper(mtf_v>mtf_h)면 모션이 H에만 작용 = 분해 타당",
    }
    # BEW_V ↔ 검출 상관 (조건 단위 검출률 vs BEW_V)
    det = long_df.groupby(["speed", "iso", "dist"])["detected"].mean().reset_index()
    det.columns = ["speed", "iso", "dist", "det_rate"]
    merged = det.merge(cam2[["speed", "iso", "dist", "bew_v_mean", "bew_h_mean"]],
                       on=["speed", "iso", "dist"], how="inner")
    if len(merged) > 3:
        r_bv = float(np.corrcoef(merged["bew_v_mean"], merged["det_rate"])[0, 1])
        r_bh = float(np.corrcoef(merged["bew_h_mean"], merged["det_rate"])[0, 1])
        res["bew_detection_corr"] = {
            "pearson_bew_v_vs_detrate": round(r_bv, 3),
            "pearson_bew_h_vs_detrate": round(r_bh, 3),
            "n_conditions": int(len(merged)),
            "note": "광학-only BEW_V의 검출 상관 (약하면 검출은 노출/모션 주도)",
        }
    return res, merged


# ── WP5-3 IQ 측정오차에 대한 POD 민감도 ─────────────────────────
def pod_sensitivity(long_df, rep, n_mc=500):
    """
    IQ 측정오차에 대한 M1 refit 민감도.
    측정오차는 조건 단위(IQ는 조건당 1회 측정, 10 폭 행에 동일) — 조건별 1회 노이즈.
    """
    rng = np.random.default_rng(SEED)
    df = long_df.dropna(subset=["sigma_motion"]).reset_index(drop=True)
    df["cond"] = list(zip(df["speed"], df["iso"], df["dist"]))
    conds = df["cond"].values
    uniq = df["cond"].unique()
    cond_idx = {c: i for i, c in enumerate(uniq)}
    row_cond = np.array([cond_idx[c] for c in conds])
    y = df["detected"].astype(int).values
    # 각 IQ 측정 CV% (없으면 노출계열은 nominal)
    cvmap = {"mtf_h": rep.get("mtf_h", {}).get("repeatability_cv_pct", 10) / 100,
             "bew_h": rep.get("bew_h", {}).get("repeatability_cv_pct", 12) / 100,
             "C_M": 0.10, "L_90": 0.05, "width_mm": 0.0}  # width는 GT, 무오차

    def fit_auc(d):
        X = sm.add_constant(d[PRED_M1].astype(float).values)
        r = sm.Logit(y, X).fit(disp=False, maxiter=200)
        return roc_auc_score(y, r.predict(X)), r.params

    base_auc, base_params = fit_auc(df)
    aucs, coefs = [], []
    for _ in range(n_mc):
        d = df.copy()
        for col, cv in cvmap.items():
            if cv > 0:
                # 조건별 1회 노이즈 → 같은 조건의 모든 폭 행에 동일 적용
                fac = 1 + rng.normal(0, cv, len(uniq))[row_cond]
                d[col] = d[col].values * fac
        try:
            a, p = fit_auc(d)
            aucs.append(a)
            coefs.append(p)
        except Exception:
            continue
    aucs = np.array(aucs)
    coefs = np.array(coefs)
    names = ["const", *PRED_M1]
    coef_stab = {}
    for i, nm in enumerate(names):
        base = float(base_params[i])
        col = coefs[:, i]
        # 부호 안정성: 노이즈 하에서 base와 같은 부호 비율
        coef_stab[nm] = {
            "base": round(base, 4),
            "mc_mean": round(float(col.mean()), 4),
            "mc_std": round(float(col.std()), 4),
            "sign_stable_pct": round(float(np.mean(np.sign(col) == np.sign(base)) * 100), 1),
        }
    return {
        "base_auc": round(float(base_auc), 4),
        "mc_auc_mean": round(float(aucs.mean()), 4),
        "mc_auc_std": round(float(aucs.std()), 4),
        "mc_auc_ci95": [round(float(np.percentile(aucs, 2.5)), 4),
                        round(float(np.percentile(aucs, 97.5)), 4)],
        "perturbation_cv": {k: round(v, 3) for k, v in cvmap.items()},
        "coef_stability": coef_stab,
        "n_mc": int(len(aucs)),
        "note": "AUC가 측정오차 하에서 좁게 유지 + 계수 부호 안정 → POD 강건",
    }


def make_figs(rep, iso_res, merged, sens):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    # (a) BEW_V vs detection rate
    if merged is not None and len(merged):
        ax[0].scatter(merged["bew_v_mean"], merged["det_rate"], s=30,
                      c="#2166ac", label="BEW_V (optics)")
        ax[0].scatter(merged["bew_h_mean"], merged["det_rate"], s=30,
                      c="#b2182b", marker="^", alpha=0.6, label="BEW_H (optics+motion)")
        ax[0].set_xlabel("Edge-spread width (px)")
        ax[0].set_ylabel("Condition detection rate")
        ax[0].set_title("(a) BEW vs detection")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
    # (b) MC AUC 분포
    cs = sens["coef_stability"]
    labels = list(cs.keys())
    signpct = [cs[k]["sign_stable_pct"] for k in labels]
    ax[1].barh(labels, signpct, color="#4393c3")
    ax[1].axvline(95, ls="--", color="gray")
    ax[1].set_xlabel("Coefficient sign-stability (%) under IQ measurement error")
    ax[1].set_title(f"(b) M1 robustness (AUC {sens['mc_auc_mean']}±{sens['mc_auc_std']})")
    ax[1].set_xlim(0, 105)
    fig.tight_layout()
    fig.savefig(FIG / "wp5_psf_uncertainty.png", dpi=140)
    print(f"figure: {FIG/'wp5_psf_uncertainty.png'}")


def main():
    cam2 = pd.read_csv(CAM2, encoding="utf-8-sig")
    cam2.columns = [c.strip() for c in cam2.columns]
    long_df = pd.read_csv(LONG)

    print(f"cam2 조건 {len(cam2)}, long {len(long_df)} 행\n")
    result = {}

    print("── WP5-1 측정 반복성·불확도 ──")
    rep = repeatability(cam2)
    result["repeatability"] = rep
    for m, v in rep.items():
        print(f"  {m}: 반복성 CV={v['repeatability_cv_pct']}% "
              f"(abs std {v['typical_abs_std']}, ~{v['n_frames_typical']} frames)")

    print("\n── WP5-2 등방 PSF 검증 ──")
    iso_res, merged = isotropic_psf(cam2, long_df)
    result["isotropic_psf"] = iso_res
    bv = iso_res["bew_v_across_conditions"]
    print(f"  BEW_V 조건간: mean={bv['mean']} CV={bv['cv_pct']}% range={bv['range']}")
    an = iso_res["bew_v_anova"]
    print(f"  BEW_V~dist+iso+speed: R²={an['r2']}, p(dist)={an['p_values'].get('dist')}, "
          f"p(speed)={an['p_values'].get('speed')}")
    hv = iso_res["hv_ratio_vs_speed"]
    print(f"  hv_ratio by speed={hv['mean_by_speed']} (speed p={hv['speed_p']})")
    mi = iso_res["mtf_isotropy"]
    print(f"  MTF: mtf_v={mi['mtf_v_mean']} > mtf_h={mi['mtf_h_mean']}? "
          f"V-sharper {mi['v_sharper_frac']*100:.0f}%")
    if "bew_detection_corr" in iso_res:
        bc = iso_res["bew_detection_corr"]
        print(f"  BEW-검출 상관: BEW_V r={bc['pearson_bew_v_vs_detrate']}, "
              f"BEW_H r={bc['pearson_bew_h_vs_detrate']}")

    print("\n── WP5-3 POD 측정오차 민감도 (Monte Carlo) ──")
    sens = pod_sensitivity(long_df, rep, n_mc=500)
    result["pod_sensitivity"] = sens
    print(f"  base AUC={sens['base_auc']} → MC {sens['mc_auc_mean']}±{sens['mc_auc_std']} "
          f"CI{sens['mc_auc_ci95']}")
    for nm, s in sens["coef_stability"].items():
        print(f"    {nm}: 부호안정 {s['sign_stable_pct']}% (base {s['base']})")

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")
    make_figs(rep, iso_res, merged, sens)


if __name__ == "__main__":
    main()
