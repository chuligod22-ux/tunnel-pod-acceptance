#!/usr/bin/env python3
"""
wp4_thresholds_region.py — Phase 4 리비전 WP4: 임계값 신중화 + joint acceptance region (R1·R3).

  WP4-1 C_M 임계값 물리범위 제약: C_M∈[0,1](Michelson)인데 A6 CI[−0.66,1.48]가
        음수 포함 → 물리적 무의미. 실용성(사실상 하한 무제약) 논의 + [0,1] clip.
  WP4-2 MTF50 부호혼동: mtf_h 다변량 계수 음수(−110). 단변량 vs 다변량 부호 비교 +
        거리 교락 정량화 → 근거리 포화 confounding 입증, M1 유지 정당화(or caveat).
  WP4-3 joint acceptance region: M1로 (L_90, BEW_H) 평면의 P_d≥0.9 영역 시각화
        (C_M·MTF50 중앙값 고정, a*=0.5mm) + 50조건 점 오버레이.

출력: wp4_thresholds_region.json + viz/wp4_joint_region.png + 콘솔.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
LONG = HERE.parent.parent.parent / "tmp/v2_analysis/long_v3.csv"
if not LONG.exists():
    LONG = (HERE.parent.parent / "01_paper/output/ieee_tim_v2/supplementary"
            "/tier1_code_data/data/long_v3.csv")
OUT = HERE / "wp4_thresholds_region.json"
FIG = HERE / "viz"

PRED_M1 = ["width_mm", "C_M", "L_90", "mtf_h", "bew_h"]
A_STAR = 0.5           # 목표 결함 폭(mm)
PD_TARGET = 0.9


def fit_m1(df):
    cc = df.dropna(subset=["sigma_motion"]).reset_index(drop=True)
    X = sm.add_constant(cc[PRED_M1].astype(float).values)
    y = cc["detected"].astype(int).values
    res = sm.Logit(y, X).fit(disp=False, maxiter=200)
    coef = dict(zip(["const", *PRED_M1], res.params))
    return res, coef, cc, y


def univariate_signs(cc, y):
    """각 IQ 단변량 로지스틱 계수 부호 (다변량과 비교 → 교락/억제 진단)."""
    out = {}
    for m in ["C_M", "L_90", "mtf_h", "bew_h"]:
        X = sm.add_constant(cc[[m]].astype(float).values)
        r = sm.Logit(y, X).fit(disp=False, maxiter=200)
        out[m] = {"uni_coef": round(float(r.params[1]), 4),
                  "uni_sign": "+" if r.params[1] > 0 else "-"}
    return out


def mtf_distance_confounding(cc):
    """MTF50 부호혼동: mtf_h와 거리/포화의 교락 정량화."""
    r_dist = float(np.corrcoef(cc["mtf_h"], cc["dist"])[0, 1])
    # 근거리(≤3.5m) vs 원거리에서 mtf_h~detection 관계
    near = cc[cc["dist"] <= 3.5]
    far = cc[cc["dist"] >= 4.5]
    def slope(d):
        if d["detected"].nunique() < 2 or len(d) < 20:
            return None
        X = sm.add_constant(d[["mtf_h"]].astype(float).values)
        try:
            return round(float(sm.Logit(d["detected"].astype(int).values, X)
                               .fit(disp=False, maxiter=200).params[1]), 3)
        except Exception:
            return None
    # L_90(노출) 통제 시 mtf_h 부호
    X2 = sm.add_constant(cc[["mtf_h", "L_90"]].astype(float).values)
    adj = sm.Logit(cc["detected"].astype(int).values, X2).fit(disp=False, maxiter=200)
    return {
        "corr_mtf_h_distance": round(r_dist, 3),
        "mtf_slope_near_d<=3.5": slope(near),
        "mtf_slope_far_d>=4.5": slope(far),
        "mtf_coef_adjusted_for_L90": round(float(adj.params[1]), 3),
        "note": "단변량/근원거리 분리/노출통제 시 부호가 바뀌면 근거리 포화 교락",
    }


def threshold_physical(coef, cc):
    """C_M/MTF50 등 per-axis 임계값 물리범위 제약 논의."""
    medians = {m: float(cc[m].median()) for m in ["C_M", "L_90", "mtf_h", "bew_h"]}
    logit_t = float(np.log(PD_TARGET / (1 - PD_TARGET)))

    def invert(axis):
        # logit = const + b_w*a* + Σ b_j*median_j; axis만 풀기
        lin = coef["const"] + coef["width_mm"] * A_STAR
        for m in ["C_M", "L_90", "mtf_h", "bew_h"]:
            if m != axis:
                lin += coef[m] * medians[m]
        # logit_t = lin + coef[axis]*thr → thr
        return (logit_t - lin) / coef[axis]

    res = {}
    phys = {"C_M": (0.0, 1.0), "L_90": (0.0, 255.0),
            "mtf_h": (0.0, 0.5), "bew_h": (0.0, None)}
    for axis in ["C_M", "L_90", "mtf_h", "bew_h"]:
        thr = invert(axis)
        lo, hi = phys[axis]
        clipped = min(max(thr, lo), hi if hi is not None else thr)
        res[axis] = {
            "threshold": round(float(thr), 4),
            "physical_range": [lo, hi],
            "sample_range": [round(float(cc[axis].min()), 3), round(float(cc[axis].max()), 3)],
            "coef_sign": "+" if coef[axis] > 0 else "-",
            "within_physical": bool((thr >= lo) and (hi is None or thr <= hi)),
        }
    res["_note"] = ("C_M 임계 CI가 음수 포함(A6 [−0.66,1.48])→물리적 무의미, "
                    "하한 사실상 무제약. L_90이 신뢰 게이트.")
    return res, medians


def joint_region_fig(coef, cc, medians):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    FIG.mkdir(exist_ok=True)
    l90 = np.linspace(cc["L_90"].min(), cc["L_90"].max(), 120)
    bew = np.linspace(cc["bew_h"].min(), cc["bew_h"].max(), 120)
    L, B = np.meshgrid(l90, bew)
    lin = (coef["const"] + coef["width_mm"] * A_STAR
           + coef["C_M"] * medians["C_M"] + coef["mtf_h"] * medians["mtf_h"]
           + coef["L_90"] * L + coef["bew_h"] * B)
    pd_grid = 1 / (1 + np.exp(-lin))

    fig, ax = plt.subplots(figsize=(6.4, 5))
    cs = ax.contourf(L, B, pd_grid, levels=np.linspace(0, 1, 11), cmap="RdYlGn", alpha=0.85)
    ax.contour(L, B, pd_grid, levels=[PD_TARGET], colors="k", linewidths=2)
    # 조건 점 (검출률로 색)
    det = cc.groupby(["speed", "iso", "dist"]).agg(
        L_90=("L_90", "first"), bew_h=("bew_h", "first"),
        det=("detected", "mean")).reset_index()
    ax.scatter(det["L_90"], det["bew_h"], c=det["det"], cmap="RdYlGn",
               edgecolors="k", s=45, vmin=0, vmax=1, zorder=5)
    ax.set_xlabel("$L_{90}$ (90th-percentile luminance, DN)")
    ax.set_ylabel("$\\mathrm{BEW}_H$ (edge-spread width, px)")
    ax.set_title(f"Joint acceptance region (M1, $a^*$={A_STAR}mm, "
                 f"$P_d\\geq${PD_TARGET})\nblack contour = acceptance boundary")
    fig.colorbar(cs, ax=ax, label="Predicted $P_d$")
    fig.tight_layout()
    fig.savefig(FIG / "wp4_joint_region.png", dpi=140)
    return str(FIG / "wp4_joint_region.png")


def main():
    df = pd.read_csv(LONG)
    res, coef, cc, y = fit_m1(df)
    print(f"M1 refit: n={len(cc)}, AUC={roc_auc_score(y, res.predict(sm.add_constant(cc[PRED_M1].astype(float).values))):.4f}")
    print("계수:", {k: round(float(v), 3) for k, v in coef.items()})

    uni = univariate_signs(cc, y)
    print("\n── WP4-2 부호: 단변량 vs 다변량 ──")
    for m, u in uni.items():
        multi = "+" if coef[m] > 0 else "-"
        flag = "  ⚠부호반전(억제/교락)" if u["uni_sign"] != multi else ""
        print(f"  {m}: 단변량 {u['uni_sign']}({u['uni_coef']}) vs 다변량 {multi}{flag}")

    conf = mtf_distance_confounding(cc)
    print("\n── MTF50 거리 교락 ──")
    for k, v in conf.items():
        if k != "note":
            print(f"  {k}: {v}")

    thr, medians = threshold_physical(coef, cc)
    print("\n── WP4-1 임계값 물리범위 ──")
    for axis in ["C_M", "L_90", "mtf_h", "bew_h"]:
        t = thr[axis]
        print(f"  {axis}: thr={t['threshold']} 부호{t['coef_sign']} "
              f"물리범위{t['physical_range']} 표본범위{t['sample_range']} "
              f"{'OK' if t['within_physical'] else '⚠범위밖'}")

    figpath = joint_region_fig(coef, cc, medians)
    print(f"\nfigure: {figpath}")

    out = {"m1_coef": {k: float(v) for k, v in coef.items()},
           "univariate_vs_multivariable": uni,
           "mtf_distance_confounding": conf,
           "threshold_physical": thr}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
