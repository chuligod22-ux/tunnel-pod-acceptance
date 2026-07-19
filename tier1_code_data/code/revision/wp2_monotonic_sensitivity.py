#!/usr/bin/env python3
"""
wp2_monotonic_sensitivity.py — Phase 4 리비전 WP2: monotonic-completion 민감도 (R1·R3).

논문은 검출 라벨에 monotonic-completion 가정(폭 w 검출 시 w'>w 모두 검출)을 적용.
리뷰어: 이 가정 없이(원 visible-list 라벨) 또는 다른 임계값에서 M1/M2 결론 유지?

기존 POD-only 민감도(ΔAIC=118)를 넘어, **다변량 IQ 로지스틱 M0/M1/M2를 원 라벨로 재적합**
하여 핵심 결론(M1≫M0, σ_motion 중복성, a90/95, 계수 부호)의 강건성을 검증.

원 라벨 = cam1_crack_detectability.csv 의 `visible=[...]` (실제 관측된 폭만 1).
출력: wp2_monotonic_sensitivity.json + 콘솔.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
LONG = HERE.parent.parent.parent / "tmp/v2_analysis/long_v3.csv"
if not LONG.exists():
    LONG = (HERE.parent.parent / "01_paper/output/ieee_tim_v2/supplementary"
            "/tier1_code_data/data/long_v3.csv")
DETECT = HERE.parent.parent / "03_src/data/cam1_crack_detectability.csv"
OUT = HERE / "wp2_monotonic_sensitivity.json"

PRED_M0 = ["width_mm"]
PRED_M1 = ["width_mm", "C_M", "L_90", "mtf_h", "bew_h"]
PRED_M2 = ["width_mm", "C_M", "L_90", "mtf_h", "bew_h", "sigma_motion"]


def parse_visible(notes: str):
    """notes에서 visible=[...] 폭 리스트 파싱 (논문 sensitivity_monotonic.py와 동일).

    반환: None(정보없음) | [](빈=없음) | [폭...]
    """
    if not isinstance(notes, str):
        return None
    m = re.search(r"visible=\[([^\]]*)\]", notes)
    if not m:
        return None
    s = m.group(1).strip()
    if not s:
        return []
    return [round(float(x.strip()), 2) for x in s.split(",")]


def build_raw_labels(long_df):
    """
    논문 정의(sensitivity_monotonic.py rows_vis)로 원 visible-list 라벨 재구성:
      user_labeled_x → 0, user_verified → 1, visible없음 → mono fallback,
      else → w in visible.
    반환: (raw_df, meta) — meta: 조건별 (mdw, notes-flags, visible)
    """
    det = pd.read_csv(DETECT)
    info = {}
    for _, r in det.iterrows():
        key = (int(r["speed"]), int(r["iso"]), float(r["dist"]))
        notes = r.get("notes", "")
        info[key] = {
            "mdw": r["min_crack_mm"] if pd.notna(r["min_crack_mm"]) else None,
            "user_x": isinstance(notes, str) and "user_labeled_x" in notes,
            "user_v": isinstance(notes, str) and "user_verified" in notes,
            "visible": parse_visible(notes),
        }
    raw = long_df.copy()
    raw_det, n_incons = [], 0
    for _, r in raw.iterrows():
        key = (int(r["speed"]), int(r["iso"]), float(r["dist"]))
        w = round(float(r["width_mm"]), 2)
        it = info.get(key, {"mdw": None, "user_x": False, "user_v": False, "visible": None})
        mono = int(r["detected"])  # long_v3 = monotonic (검증됨)
        if it["user_x"]:
            vis = 0
        elif it["user_v"]:
            vis = 1
        elif it["visible"] is None:
            vis = mono  # 정보없음 → mono fallback
        else:
            vis = 1 if w in it["visible"] else 0
        raw_det.append(vis)
        if mono == 1 and vis == 0 and it["visible"] is not None \
                and not it["user_x"] and not it["user_v"]:
            n_incons += 1
    raw["detected"] = raw_det
    return raw, info, n_incons


def fit(df, predictors):
    d = df.dropna(subset=["sigma_motion"]) if "sigma_motion" in predictors else df
    X = sm.add_constant(d[predictors].astype(float).values)
    y = d["detected"].astype(int).values
    res = sm.Logit(y, X).fit(disp=False, maxiter=200)
    return {"aic": float(res.aic), "llf": float(res.llf), "n_params": X.shape[1],
            "auc": float(roc_auc_score(y, res.predict(X))),
            "coef": {n: float(v) for n, v in zip(["const", *predictors], res.params)},
            "n": int(len(y))}


def lrt(m_small, m_big):
    stat = 2 * (m_big["llf"] - m_small["llf"])
    p = float(sps.chi2.sf(stat, m_big["n_params"] - m_small["n_params"]))
    return {"delta_aic": round(m_small["aic"] - m_big["aic"], 2), "p_value": p}


def a90_95(m0):
    b0, b1 = m0["coef"]["const"], m0["coef"]["width_mm"]
    a90 = (np.log(0.9 / 0.1) - b0) / b1
    return round(float(a90), 3)


def analyze_label_set(df, tag):
    """한 라벨셋(monotonic 또는 raw)에 M0/M1/M2 적합 + 핵심 지표."""
    cc = df.dropna(subset=["sigma_motion"])
    m0 = fit(cc, PRED_M0)
    m1 = fit(cc, PRED_M1)
    m2 = fit(cc, PRED_M2)
    return {
        "tag": tag, "n_rows": int(len(cc)),
        "n_detected": int(cc["detected"].sum()),
        "M0": {"aic": round(m0["aic"], 2), "auc": round(m0["auc"], 4)},
        "M1": {"aic": round(m1["aic"], 2), "auc": round(m1["auc"], 4),
               "coef_signs": {k: ("+" if v > 0 else "-")
                              for k, v in m1["coef"].items()}},
        "M2": {"aic": round(m2["aic"], 2), "auc": round(m2["auc"], 4)},
        "LRT_M0_M1": lrt(m0, m1),
        "LRT_M1_M2": lrt(m1, m2),
        "a90_95_wald_mm": a90_95(m0),
    }


def main():
    long_df = pd.read_csv(LONG)
    raw, info, n_incons = build_raw_labels(long_df)

    n_diff = int((raw["detected"].values != long_df["detected"].values).sum())
    print(f"라벨 데이터: {len(long_df)} 행")
    print(f"monotonic vs visible-list 불일치(논문 정의, restricted): {n_incons} "
          f"(문서 기록 204와 대조)")
    print(f"전체 라벨 상이 행: {n_diff}\n")

    vis_lists = [it["visible"] for it in info.values() if it["visible"]]
    # 라벨 카운트 (WP2 투명성)
    label_counts = {
        "total_rows": int(len(long_df)),
        "monotonic_detected": int(long_df["detected"].sum()),
        "visible_list_detected": int(raw["detected"].sum()),
        "n_inconsistencies_restricted": n_incons,
        "n_conditions_user_verified": int(sum(1 for it in info.values() if it["user_v"])),
        "n_conditions_user_x": int(sum(1 for it in info.values() if it["user_x"])),
        "n_conditions_with_visible_list": int(len(vis_lists)),
        "avg_visible_widths_per_cond": round(float(np.mean([len(v) for v in vis_lists])), 2)
        if vis_lists else 0,
    }
    print("── 라벨 카운트 ──")
    for k, v in label_counts.items():
        print(f"  {k}: {v}")

    mono = analyze_label_set(long_df, "monotonic_completion")
    rawr = analyze_label_set(raw, "raw_visible_list")

    print("\n── 민감도: 핵심 결론 강건성 ──")
    print(f"{'지표':<22}{'monotonic':>14}{'raw(no-compl)':>16}")
    rows = [
        ("M1 AUC", mono["M1"]["auc"], rawr["M1"]["auc"]),
        ("ΔAIC(M0→M1)", mono["LRT_M0_M1"]["delta_aic"], rawr["LRT_M0_M1"]["delta_aic"]),
        ("M1→M2 LRT p", round(mono["LRT_M1_M2"]["p_value"], 3), round(rawr["LRT_M1_M2"]["p_value"], 3)),
        ("a90/95 (mm)", mono["a90_95_wald_mm"], rawr["a90_95_wald_mm"]),
        ("n_detected", mono["n_detected"], rawr["n_detected"]),
    ]
    for name, a, b in rows:
        print(f"{name:<22}{a:>14}{b:>16}")
    print(f"\nM1 계수 부호 일치: "
          f"{mono['M1']['coef_signs'] == rawr['M1']['coef_signs']} "
          f"(mono {mono['M1']['coef_signs']})")

    result = {"n_inconsistencies": n_incons, "label_counts": label_counts,
              "monotonic": mono, "raw_visible": rawr}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
