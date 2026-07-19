#!/usr/bin/env python3
"""
wp3_m0_cv.py — WP3 addendum: M0(width-only)의 조건단위 GroupKFold CV AUC.

배경: wp3_diagnostics.py는 M1/M2의 CV AUC만 산출했으나, "IQ의 정직한(표본 외)
판별 기여"를 보이려면 동일 파이프라인의 M0 CV가 기준선으로 필요 (리뷰어 선제 대응).

결과 (2026-07-17): M0 CV 0.7012 < M2 CV 0.7064 < M1 CV 0.7191
→ 표본 외에서도 모델 순위 보존, IQ 정직 기여 +0.018 (겉보기 +0.108보다 작음 — 본문 병기).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).parent
SRC = HERE.parent.parent.parent / "tmp/v2_analysis/long_v3.csv"
OUT = HERE / "wp3_m0_cv.json"

M0 = ["width_mm"]
M1 = ["width_mm", "C_M", "L_90", "mtf_h", "bew_h"]
M2 = M1 + ["sigma_motion"]


def cv_auc(df, y, groups, preds, k=5):
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(df, y, groups):
        X = sm.add_constant(df.iloc[tr][preds].astype(float).values)
        m = sm.Logit(y[tr], X).fit(disp=0, maxiter=200)
        Xte = sm.add_constant(df.iloc[te][preds].astype(float).values)
        oof[te] = m.predict(Xte)
    return round(float(roc_auc_score(y, oof)), 4)


def main():
    df = (pd.read_csv(SRC).dropna(subset=["sigma_motion"]).reset_index(drop=True))
    df["cond"] = list(zip(df["speed"], df["iso"], df["dist"]))
    groups = df["cond"].astype(str).values
    y = df["detected"].astype(int).values
    res = {name: cv_auc(df, y, groups, preds)
           for name, preds in [("M0", M0), ("M1", M1), ("M2", M2)]}
    res["note"] = ("조건단위 GroupKFold 5-fold, wp3_diagnostics.py와 동일 파이프라인. "
                   "M0 CV가 기준선: 정직한 IQ 판별 기여 = M1 CV - M0 CV.")
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(res)


if __name__ == "__main__":
    main()
