#!/usr/bin/env python3
"""
external_validation_prep.py — 외적검증 병합 테이블 준비 (v2: 논문 GT 기반).

⚠️ v1은 구 Measurement 시절 VISUAL 육안표(2026-03-26)를 비교자로 사용했으나,
이는 논문 3.4절 GT(`data/cam1_crack_detectability.csv`, CV+전문가 3-stage)와
모순됨(oosota 리뷰 C1, 2026-07-17). v2는 논문 GT에서 두 수준의 비교자를 도출:
  (1) chart_identifiable   — 노출-적정성 수준 (MDW 존재 = 취득이 어떤 폭이든 지원)
  (2) chart_0p3_detectable — 폭 수준 (MDW ≤ 0.3 mm, 원고 Table III의 D(0.3)과 동일)

+ cam2 IQ(MTF/BEW/σ_motion) 병합. CNN 실측 검출률은 별도 산출 후 join.
"""
import csv
from pathlib import Path

HERE = Path(__file__).parent
CAM2_IQ = HERE.parent / "data/cam2_analysis.csv"
GT_CSV = HERE.parent / "data/cam1_crack_detectability.csv"
OUT = HERE / "external_validation_table.csv"

GT_WIDTH = 0.3


def _f(s, nd):
    """빈 문자열/결측 안전 float 변환."""
    try:
        return round(float(s), nd)
    except (ValueError, TypeError):
        return ""


def load_gt():
    """논문 3.4절 GT: (spd,iso,dist) -> MDW(float) 또는 None(expert-flagged-fail)."""
    gt = {}
    with GT_CSV.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (int(r["speed"]), int(r["iso"]), float(r["dist"]))
            m = r["min_crack_mm"]
            gt[key] = float(m) if m not in ("", None) else None
    return gt


def load_iq():
    iq = {}
    with CAM2_IQ.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (int(r["speed"]), int(r["iso"]), float(r["dist"]))
            iq[key] = {
                "mtf_h": _f(r.get("mtf_h_mean"), 4),
                "bew_h": _f(r.get("bew_h_mean"), 3),
                "bew_v": _f(r.get("bew_v_mean"), 3),
                "sigma_motion": _f(r.get("sigma_motion"), 3),
            }
    return iq


def main():
    gt = load_gt()
    iq = load_iq()
    rows = []
    for (spd, iso, dist), minw in sorted(gt.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        cond = f"crack_d{int(dist*10)}_ISO{iso}_V{spd}"
        row = {
            "condition": cond, "speed_kmh": spd, "iso": iso, "distance_m": dist,
            "chart_min_width_mm": "NA" if minw is None else minw,
            "chart_identifiable": int(minw is not None),
            "chart_0p3_detectable": int(minw is not None and minw <= GT_WIDTH),
        }
        row.update(iq.get((spd, iso, dist), {"mtf_h": "", "bew_h": "", "bew_v": "", "sigma_motion": ""}))
        rows.append(row)

    fields = ["condition", "speed_kmh", "iso", "distance_m", "chart_min_width_mm",
              "chart_identifiable", "chart_0p3_detectable",
              "mtf_h", "bew_h", "bew_v", "sigma_motion"]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_id = sum(r["chart_identifiable"] for r in rows)
    n_03 = sum(r["chart_0p3_detectable"] for r in rows)
    print(f"병합 테이블 저장: {OUT} ({len(rows)} 조건) — 논문 GT 기반")
    print(f"chart_identifiable(노출-적정성): {n_id}/50")
    print(f"chart_0p3_detectable(MDW<=0.3): {n_03}/50  [원고 Table III D(0.3)=31/50과 일치해야 함]")
    print("expert-flagged-fail:", [r["condition"] for r in rows if not r["chart_identifiable"]])


if __name__ == "__main__":
    main()
