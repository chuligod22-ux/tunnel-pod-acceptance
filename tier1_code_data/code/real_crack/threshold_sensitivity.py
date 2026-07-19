#!/usr/bin/env python3
"""
threshold_sensitivity.py — WP1 검출 판정 임계값 민감도 (R1-1/R2-3 Methods 3.7 뒷받침).

프레임 임계값 CRACK_PX_DETECT(기본 150px)과 조건 판정 비율 COND_DETECT_FRAC(기본 0.5)을
넓은 범위로 변화시키며:
  (1) 차트 노출-적정성 예측(chart_identifiable, 논문 GT) vs CNN 조건 검출 일치율
  (2) ISO 역-U 패턴 유지 여부 (중간 ISO 400/800 평균 > 양 극단 100/1600 평균)
  (3) 거리 단조성 (Spearman 부호) 유지 여부
출력: threshold_sensitivity.json + 콘솔 표
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CNN = HERE / "cam2_cnn_detection.csv"
TABLE = HERE / "external_validation_table.csv"
OUT = HERE / "threshold_sensitivity.json"

PX_GRID = [50, 75, 100, 150, 200, 300, 500]
FRAC_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


def load():
    frames = defaultdict(list)  # cond -> [crack_px,...]
    meta = {}
    with CNN.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            frames[r["condition"]].append(float(r["crack_px"]))
            meta[r["condition"]] = (int(r["iso"]), float(r["distance_m"]))
    chart = {}
    with TABLE.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            chart[r["condition"]] = int(r["chart_identifiable"])
    return frames, meta, chart


def evaluate(frames, meta, chart, px_thr, frac_thr):
    det_rate = {c: np.mean([p >= px_thr for p in v]) for c, v in frames.items()}
    cond_det = {c: int(det_rate[c] >= frac_thr) for c in det_rate}
    conds = sorted(det_rate)
    agree = np.mean([cond_det[c] == chart[c] for c in conds])
    # ISO 역-U: mid(400,800) 평균 검출률 > 극단(100,1600) 평균
    by_iso = defaultdict(list)
    by_d = defaultdict(list)
    for c in conds:
        iso, d = meta[c]
        by_iso[iso].append(det_rate[c])
        by_d[d].append(det_rate[c])
    iso_mean = {k: np.mean(v) for k, v in by_iso.items()}
    inv_u = (np.mean([iso_mean[400], iso_mean[800]])
             > np.mean([iso_mean[100], iso_mean[1600]]))
    # 거리 단조감소: 2.5m 평균 > 6.5m 평균 & Spearman<0
    d_sorted = sorted(by_d)
    d_means = [np.mean(by_d[d]) for d in d_sorted]
    rho = np.corrcoef(np.argsort(np.argsort(d_sorted)),
                      np.argsort(np.argsort(d_means)))[0, 1]
    dist_mono = d_means[0] > d_means[-1] and rho < 0
    return {"px_thr": px_thr, "frac_thr": frac_thr,
            "agreement": round(float(agree), 3),
            "n_cond_detected": int(sum(cond_det.values())),
            "iso_inverted_u": bool(inv_u),
            "iso_means": {str(k): round(float(v), 3) for k, v in sorted(iso_mean.items())},
            "dist_means": {str(d): round(float(m), 3) for d, m in zip(d_sorted, d_means)},
            "dist_monotone_decreasing": bool(dist_mono)}


def main():
    frames, meta, chart = load()
    results = []
    # (A) px 임계 스윕 (frac=0.5 고정)
    print("── (A) 프레임 픽셀 임계 스윕 (조건 판정 비율 0.5 고정) ──")
    print(f"{'px_thr':>7} {'일치율':>7} {'검출조건':>8} {'ISO역U':>7} {'거리단조':>8}")
    for px in PX_GRID:
        r = evaluate(frames, meta, chart, px, 0.5)
        results.append(r)
        print(f"{px:>7} {r['agreement']:>7.2f} {r['n_cond_detected']:>8} "
              f"{str(r['iso_inverted_u']):>7} {str(r['dist_monotone_decreasing']):>8}")
    # (B) frac 스윕 (px=150 고정)
    print("\n── (B) 조건 판정 비율 스윕 (픽셀 임계 150 고정) ──")
    print(f"{'frac':>7} {'일치율':>7} {'검출조건':>8} {'ISO역U':>7} {'거리단조':>8}")
    for fr in FRAC_GRID:
        r = evaluate(frames, meta, chart, 150, fr)
        results.append(r)
        print(f"{fr:>7} {r['agreement']:>7.2f} {r['n_cond_detected']:>8} "
              f"{str(r['iso_inverted_u']):>7} {str(r['dist_monotone_decreasing']):>8}")
    agr = [r["agreement"] for r in results]
    summary = {"agreement_min": min(agr), "agreement_max": max(agr),
               "iso_inverted_u_all": all(r["iso_inverted_u"] for r in results),
               "dist_monotone_all": all(r["dist_monotone_decreasing"] for r in results),
               "px_grid": PX_GRID, "frac_grid": FRAC_GRID}
    OUT.write_text(json.dumps({"summary": summary, "results": results},
                              indent=2, ensure_ascii=False))
    print(f"\n요약: 일치율 [{summary['agreement_min']:.2f}, {summary['agreement_max']:.2f}], "
          f"ISO역U 전부 유지={summary['iso_inverted_u_all']}, "
          f"거리단조 전부 유지={summary['dist_monotone_all']}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
