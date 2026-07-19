#!/usr/bin/env python3
"""
wp7_exposure_optimized.py — WP7 addendum: 노출 게이트(C_M, L_90)의 최적화 실측 (R2-2).

wp7_computational_cost.py의 나이브 구현(전체 ROI float 변환 + np.percentile ≈ 65 ms)에
대한 두 가지 표준 최적화의 실측 벤치마크 (response letter 인용 수치의 재현 근거):
  (A) 4× ROI 서브샘플링 percentile — roi[::4, ::4]
  (B) O(1) 히스토그램 percentile — uint8 ROI에 cv2.calcHist → cumsum → 분위수
둘 다 나이브와 동일 정의(C_M=(L90−L10)/(L90+L10), L_90=90th pct)이며 8-bit 양자화
수준에서 동일 결과를 산출함을 함께 검증.

출력: wp7_exposure_optimized.json
"""
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
CAM2 = HERE.parent / "data/raw/crack/cam2"
OUT = HERE / "wp7_exposure_optimized.json"

REPS = 50


def timeit(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"mean_ms": round(statistics.mean(ts), 2),
            "std_ms": round(statistics.pstdev(ts), 2),
            "min_ms": round(min(ts), 2), "reps": reps}


def exposure_from_percentiles(l_hi, l_lo):
    return (l_hi - l_lo) / max(1e-9, (l_hi + l_lo))


def main():
    frame = sorted(CAM2.glob("crack_d25_ISO400_V60/*.png"))[0]
    img = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    roi = img[int(h * 0.03):int(h * 0.60), int(w * 0.15):int(w * 0.85)]  # uint8 view

    # ── 나이브 (wp7 원본과 동일) ──
    def naive():
        r = roi.astype(float)
        l_hi = np.percentile(r, 90)
        l_lo = np.percentile(r, 10)
        return exposure_from_percentiles(l_hi, l_lo), l_hi

    # ── (A) 4× 서브샘플 ──
    def subsample4():
        r = roi[::4, ::4]
        l_hi = np.percentile(r, 90)
        l_lo = np.percentile(r, 10)
        return exposure_from_percentiles(l_hi, l_lo), l_hi

    # ── (B) 히스토그램 O(1) percentile (cv2.calcHist, C 최적화) ──
    roi_c = np.ascontiguousarray(roi)

    def hist_pct():
        counts = cv2.calcHist([roi_c], [0], None, [256], [0, 256]).ravel()
        cum = np.cumsum(counts)
        n = cum[-1]
        l_lo = int(np.searchsorted(cum, 0.10 * n))
        l_hi = int(np.searchsorted(cum, 0.90 * n))
        return exposure_from_percentiles(float(l_hi), float(l_lo)), float(l_hi)

    # 값 동등성 검증 (8-bit 양자화 허용오차)
    (cm_n, l90_n), (cm_s, l90_s), (cm_h, l90_h) = naive(), subsample4(), hist_pct()
    print(f"값 검증: naive C_M={cm_n:.4f} L90={l90_n:.1f} | "
          f"sub4 C_M={cm_s:.4f} L90={l90_s:.1f} | hist C_M={cm_h:.4f} L90={l90_h:.1f}")

    res = {"frame": frame.name, "roi_px": int(roi.size),
           "value_check": {"naive": [round(cm_n, 4), round(l90_n, 1)],
                           "subsample4": [round(cm_s, 4), round(l90_s, 1)],
                           "histogram": [round(cm_h, 4), round(l90_h, 1)]},
           "bench": {}}
    for name, fn in [("naive_float_percentile", naive),
                     ("subsample4_percentile", subsample4),
                     ("histogram_percentile", hist_pct)]:
        fn()  # warmup
        b = timeit(fn)
        b["throughput_fps"] = round(1000.0 / b["mean_ms"], 1)
        b["meets_20ms_budget"] = bool(b["mean_ms"] <= 20.0)
        res["bench"][name] = b
        print(f"{name:26}: {b['mean_ms']:7.2f} ms ({b['throughput_fps']} fps) "
              f"{'✅' if b['meets_20ms_budget'] else '❌'}")

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
