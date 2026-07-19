#!/usr/bin/env python3
"""
wp7_computational_cost.py — Phase 4 리비전 WP7: 연산비용 실측 (R2-2).

리뷰어: real-time 지향인지 명확화 + 특정 하드웨어에서 연산비용 보고 →
60-80 km/h(50 fps/camera) 대응 가능성 평가.

측정(본 하드웨어, 실제 8192×5460 프레임):
  (1) 노출 IQ (C_M, L_90)            — grayscale + ROI + percentile
  (2) 광학 IQ (MTF50 + BEW, ISO12233) — Canny + ESF + FFT (slant-edge)
  (3) CNN (SegFormer-B4) 검출         — 콘크리트밴드 네이티브 512 타일링, MPS

요구 처리량: 카메라당 50 fps → 프레임당 20 ms 예산 (2 카메라 = 100 fps 총).
프레임워크는 취득 영상에 대한 offline acceptance-gating이며 실시간 검출기가 아님을
명확히 하되, 각 단계의 실측 throughput으로 실시간 실행 여지를 정직하게 보고.

출력: wp7_computational_cost.json + 콘솔.
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
SRC = HERE.parent.parent / "03_src"
DATA = SRC / "data"
CAM2 = DATA / "raw/crack/cam2"
CAM1 = DATA / "raw/crack/cam1"
OUT = HERE / "wp7_computational_cost.json"

FPS_PER_CAM = 50
BUDGET_MS = 1000.0 / FPS_PER_CAM   # 20 ms


def timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"mean_ms": round(statistics.mean(ts), 2),
            "std_ms": round(statistics.pstdev(ts), 2) if len(ts) > 1 else 0.0,
            "min_ms": round(min(ts), 2), "reps": reps}


def bench_exposure(frame_path):
    """노출 IQ: grayscale + 고정 ROI + Michelson/L_90 (crack_contrast_analysis 방식)."""
    def run():
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        roi = img[int(h * 0.03):int(h * 0.60), int(w * 0.15):int(w * 0.85)].astype(float)
        l_hi = np.percentile(roi, 90)
        l_lo = np.percentile(roi, 10)
        _ = (l_hi - l_lo) / (l_hi + l_lo)
    # 첫 회 디코드 캐시 배제 위해 warmup
    run()
    return timeit(run, 8)


def bench_exposure_nodecode(frame_path):
    """디코드 제외한 순수 IQ 연산 (이미 로드된 프레임)."""
    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    def run():
        roi = img[int(h * 0.03):int(h * 0.60), int(w * 0.15):int(w * 0.85)].astype(float)
        l_hi = np.percentile(roi, 90)
        l_lo = np.percentile(roi, 10)
        _ = (l_hi - l_lo) / (l_hi + l_lo)
    run()
    return timeit(run, 20)


def bench_mtf_bew(frame_path):
    """MTF50 + BEW (ISO 12233): mtf50_bew_calculator.analyze_image."""
    import sys
    sys.path.insert(0, str(SRC))
    try:
        from mtf50_bew_calculator import analyze_image
    except Exception as e:  # noqa: BLE001
        return {"error": f"import 실패: {e}"}

    def run():
        try:
            analyze_image(str(frame_path))
        except Exception:  # noqa: BLE001
            pass
    run()
    return timeit(run, 5)


def bench_cnn(frame_path, dist):
    """CNN SegFormer-B4 검출 (콘크리트밴드 타일링, MPS)."""
    import sys
    sys.path.insert(0, str(HERE.parent / "real_crack"))
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from cnn_detect import detect_frame, load_model, DEVICE
    except Exception as e:  # noqa: BLE001
        return {"error": f"import 실패: {e}"}, None
    load_model()  # 모델 로드(1회, 측정 제외)

    def run():
        detect_frame(str(frame_path), dist)
    run()  # warmup(타일 커널 컴파일 등)
    return timeit(run, 3), DEVICE


def main():
    result = {"hardware": {
        "platform": platform.platform(),
        "processor": platform.processor() or "Apple Silicon (arm64)",
        "python": platform.python_version(),
    }, "acquisition": {
        "resolution": "8192x5460", "fps_per_camera": FPS_PER_CAM,
        "cameras": 2, "budget_ms_per_frame": BUDGET_MS,
    }}
    print(f"하드웨어: {result['hardware']['platform']}")
    print(f"요구: {FPS_PER_CAM} fps/camera → 프레임당 예산 {BUDGET_MS:.1f} ms\n")

    # 대표 프레임
    cam2_f = sorted(CAM2.glob("crack_d25_ISO400_V60/*.png"))
    cam1_f = sorted(CAM1.glob("*.png"))
    frame2 = cam2_f[0] if cam2_f else None
    frame1 = cam1_f[0] if cam1_f else frame2

    stages = {}
    print("── (1) 노출 IQ (C_M, L_90) ──")
    stages["exposure_with_decode"] = bench_exposure(frame2)
    stages["exposure_compute_only"] = bench_exposure_nodecode(frame2)
    print(f"  디코드 포함: {stages['exposure_with_decode']['mean_ms']} ms "
          f"| 연산만: {stages['exposure_compute_only']['mean_ms']} ms")

    print("\n── (2) 광학 IQ (MTF50 + BEW, ISO 12233) ──")
    stages["mtf_bew"] = bench_mtf_bew(frame1)
    print(f"  {stages['mtf_bew']}")

    print("\n── (3) CNN (SegFormer-B4) 검출 ──")
    cnn_res, dev = bench_cnn(frame2, 2.5)
    stages["cnn_segformer"] = cnn_res
    stages["cnn_segformer"]["device"] = dev
    print(f"  {cnn_res} (device={dev})")

    # throughput 대조
    print("\n── 요구 처리량(50 fps=20 ms/frame) 대조 ──")
    summary = {}
    for name in ["exposure_compute_only", "mtf_bew", "cnn_segformer"]:
        s = stages.get(name, {})
        ms = s.get("mean_ms")
        if ms:
            fps = 1000.0 / ms
            summary[name] = {"ms_per_frame": ms, "throughput_fps": round(fps, 1),
                             "meets_50fps": bool(ms <= BUDGET_MS)}
            print(f"  {name:24}: {ms:8.2f} ms/frame → {fps:8.1f} fps "
                  f"{'✅ ≥50fps' if ms <= BUDGET_MS else '❌ <50fps (offline)'}")

    result["stages"] = stages
    result["throughput_summary"] = summary
    result["interpretation"] = (
        "IQ 게이팅(노출+광학)은 실시간 예산 내 실행 가능. CNN 검출기는 8K 프레임 "
        "타일링으로 프레임당 수백~수천 ms → 실시간 아님. 단 본 프레임워크는 취득 "
        "영상에 대한 offline acceptance-gating이며 CNN은 외적검증용 독립 검출기임."
    )
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
