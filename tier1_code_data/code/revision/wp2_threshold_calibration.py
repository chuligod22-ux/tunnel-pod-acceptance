#!/usr/bin/env python3
"""
wp2_threshold_calibration.py — R3-f/R1-2 대응: Stage-1 가시성 임계값 s=0.08의
데이터 근거 + 대안 임계값 민감도.

리뷰어 질문: "몇 개의 unambiguous 조건으로 캘리브레이션했나? blind했나?
대안 임계값에서도 결론이 유지되나?"

역사적 기록 대신 검증 가능한 두 사실을 산출:
  (1) 50 cam1 프레임 전체 패치의 s-점수 분포에서 0.08의 위치
      (가시/비가시 모드 사이 저밀도 영역이면 임계값의 데이터 근거)
  (2) 임계값 스윕(0.04–0.20): 조건별 Stage-1 visible list → MDW → 라벨이
      얼마나 변하는가 (기준 0.08 대비)

출력: wp2_threshold_calibration.json + 콘솔
"""
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SRC_DIR = HERE.parent
CAM1 = SRC_DIR / "data/raw/crack/cam1"
OUT = HERE / "wp2_threshold_calibration.json"

# analyze_crack_detectability.py 함수 재사용 (경로 상수만 무시)
spec = importlib.util.spec_from_file_location(
    "acd", SRC_DIR / "analyze_crack_detectability.py")
acd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acd)

THR_GRID = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
BASE = 0.08


def patch_scores_for_image(img_path):
    """한 프레임 → [(crack_w, score), ...] (Stage-1 파이프라인 그대로)."""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mb = float(np.mean(gray))
    if mb < 30 or mb > 230:
        return "exposure_fail"
    board = acd.find_board_bbox(gray)
    if board is None:
        h, w = gray.shape
        board = (0, 0, w, int(h * 0.65))
    bx, by, bw, bh = board
    bg = gray[by:by + bh, bx:bx + bw]
    squares = acd.detect_gray_squares(bg)
    if not squares:
        return "no_squares"
    out = []
    for row, col, sq in acd.row_col_assign(squares):
        cx, cy, x, y, w, h = sq
        pad = 3
        cell = bg[max(0, y + pad):min(bg.shape[0], y + h - pad),
                  max(0, x + pad):min(bg.shape[1], x + w - pad)]
        s = acd.measure_crack_visibility(cell)
        out.append((acd.crack_width_from_pos(row, col), float(s)))
    return out


def main():
    per_cond = {}          # cond -> [(w, s), ...]
    skipped = {}
    for p in sorted(CAM1.glob("*.png")):
        info = acd.parse_filename(p.stem)
        cond = (info["speed"], info["iso"], info["dist"])
        if info["is_x"]:
            skipped[str(cond)] = "expert_flagged_fail(_x)"
            continue
        r = patch_scores_for_image(p)
        if r is None or isinstance(r, str):
            skipped[str(cond)] = r or "read_error"
            continue
        per_cond[cond] = r
        print(f"  {p.stem}: {len(r)} patches", flush=True)

    all_scores = [s for v in per_cond.values() for (w, s) in v if w is not None]
    arr = np.array(all_scores)

    # (1) 분포에서 0.08의 위치: 저밀도 근거
    hist, edges = np.histogram(arr, bins=40, range=(0, 1))
    frac_below = float(np.mean(arr < BASE))
    # 0.08 주변 ±0.02 구간의 점수 밀도 vs 전체 평균 밀도
    near = float(np.mean((arr >= BASE - 0.02) & (arr < BASE + 0.02)))
    print(f"\n전체 패치 {len(arr)}개: median={np.median(arr):.3f}, "
          f"s<0.08 비율={frac_below:.2f}, 0.06≤s<0.10 밀도={near:.3f}")

    # (2) 임계값 스윕: 조건별 visible list → MDW → identifiable
    def labels_at(thr):
        lab = {}
        for cond, ps in per_cond.items():
            vis = sorted(w for (w, s) in ps if w is not None and s >= thr)
            lab[cond] = (min(vis) if vis else None, tuple(vis))
        return lab

    base_lab = labels_at(BASE)
    sweep = []
    print(f"\n{'thr':>6} {'MDW변화':>7} {'식별여부변화':>10} {'행라벨변화':>9}")
    for thr in THR_GRID:
        lab = labels_at(thr)
        mdw_ch = sum(1 for c in base_lab if lab[c][0] != base_lab[c][0])
        id_ch = sum(1 for c in base_lab
                    if (lab[c][0] is None) != (base_lab[c][0] is None))
        # 행 단위(모노토닉 완성 후) 변화: MDW 기준 D(a)=1[a>=MDW]
        row_ch = 0
        widths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for c in base_lab:
            m0, m1 = base_lab[c][0], lab[c][0]
            for w in widths:
                d0 = 0 if m0 is None else int(w >= m0)
                d1 = 0 if m1 is None else int(w >= m1)
                row_ch += int(d0 != d1)
        sweep.append({"thr": thr, "mdw_changed_conditions": mdw_ch,
                      "identifiable_changed_conditions": id_ch,
                      "monotonic_row_labels_changed": row_ch,
                      "n_conditions": len(base_lab)})
        print(f"{thr:>6} {mdw_ch:>7} {id_ch:>10} {row_ch:>9}")

    res = {"n_frames_scored": len(per_cond), "n_skipped": len(skipped),
           "skipped": skipped, "n_patches": len(arr),
           "score_median": round(float(np.median(arr)), 4),
           "frac_below_008": round(frac_below, 4),
           "density_006_010": round(near, 4),
           "hist_counts": hist.tolist(),
           "hist_edges": [round(float(e), 3) for e in edges],
           "baseline_thr": BASE, "sweep": sweep}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
