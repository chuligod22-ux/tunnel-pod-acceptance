#!/usr/bin/env python3
"""
wp2_borderline_figure.py — R1-2 "borderline case 예시" 그림 생성 (v2: 클린 3-panel).

(a) 명확 가시 (60/200/2.5, MDW 0.1) / (b) 경계 사례 (60/1600/3.5, 부분 과노출 —
상단 워시아웃·하단 가시, expert-revised MDW 0.7) / (c) 실패 (60/100/6.5,
expert-flagged-fail, 저노출). 자동 패치 박스는 스퓨리어스가 많아 제외 —
원시 프레임의 시각적 대비가 자명하므로 설명은 캡션에 위임.

출력: revision/viz/wp2_borderline_cases.png
"""
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
CAM1 = HERE.parent / "data/raw/crack/cam1"
OUT = HERE / "viz/wp2_borderline_cases.png"

# (파일, 타이틀 2줄 — ASCII만: cv2.putText는 non-ASCII 미지원)
PANELS = [
    ("cam1_v60_iso200_d25.png",
     ["(a) Clearly visible", "60 km/h, ISO 200, 2.5 m  |  MDW 0.1 mm"]),
    ("cam1_v60_iso1600_d35.png",
     ["(b) Borderline: partial over-exposure",
      "60 km/h, ISO 1600, 3.5 m  |  MDW 0.7 mm (expert-revised)"]),
    ("cam1_v60_iso100_d65_x.png",
     ["(c) Expert-flagged-fail: under-exposure",
      "60 km/h, ISO 100, 6.5 m  |  non-identifiable"]),
]
TARGET_W = 1400


def main():
    """3-panel 생성: 이미지 + 하단 중앙 서브캡션 (논문 양식)."""
    OUT.parent.mkdir(exist_ok=True)
    font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2
    panels = []
    for fname, title in PANELS:
        g = cv2.imread(str(CAM1 / fname), cv2.IMREAD_GRAYSCALE)
        scale = TARGET_W / g.shape[1]
        gr = cv2.resize(g, (TARGET_W, int(g.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA)
        vis = cv2.cvtColor(gr, cv2.COLOR_GRAY2BGR)
        # 서브캡션: 패널 하단 중앙
        band = np.full((104, TARGET_W, 3), 255, np.uint8)
        for i, line in enumerate(title):
            (tw, _), _ = cv2.getTextSize(line, font, fscale, thick)
            cv2.putText(band, line, ((TARGET_W - tw) // 2, 42 + i * 42),
                        font, fscale, (0, 0, 0), thick, cv2.LINE_AA)
        sep = np.full((6, TARGET_W, 3), 90, np.uint8)
        panels.append(np.vstack([vis, band, sep]))
        print(f"  {fname}: mean={g.mean():.0f} DN")
    canvas = np.vstack(panels)
    cv2.imwrite(str(OUT), canvas)
    print(f"저장: {OUT}  ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
