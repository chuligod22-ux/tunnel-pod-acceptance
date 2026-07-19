#!/usr/bin/env python3
"""
cnn_detect.py — WP1 real-crack 사전학습 CNN(SegFormer-B4) 크랙 세그멘테이션 (R2 대응).

모델: varcoder/segformer-b4-crack-segmentation-dataset (id2label {0:bg, 1:crack}).
8192×5460 대형 프레임은 콘크리트 밴드를 네이티브 해상도 512 타일로 분할해 추론
(밴드 전체 리사이즈 시 0.3mm 균열이 sub-pixel 소실되는 문제 회피).

모드:
  python3 cnn_detect.py --viz-cond crack_d35_ISO1600_V60   # 한 조건 오버레이
  python3 cnn_detect.py --scan-iso 1600                     # 거리×ISO 검출률
  python3 cnn_detect.py                                     # 전체 738프레임 → CSV
"""
import argparse
import csv
import glob
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from detect2 import CONCRETE_BOTTOM, COL_MARGIN, TOP_MARGIN

HERE = Path(__file__).parent
DATA = HERE.parent / "data/raw/crack/cam2"
MANIFEST = HERE / "cam2_real_crack_manifest.csv"
OUT_CSV = HERE / "cam2_cnn_detection.csv"
VIZ = HERE / "viz"
MODEL_ID = "varcoder/segformer-b4-crack-segmentation-dataset"

TILE = 512
STRIDE = 512               # 겹침 없음(속도). 필요시 축소.
CRACK_PX_DETECT = 150      # 프레임 검출 임계: 밴드 내 크랙 픽셀 수 (calibration 대상)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_model = None
_proc = None


def load_model():
    global _model, _proc
    if _model is None:
        _model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(DEVICE).eval()
        _proc = SegformerImageProcessor()
    return _model, _proc


@torch.no_grad()
def crack_mask_tiles(pil_band):
    """밴드 PIL → 크랙 이진 마스크(np.uint8, 밴드 크기). 네이티브 512 타일 배치 추론."""
    model, proc = load_model()
    W, H = pil_band.size
    mask = np.zeros((H, W), np.uint8)
    coords, patches = [], []
    for cy in range(0, max(1, H - 1), STRIDE):
        for cx in range(0, max(1, W - 1), STRIDE):
            x1, y1 = min(cx + TILE, W), min(cy + TILE, H)
            x0, y0 = max(0, x1 - TILE), max(0, y1 - TILE)
            coords.append((x0, y0, x1, y1))
            patches.append(pil_band.crop((x0, y0, x1, y1)))
    # 배치 추론
    B = 8
    for i in range(0, len(patches), B):
        chunk = patches[i:i + B]
        inp = proc(images=chunk, return_tensors="pt").to(DEVICE)
        logits = model(**inp).logits  # (b,C,h,w)
        up = torch.nn.functional.interpolate(
            logits, size=(TILE, TILE), mode="bilinear", align_corners=False)
        preds = up.argmax(1).cpu().numpy().astype(np.uint8)  # (b,512,512)
        for j, (x0, y0, x1, y1) in enumerate(coords[i:i + B]):
            ph, pw = y1 - y0, x1 - x0
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], preds[j][:ph, :pw])
    return mask


def band_box(gray_shape, dist):
    h, w = gray_shape
    r0, r1 = int(h * TOP_MARGIN), int(h * CONCRETE_BOTTOM[dist])
    c0, c1 = int(w * COL_MARGIN[0]), int(w * COL_MARGIN[1])
    return r0, r1, c0, c1


def detect_frame(path, dist):
    """한 프레임 → 크랙 픽셀 수 + 검출여부 + 마스크(밴드) + 오프셋."""
    img = Image.open(path).convert("RGB")
    W, H = img.size
    r0, r1, c0, c1 = band_box((H, W), dist)
    band = img.crop((c0, r0, c1, r1))
    mask = crack_mask_tiles(band)
    # 노이즈 제거: 작은 성분 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    crack_px = int(mask.sum())
    # 최대 연결성분 길이(연속 균열 지표)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    max_len = 0.0
    for i in range(1, n):
        _, _, bw, bh, _ = stats[i]
        max_len = max(max_len, float(np.hypot(bw, bh)))
    detected = int(crack_px >= CRACK_PX_DETECT)
    return {"crack_px": crack_px, "max_comp_len": round(max_len, 1),
            "detected": detected, "mask": mask, "offset": (c0, r0)}


def read_manifest():
    with MANIFEST.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dist_of(cond):
    return int(cond.split("_d")[1][:2]) / 10.0


def cmd_viz_cond(cond):
    VIZ.mkdir(exist_ok=True)
    dist = dist_of(cond)
    fs = sorted(glob.glob(str(DATA / cond / "*.png")))
    panels = []
    ndet = 0
    for i, fp in enumerate(fs):
        res = detect_frame(fp, dist)
        ndet += res["detected"]
        gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        c0, r0 = res["offset"]
        mask = res["mask"]
        # 크랙 마스크 빨강 오버레이
        ys, xs = np.where(mask > 0)
        vis[r0 + ys, c0 + xs] = (0, 0, 255)
        crop = vis[:int(h * (CONCRETE_BOTTOM[dist] + 0.02))]
        small = cv2.resize(crop, (w // 7, crop.shape[0] // 7), interpolation=cv2.INTER_AREA)
        cv2.putText(small, f"f{i} det={res['detected']} px={res['crack_px']} len={res['max_comp_len']}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        panels.append(small)
    cv2.imwrite(str(VIZ / f"cnn_{cond}.png"), np.vstack(panels))
    print(f"{cond}: {ndet}/{len(fs)} 검출 -> cnn_{cond}.png")


def cmd_scan(iso):
    for dist in sorted(CONCRETE_BOTTOM):
        for spd in (60, 80):
            cond = f"crack_d{int(dist*10)}_ISO{iso}_V{spd}"
            fs = sorted(glob.glob(str(DATA / cond / "*.png")))
            if not fs:
                continue
            res = [detect_frame(f, dist) for f in fs]
            nd = sum(r["detected"] for r in res)
            mpx = np.mean([r["crack_px"] for r in res])
            print(f"  d{dist} ISO{iso} V{spd}: {nd:2d}/{len(fs)} 검출  "
                  f"평균 crack_px={mpx:.0f}  평균 max_len={np.mean([r['max_comp_len'] for r in res]):.0f}",
                  flush=True)


def cmd_full():
    rows = read_manifest()
    recs = []
    for r in rows:
        cond = r["condition"]
        dist = float(r["distance_m"])
        for fp in sorted(glob.glob(str(DATA / cond / "*.png"))):
            res = detect_frame(fp, dist)
            recs.append({"condition": cond, "frame": Path(fp).name,
                         "speed_kmh": r["speed_kmh"], "distance_m": r["distance_m"],
                         "iso": r["iso"], "gt_width_mm": r["gt_width_mm"],
                         "crack_px": res["crack_px"], "max_comp_len": res["max_comp_len"],
                         "detected": res["detected"]})
        print(f"  {cond} done ({len(recs)})", file=sys.stderr, flush=True)
    fields = ["condition", "frame", "speed_kmh", "distance_m", "iso", "gt_width_mm",
              "crack_px", "max_comp_len", "detected"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(recs)
    print(f"저장: {OUT_CSV} ({len(recs)} 프레임)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz-cond", default=None)
    ap.add_argument("--scan-iso", default=None)
    args = ap.parse_args()
    print(f"device={DEVICE}", file=sys.stderr)
    if args.viz_cond:
        cmd_viz_cond(args.viz_cond)
    elif args.scan_iso:
        cmd_scan(int(args.scan_iso))
    else:
        cmd_full()


if __name__ == "__main__":
    main()
