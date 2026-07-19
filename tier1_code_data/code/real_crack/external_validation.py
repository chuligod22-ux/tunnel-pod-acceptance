#!/usr/bin/env python3
"""
external_validation.py — 실제 균열(0.3mm) CNN 검출률 vs 프레임워크 예측 외적검증.

입력:
  cam2_cnn_detection.csv       (프레임 단위 CNN 검출: crack_px, max_comp_len, detected)
  external_validation_table.csv (조건별 차트 0.3mm 예측 + cam2 IQ)
출력:
  cam2_realcrack_condition_summary.csv (조건별 검출률 + 예측 + IQ 병합)
  요약 통계 (콘솔) + figure

핵심: 인쇄 차트 기반 프레임워크가 "0.3mm 검출가능" 예측한 조건 vs 실제 콘크리트
균열의 CNN 검출률 대조 (외적 타당성). 거리·IQ 의존성 포함.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CNN = HERE / "cam2_cnn_detection.csv"
TABLE = HERE / "external_validation_table.csv"
OUT = HERE / "cam2_realcrack_condition_summary.csv"
FIG = HERE / "viz" / "external_validation.png"

COND_DETECT_FRAC = 0.5  # 조건 검출 판정: 프레임 검출비율 ≥ 0.5


def load_cnn():
    """조건별 CNN 집계: 검출률, 평균 crack_px, 평균 max_len, n."""
    by = defaultdict(list)
    with CNN.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by[r["condition"]].append(r)
    agg = {}
    for cond, rows in by.items():
        det = [int(r["detected"]) for r in rows]
        px = [float(r["crack_px"]) for r in rows]
        ln = [float(r["max_comp_len"]) for r in rows]
        agg[cond] = {
            "n_frames": len(rows),
            "det_rate": round(np.mean(det), 3),
            "n_detected": int(sum(det)),
            "mean_crack_px": round(float(np.mean(px)), 1),
            "median_crack_px": round(float(np.median(px)), 1),
            "mean_max_len": round(float(np.mean(ln)), 1),
        }
    return agg


def load_table():
    with TABLE.open(encoding="utf-8") as f:
        return {r["condition"]: r for r in csv.DictReader(f)}


def main():
    if not CNN.exists():
        print(f"[대기] {CNN.name} 아직 없음 — CNN 전체 실행 완료 후 재실행")
        return
    cnn = load_cnn()
    tbl = load_table()

    rows = []
    for cond, t in tbl.items():
        c = cnn.get(cond)
        if not c:
            continue
        cnn_cond_detected = int(c["det_rate"] >= COND_DETECT_FRAC)
        rows.append({
            "condition": cond, "speed_kmh": t["speed_kmh"], "iso": t["iso"],
            "distance_m": float(t["distance_m"]),
            "chart_identifiable": int(t["chart_identifiable"]),
            "chart_0p3_detectable": int(t["chart_0p3_detectable"]),
            "chart_min_width_mm": t["chart_min_width_mm"],
            "cnn_det_rate": c["det_rate"], "cnn_n_detected": c["n_detected"],
            "cnn_n_frames": c["n_frames"], "cnn_cond_detected": cnn_cond_detected,
            "mean_crack_px": c["mean_crack_px"], "mean_max_len": c["mean_max_len"],
            "mtf_h": t["mtf_h"], "bew_h": t["bew_h"], "sigma_motion": t["sigma_motion"],
        })
    rows.sort(key=lambda r: (r["speed_kmh"], r["distance_m"], int(r["iso"])))

    fields = list(rows[0].keys())
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ── 요약 통계 ──────────────────────────────────────────────
    print(f"조건 요약 저장: {OUT} ({len(rows)} 조건)\n")

    # 1) 거리별 검출률
    print("── 거리별 CNN 검출률 ──")
    by_d = defaultdict(list)
    for r in rows:
        by_d[r["distance_m"]].append(r["cnn_det_rate"])
    for d in sorted(by_d):
        v = by_d[d]
        print(f"  {d} m: 평균 검출률 {np.mean(v):.2f}  (조건 {len(v)}개)")

    # 2) ISO별 검출률
    print("\n── ISO별 CNN 검출률 ──")
    by_iso = defaultdict(list)
    for r in rows:
        by_iso[int(r["iso"])].append(r["cnn_det_rate"])
    for iso in sorted(by_iso):
        print(f"  ISO{iso}: 평균 검출률 {np.mean(by_iso[iso]):.2f}")

    # 3) 차트 예측 vs CNN 실측 혼동행렬 (조건 단위, 두 수준 — 논문 GT 기반)
    def confusion(pred_key, label):
        tp = sum(1 for r in rows if r[pred_key] and r["cnn_cond_detected"])
        fn = sum(1 for r in rows if r[pred_key] and not r["cnn_cond_detected"])
        fp = sum(1 for r in rows if not r[pred_key] and r["cnn_cond_detected"])
        tn = sum(1 for r in rows if not r[pred_key] and not r["cnn_cond_detected"])
        agree = (tp + tn) / max(1, len(rows))
        print(f"\n── {label} vs CNN (조건 단위) ──")
        print(f"  TP={tp} FN={fn} FP={fp} TN={tn}  일치율: {agree:.2f}")
        return tp, fn, fp, tn, agree

    confusion("chart_identifiable",
              "① 노출-적정성 수준 (chart_identifiable, 44/6)")
    confusion("chart_0p3_detectable",
              "② 폭 수준 (MDW<=0.3, Table III D(0.3)=31/50)")
    # 실패 6조건의 CNN 검출률 (물리 확증)
    fails = [(r["condition"], r["cnn_det_rate"]) for r in rows
             if not r["chart_identifiable"]]
    print("  expert-flagged-fail 6조건 CNN det_rate:", fails)

    # 4) figure
    make_fig(rows)


def make_fig(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib 없음 — figure 생략)")
        return
    FIG.parent.mkdir(exist_ok=True)
    isos = sorted({int(r["iso"]) for r in rows})
    dists = sorted({r["distance_m"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) 거리 × ISO 검출률 히트맵
    grid = np.full((len(isos), len(dists)), np.nan)
    for r in rows:
        i, j = isos.index(int(r["iso"])), dists.index(r["distance_m"])
        # 두 속도 평균
        vals = [x["cnn_det_rate"] for x in rows
                if int(x["iso"]) == int(r["iso"]) and x["distance_m"] == r["distance_m"]]
        grid[i, j] = np.mean(vals)
    im = ax[0].imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax[0].set_xticks(range(len(dists)), [f"{d}" for d in dists])
    ax[0].set_yticks(range(len(isos)), [f"{i}" for i in isos])
    ax[0].set_xlabel("Distance (m)")
    ax[0].set_ylabel("ISO")
    ax[0].set_title("(a) Real-crack CNN detection rate")
    for i in range(len(isos)):
        for j in range(len(dists)):
            if not np.isnan(grid[i, j]):
                ax[0].text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    # (b) 검출률 vs 거리
    by_d = defaultdict(list)
    for r in rows:
        by_d[r["distance_m"]].append(r["cnn_det_rate"])
    ax[1].plot(sorted(by_d), [np.mean(by_d[d]) for d in sorted(by_d)],
               "o-", color="#2166ac")
    ax[1].set_xlabel("Distance (m)")
    ax[1].set_ylabel("Mean CNN detection rate")
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title("(b) Detectability vs stand-off distance")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    print(f"\nfigure 저장: {FIG}")


if __name__ == "__main__":
    main()
