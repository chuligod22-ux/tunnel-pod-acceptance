"""Fixed artifact for the under-exposure failure-mode statement of Sec 4.7.

The manuscript states that at 6.5 m stand-off and ISO 100 the mean grey level
of the concrete band is approximately 28 DN out of 255, and that the CNN
returns zero crack pixels in every frame of both speed conditions. This script
pins that statement to a reproducible artifact by computing, for the two
under-exposed conditions (60 and 80 km/h at ISO 100, 6.5 m), the mean 8-bit
grey level of the concrete band of every archived cam2 frame.

The concrete-band geometry is identical to the CNN detection pipeline
(cnn_detect.py / detect2.py): rows from 3 % of the frame height down to the
distance-dependent concrete/tile boundary (79 % at 6.5 m), columns 10-90 %
(vignetting margins excluded).

Input: the cam2 real-crack frame archive (50 conditions, 737 frames), which is
available from the corresponding author on reasonable request (Tier 3); set
CAM2_ROOT below or via the environment variable CAM2_ROOT.

Output: results_json/revision/A7_underexposure_band_brightness.json
(per-frame and per-condition band means; the published run gives condition
means of 28.4 DN at 60 km/h and 30.1 DN at 80 km/h, reported in the
manuscript as 28-30 DN).
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).parent
CAM2_ROOT = Path(os.environ.get(
    "CAM2_ROOT",
    HERE.parent.parent / "data" / "real_crack" / "cam2"))
OUT = HERE.parent.parent / "results_json" / "revision" / "A7_underexposure_band_brightness.json"

# Concrete-band geometry (identical to the CNN detection pipeline)
CONCRETE_BOTTOM = {2.5: 0.37, 3.5: 0.45, 4.5: 0.54, 5.5: 0.65, 6.5: 0.79}
COL_MARGIN = (0.10, 0.90)
TOP_MARGIN = 0.03

CONDITIONS = [  # the two under-exposed expert-flagged-fail conditions at 6.5 m / ISO 100
    ("crack_d65_ISO100_V60", 6.5, 100, 60),
    ("crack_d65_ISO100_V80", 6.5, 100, 80),
]


def band_mean(path, dist):
    img = Image.open(path).convert("L")
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape
    r0, r1 = int(h * TOP_MARGIN), int(h * CONCRETE_BOTTOM[dist])
    c0, c1 = int(w * COL_MARGIN[0]), int(w * COL_MARGIN[1])
    return float(a[r0:r1, c0:c1].mean())


def main():
    out = {"band_geometry": {"top_margin": TOP_MARGIN,
                             "concrete_bottom_at_6p5m": CONCRETE_BOTTOM[6.5],
                             "col_margin": list(COL_MARGIN)},
           "conditions": {}}
    for folder, dist, iso, speed in CONDITIONS:
        d = CAM2_ROOT / folder
        frames = sorted(d.glob("*.png"))
        if not frames:
            raise SystemExit(f"no frames under {d} — set CAM2_ROOT to the cam2 archive")
        means = {f.name: round(band_mean(f, dist), 2) for f in frames}
        vals = np.array(list(means.values()))
        out["conditions"][folder] = {
            "speed_kmh": speed, "iso": iso, "dist_m": dist,
            "n_frames": len(frames),
            "band_mean_DN_mean": round(float(vals.mean()), 2),
            "band_mean_DN_min": round(float(vals.min()), 2),
            "band_mean_DN_max": round(float(vals.max()), 2),
            "per_frame": means,
        }
    all_vals = [c["band_mean_DN_mean"] for c in out["conditions"].values()]
    out["summary"] = {
        "statement": "mean grey level of the concrete band at 6.5 m / ISO 100 is approximately 28 DN out of 255",
        "condition_means_DN": all_vals,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("written", OUT)
    for k, c in out["conditions"].items():
        print(f"{k}: n={c['n_frames']} band mean {c['band_mean_DN_mean']} DN "
              f"[{c['band_mean_DN_min']}, {c['band_mean_DN_max']}]")


if __name__ == "__main__":
    main()
