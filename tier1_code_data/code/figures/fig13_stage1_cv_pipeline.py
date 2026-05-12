"""
Fig. 13 — Stage 1 CV Pipeline Detail (Sec 3.4)

Horizontal 4-stage flowchart with per-stage mini visuals (Option B):
  (a) Chart Board Localization  — raw frame + red bounding box
  (b) Patch Grid Recovery       — chart-board zoom + 10 patch overlays (green)
  (c) Per-Patch Scoring (Eq. 25) — single demo patch with Canny edges (red)
                                    + Hough line segments (green) overlay
  (d) Visibility Threshold      — bar chart of 10 patch s values
                                    with horizontal line at s = 0.08

Layout: stages flow left → right, each column = (text box, mini visual).
Distinct from Fig. 12 (vertical 3-stage workflow).
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

OUT = (
    "/Users/lch/home/code/tunnelscanning/01_tunnelscanning/01_paper/"
    "output/ieee_tim_v1/figures/F13_stage1_cv_pipeline"
)
RAW = (
    "/Users/lch/home/code/tunnelscanning/01_tunnelscanning/03_src/data/"
    "raw/crack/cam1/cam1_v60_iso800_d35.png"
)


# ============================================================
# Stage 1: chart board localization
# ============================================================
img_full = np.array(Image.open(RAW).convert("L"))
H, W = img_full.shape

roi_top = img_full[: int(H * 0.75), :]
_, bright = cv2.threshold(roi_top, 180, 255, cv2.THRESH_BINARY)
kernel15 = np.ones((15, 15), np.uint8)
bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel15)
bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel15)
contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
board_box = None
if contours:
    largest = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(largest)
    board_box = (bx, by, bw, bh)

DS = 16
img_display = img_full[::DS, ::DS]


# ============================================================
# Stage 2: patch grid recovery
# ============================================================
patches = []
if board_box is not None:
    bx, by, bw, bh = board_box
    board_gray = img_full[by:by + bh, bx:bx + bw]
    mean_val = float(np.mean(board_gray))
    lo = max(0, int(mean_val) - 60)
    hi = min(255, int(mean_val) + 30)
    mask = cv2.inRange(board_gray, lo, hi)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    board_area = bw * bh
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 0.5 * board_area:
            continue
        aspect = w / max(h, 1)
        if 0.4 < aspect < 2.5:
            patches.append((bx + x, by + y, w, h, w * h))
patches.sort(key=lambda p: p[4], reverse=True)
patches_algo = [p[:4] for p in patches[:10]]

DS_CHART = 4
if board_box is not None:
    bx, by, bw, bh = board_box
    chart_zoom = img_full[by:by + bh, bx:bx + bw][::DS_CHART, ::DS_CHART]
else:
    chart_zoom = img_full[: int(H * 0.6), int(W * 0.15):int(W * 0.85)][::DS_CHART, ::DS_CHART]


# ============================================================
# Stage 3 helper: per-patch score (Eq. 25)
# ============================================================
def score_patch(patch_img):
    """Mirrors the Eq.(25) implementation in analyse_crack_detectability.py."""
    h, w = patch_img.shape
    mask_inner = np.ones_like(patch_img, dtype=np.uint8)
    if h > 10 and w > 10:
        mask_inner[:5, :] = 0
        mask_inner[-5:, :] = 0
        mask_inner[:, :5] = 0
        mask_inner[:, -5:] = 0
    blur = cv2.GaussianBlur(patch_img, (3, 3), 0)
    canny = cv2.Canny(blur, 20, 80)
    canny_in = (canny * mask_inner).astype(np.uint8)
    inner_pixels = max(1, int(mask_inner.sum()))
    e_canny = float(canny_in.sum() / 255) / inner_pixels
    min_line = max(5, int(min(h, w) / 4))
    lines = cv2.HoughLinesP(
        canny_in, rho=1, theta=np.pi / 180,
        threshold=8, minLineLength=min_line, maxLineGap=5,
    )
    sum_l = 0.0
    line_segments = []
    if lines is not None:
        for L in lines:
            x1, y1, x2, y2 = L[0]
            d = float(np.hypot(x2 - x1, y2 - y1))
            sum_l += d
            line_segments.append(((x1, y1), (x2, y2)))
    s = min(
        1.0,
        0.4 * (100 * e_canny)
        + 0.6 * min(1.0, sum_l / (2 * min(h, w))),
    )
    return s, e_canny, sum_l, canny_in, line_segments


if patches_algo:
    px_d, py_d, pw_d, ph_d = patches_algo[0]
    demo_patch = img_full[py_d:py_d + ph_d, px_d:px_d + pw_d].copy()
    s_demo, e_demo, sl_demo, canny_demo, lines_demo = score_patch(demo_patch)
else:
    demo_patch = np.zeros((100, 100), dtype=np.uint8)
    canny_demo = np.zeros_like(demo_patch)
    lines_demo = []
    s_demo, e_demo, sl_demo = 0.0, 0.0, 0.0


# ============================================================
# Stage 4: per-patch scores for all detected patches
# ============================================================
def cluster_by_y(boxes, gap=200):
    rows = []
    for b in sorted(boxes, key=lambda p: p[1]):
        if not rows or b[1] - rows[-1][-1][1] > gap:
            rows.append([b])
        else:
            rows[-1].append(b)
    return rows

rows = cluster_by_y(patches_algo)
ordered_patches = []
for row in rows:
    ordered_patches.extend(sorted(row, key=lambda p: p[0]))

patch_scores = []
for (pxi, pyi, pwi, phi) in ordered_patches[:10]:
    pimg = img_full[pyi:pyi + phi, pxi:pxi + pwi].copy()
    s_i, _, _, _, _ = score_patch(pimg)
    patch_scores.append(s_i)
while len(patch_scores) < 10:
    patch_scores.append(0.0)


# ============================================================
# Horizontal layout
# ============================================================
fig = plt.figure(figsize=(18, 8.0), dpi=150)
ax_d = fig.add_axes([0.01, 0.01, 0.98, 0.98])
ax_d.set_xlim(0, 40)
ax_d.set_ylim(0, 16)
ax_d.axis("off")


def shaded_box(ax, x, y, w, h, fc, ec="black"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.10",
        fc=fc, ec=ec, linewidth=1.4))


def left_text(ax, x, y, text, fontsize=10, fw="normal"):
    ax.text(x, y, text, ha="left", va="top",
            fontsize=fontsize, fontweight=fw)


def centred_text(ax, x, y, text, fontsize=10, fw="normal"):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fw)


def harrow(ax, x1, x2, y):
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle="-|>", mutation_scale=22,
        lw=2.0, color="black", zorder=20))


# Title (top)
ax_d.text(20, 15.3, "Stage 1 — Computer-Vision Pipeline Detail",
          ha="center", fontsize=15, fontweight="bold")

# Input label (top-left of stage 1)
ax_d.text(0.4, 14.2, "Input:  cam1 raw frame",
          ha="left", va="center", fontsize=11, fontweight="bold",
          color="#333")

# Output label (bottom-right)
ax_d.text(39.6, 0.7,
          "Output:  visible-widths list,  MDW = min(visible widths)",
          ha="right", va="center", fontsize=11, fontweight="bold",
          color="#333")


# Stage box geometry
COL_W = 8.6
COL_H = 12.0
COL_Y = 1.6
COL_GAP = 1.8
COL_X = [0.4, 0.4 + (COL_W + COL_GAP), 0.4 + 2 * (COL_W + COL_GAP),
         0.4 + 3 * (COL_W + COL_GAP)]

STAGE_COLOURS = ["#d4f1d4", "#d4e3f1", "#f1e3d4", "#f1d4e3"]
STAGE_TITLES = [
    "(a)  Chart Board\nLocalization",
    "(b)  Patch Grid\nRecovery",
    "(c)  Per-Patch Scoring\n(Eq. 25)",
    "(d)  Visibility\nThreshold",
]
STAGE_TEXTS = [
    "$\\cdot$ Intensity threshold:\n      DN $\\leq$ 180\n"
    "$\\cdot$ Morphology closing\n      + opening (15$\\times$15)\n"
    "$\\cdot$ Largest connected\n      component\n\n"
    "$\\rightarrow$ chart bounding box",
    "$\\cdot$ Intensity mask:\n      [mean$-$60, mean$+$30] DN\n"
    "$\\cdot$ Aspect ratio $\\in$ [0.4, 2.5]\n"
    "$\\cdot$ y-cluster (gap $\\leq$ 30 px)\n      $\\rightarrow$ rows\n"
    "$\\cdot$ x-order $\\rightarrow$ cols\n\n"
    "$\\rightarrow$ 10 patches indexed",
    "Canny: G(3$\\times$3),\n"
    "  thresholds 20 / 80,\n"
    "  5-px border excl.\n"
    "  $\\rightarrow e_\\mathrm{Canny}$\n"
    "Hough P:\n"
    "  threshold = 8,\n"
    "  minLine = max(5,\n"
    "    min(h,w)/4),\n"
    "  maxGap = 5\n"
    "  $\\rightarrow \\sum \\ell_i$",
    "$s = \\min\\{1,$\n"
    "  $\\;\\;\\;0.4\\,(100\\,e_\\mathrm{Canny})$\n"
    "  $+ 0.6\\,\\min(1,$\n"
    "  $\\;\\;\\;\\sum\\ell_i / (2\\min(h,w))) \\}$\n\n"
    "$s \\geq 0.08 \\rightarrow$ \"visible\"\n"
    "(threshold fixed pre-\n"
    " validation, applied\n"
    " uniformly)",
]

# Draw 4 stage boxes
for i in range(4):
    x = COL_X[i]
    shaded_box(ax_d, x, COL_Y, COL_W, COL_H, STAGE_COLOURS[i])
    # Title block (top of column)
    ax_d.text(x + COL_W / 2, COL_Y + COL_H - 0.55,
              STAGE_TITLES[i],
              ha="center", va="top", fontsize=12, fontweight="bold")
    # Detail text (mid-upper portion)
    left_text(ax_d, x + 0.45, COL_Y + COL_H - 2.2,
              STAGE_TEXTS[i], fontsize=9.6)

# Horizontal arrows between stages (in the gap region)
for i in range(3):
    x_left = COL_X[i] + COL_W
    x_right = COL_X[i + 1]
    harrow(ax_d, x_left + 0.15, x_right - 0.15, COL_Y + COL_H / 2)


# ============================================================
# Mini-visuals at the bottom of each stage column
# (image axes via fig.add_axes for natural raster rendering)
# ============================================================
_AX_X0, _AX_Y0, _AX_W, _AX_H = 0.01, 0.01, 0.98, 0.98
def _fx(x_d):
    return _AX_X0 + (x_d / 40.0) * _AX_W
def _fy(y_d):
    return _AX_Y0 + (y_d / 16.0) * _AX_H


VIS_Y0_d = COL_Y + 0.45      # bottom margin of visual
VIS_H_d = 4.2                # height in diagram coords
VIS_W_d = 7.0                # width in diagram coords


def _add_vis_axes(col_idx):
    cx_d = COL_X[col_idx] + COL_W / 2
    x0 = _fx(cx_d - VIS_W_d / 2)
    y0 = _fy(VIS_Y0_d)
    w = _fx(cx_d + VIS_W_d / 2) - _fx(cx_d - VIS_W_d / 2)
    h = _fy(VIS_Y0_d + VIS_H_d) - _fy(VIS_Y0_d)
    return fig.add_axes([x0, y0, w, h])


# (a) Stage 1: raw frame + bbox
ax_v1 = _add_vis_axes(0)
ax_v1.imshow(img_display, cmap="gray", vmin=0, vmax=255, aspect="auto")
if board_box is not None:
    bx0, by0, bw0, bh0 = board_box
    rect = Rectangle((bx0 / DS, by0 / DS), bw0 / DS, bh0 / DS,
                     fill=False, edgecolor="red", linewidth=1.6)
    ax_v1.add_patch(rect)
ax_v1.set_xticks([])
ax_v1.set_yticks([])

# (b) Stage 2: chart zoom + patches
ax_v2 = _add_vis_axes(1)
ax_v2.imshow(chart_zoom, cmap="gray", vmin=0, vmax=255, aspect="auto")
if board_box is not None:
    bx0, by0, _, _ = board_box
    for (pxi, pyi, pwi, phi) in patches_algo:
        rect = Rectangle(
            ((pxi - bx0) / DS_CHART, (pyi - by0) / DS_CHART),
            pwi / DS_CHART, phi / DS_CHART,
            fill=False, edgecolor="lime", linewidth=1.5,
        )
        ax_v2.add_patch(rect)
ax_v2.set_xticks([])
ax_v2.set_yticks([])

# (c) Stage 3: demo patch + Canny + Hough
ax_v3 = _add_vis_axes(2)
ax_v3.imshow(demo_patch, cmap="gray", vmin=0, vmax=255, aspect="auto")
canny_rgba = np.zeros((*canny_demo.shape, 4), dtype=np.uint8)
canny_rgba[canny_demo > 0] = [255, 0, 0, 220]
ax_v3.imshow(canny_rgba, aspect="auto")
for ((x1, y1), (x2, y2)) in lines_demo:
    ax_v3.plot([x1, x2], [y1, y2], color="lime", linewidth=1.2)
ax_v3.set_xticks([])
ax_v3.set_yticks([])
ax_v3.text(
    0.5, -0.06,
    f"$s$ = {s_demo:.3f}    "
    f"($e_\\mathrm{{Canny}}$ = {e_demo:.4f}, "
    f"$\\sum\\ell_i$ = {sl_demo:.0f})",
    transform=ax_v3.transAxes, ha="center", va="top", fontsize=8.5,
)

# (d) Stage 4: bar chart of patch scores + threshold line
ax_v4 = _add_vis_axes(3)
xs = np.arange(1, len(patch_scores) + 1)
colors = ["#7ed47e" if s >= 0.08 else "#d47e7e" for s in patch_scores]
ax_v4.bar(xs, patch_scores, color=colors, edgecolor="black", linewidth=0.5)
ax_v4.axhline(y=0.08, color="black", linestyle="--", linewidth=1.0)
ax_v4.text(
    len(patch_scores) + 0.45, 0.08, "0.08",
    fontsize=8, va="center", ha="left",
)
ax_v4.set_ylabel("$s$", fontsize=9)
ax_v4.set_xlabel("patch index", fontsize=9)
ax_v4.tick_params(axis="both", labelsize=8)
ax_v4.set_xticks(xs)
ax_v4.set_xlim(0.4, len(patch_scores) + 1.6)


# Save
plt.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
plt.savefig(OUT + ".pdf", bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}.png")
print(f"Saved: {OUT}.pdf")
