"""
Fig. 11 — Reference-Crack Visibility Determination Workflow Diagram (Sec 3.4)
Each stage has its own visual element placed alongside the text box:
  - Stage 1: cam1 raw frame (board detect) + chart schematic (10 patches)
  - Stage 2: label icons (3 classes with colour codes)
  - Stage 3: D(a) hit/miss matrix schema (with example MDW)
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

OUT = "/Users/lch/home/code/tunnelscanning/01_tunnelscanning/01_paper/output/ieee_tim_v1/figures/F12_detection_workflow"
RAW = "/Users/lch/home/code/tunnelscanning/01_tunnelscanning/03_src/data/raw/crack/cam1/cam1_v60_iso800_d35.png"

# -------------------------------------------------------------------------
# Load example raw image and detect chart board
# -------------------------------------------------------------------------
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

# Patch detection within board (for (b) ROI overlay)
# Mirrors the actual analyze_crack_detectability.py pipeline.
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
        # Skip the board outline itself (occupies most of the cropped image)
        if w * h > 0.5 * board_area:
            continue
        aspect = w / max(h, 1)
        if 0.4 < aspect < 2.5:
            patches.append((bx + x, by + y, w, h, w * h))
patches.sort(key=lambda p: p[4], reverse=True)
patches_algo = [p[:4] for p in patches[:9]]  # Stage 1 algorithm detects up to 9

# Stage 2 expert-added 10th patch — geometric extrapolation from row spacing.
# CRACK_GRID has 3+3+4 = 10 specimens; in this representative frame the
# automated pipeline misses one because its mask connects to the wall.
# Place the 10th at the symmetric position in the row with fewest detections.
def cluster_by_y(boxes, gap=200):
    rows = []
    for b in sorted(boxes, key=lambda p: p[1]):
        if not rows or b[1] - rows[-1][-1][1] > gap:
            rows.append([b])
        else:
            rows[-1].append(b)
    return rows
rows = cluster_by_y(patches_algo)
row_lens = [len(r) for r in rows]
patch_expert = None
# Find a row that is exactly one short of the longest row.
# Iso800_d35 has rows 4-3-2 detected; the row missing exactly one specimen
# (mid-row, 3 of 4) corresponds to the patch the wall-connection ate during
# Stage 1 — that is the genuine 10th specimen recovered in Stage 2.
if rows:
    target_len = max(row_lens) - 1
    candidate_rows = [r for r in rows if len(r) == target_len]
    long_row = rows[row_lens.index(max(row_lens))]
    long_xs = sorted([b[0] for b in long_row])
    for short_row in candidate_rows:
        short_xs = sorted([b[0] for b in short_row])
        for lx in long_xs:
            if not any(abs(lx - sx) < 400 for sx in short_xs):
                ref = short_row[0]
                patch_expert = (lx, ref[1], ref[2], ref[3])
                break
        if patch_expert:
            break

# Chart-board zoom for (b)
DS_CHART = 4
if board_box is not None:
    bx, by, bw, bh = board_box
    chart_zoom = img_full[by:by + bh, bx:bx + bw][::DS_CHART, ::DS_CHART]
else:
    chart_zoom = img_full[: int(H * 0.6), int(W * 0.15):int(W * 0.85)][::DS_CHART, ::DS_CHART]

# -------------------------------------------------------------------------
# Figure layout — left column: text diagram; right column: per-stage visuals
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 12), dpi=150)

# Single full-width diagram axis. Each stage box now spans most of the
# figure width and the per-stage visuals sit inside the box's right half,
# so the colour-coded box visually groups text + illustration together.
ax_d = fig.add_axes([0.02, 0.02, 0.96, 0.96])
ax_d.set_xlim(0, 10)
ax_d.set_ylim(4, 24)
ax_d.axis("off")


def shaded_box(ax, x, y, w, h, fc, ec="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                fc=fc, ec=ec, linewidth=1.4))


def left_text(ax, x, y, text, fontsize=10, fw="normal"):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=fontsize, fontweight=fw)


def centred_text(ax, x, y, text, fontsize=10, fw="normal"):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fw)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="->", mutation_scale=18,
                                 lw=1.5, color="black"))


# Title
ax_d.text(5, 23.5, "Reference-Crack Visibility Determination Workflow",
          ha="center", fontsize=15, fontweight="bold")

# Input box (narrow, centred)
shaded_box(ax_d, 3, 21.4, 4, 1.1, "#e8e8e8")
centred_text(ax_d, 5, 21.95,
             "Raw inspection frames",
             fontsize=12, fw="bold")
arrow(ax_d, 5, 21.4, 5, 20.65)

# Stage 1 box  (text on left half; (a)+(b) embedded on right half)
shaded_box(ax_d, 0.4, 15.65, 8.6, 5.0, "#d4f1d4")
left_text(ax_d, 0.7, 18.15,
          "Stage 1 — Automated CV Pipeline\n\n"
          "$\\cdot$ Board detection (intensity threshold + morphology)\n"
          "$\\cdot$ 10-patch identification (intensity range + aspect ratio)\n"
          "$\\cdot$ Per-patch visibility score $s$ — Eq. (26)\n"
          "$\\cdot$ Threshold $s \\geq 0.08$ $\\rightarrow$ \"visible\"\n\n"
          "Output: per-condition (visible-widths list, MDW)",
          fontsize=12)
arrow(ax_d, 5, 15.65, 5, 14.9)

# Stage 2 box  (taller to accommodate larger thumbnails)
shaded_box(ax_d, 0.4, 10.4, 8.6, 4.5, "#d4e3f1")
left_text(ax_d, 0.7, 12.65,
          "Stage 2 — Domain-Expert Validation\n\n"
          "Each condition assigned to one of three classes:\n"
          "$\\cdot$ algorithm-confirmed\n"
          "$\\cdot$ expert-verified\n"
          "$\\cdot$ expert-flagged-fail",
          fontsize=12)
arrow(ax_d, 5, 10.4, 5, 9.65)

# Stage 3 box  (text on left half; (d) D(a) matrix embedded on right half)
shaded_box(ax_d, 0.4, 6.45, 8.6, 3.2, "#f1e3d4")
left_text(ax_d, 0.7, 8.05,
          "Stage 3 — Monotonic Completion (Section 2.4)\n\n"
          "$D(a) = 1$ for $a \\geq$ MDW;\n"
          "$D(a) = 0$ for $a <$ MDW\n\n"
          "(NDT POD standard, MIL-HDBK-1823 [12], [13])",
          fontsize=12)
arrow(ax_d, 5, 6.45, 5, 5.7)

# Output box
shaded_box(ax_d, 1.5, 4.6, 7.0, 1.1, "#e8e8e8")
centred_text(ax_d, 5, 5.15,
             "500-row hit/miss dataset $\\rightarrow$ POD analysis",
             fontsize=12, fw="bold")


# Helper to convert diagram coords (xlim 0-10, ylim 4-24) into figure coords
# given the diagram axis position.
_AX_X0, _AX_Y0, _AX_W, _AX_H = 0.02, 0.02, 0.96, 0.96
def _fx(x_d):
    return _AX_X0 + (x_d / 10.0) * _AX_W
def _fy(y_d):
    return _AX_Y0 + ((y_d - 4) / 20.0) * _AX_H

# -------------------------------------------------------------------------
# Stage 1 visuals — split into raw image + chart schematic
# -------------------------------------------------------------------------
# Stage 1 visuals: (a) and (b) embedded inside Stage 1 box's right half
# Stage 1 box diagram coords: x=0.4-9.0, y=15.65-20.65; reserve x=4.7-8.9 for visuals.
_S1_THUMB_W = 0.18
_S1_THUMB_H = 0.225
_S1_Y = (_fy(15.65) + _fy(20.65)) / 2 - _S1_THUMB_H / 2  # vertical centre of box
_S1_AX = _fx(4.7); _S1_BX = _fx(7.0)  # left edges of (a) and (b)
ax_v1a = fig.add_axes([_S1_AX, _S1_Y, _S1_THUMB_W, _S1_THUMB_H])
ax_v1a.imshow(img_display, cmap="gray", vmin=0, vmax=255)
if board_box is not None:
    bx0, by0, bw0, bh0 = board_box
    rect = Rectangle((bx0 / DS, by0 / DS), bw0 / DS, bh0 / DS,
                     fill=False, edgecolor="red", linewidth=1.4)
    ax_v1a.add_patch(rect)
ax_v1a.set_xticks([]); ax_v1a.set_yticks([])
ax_v1a.text(0.5, -0.06,
            "(a) Example cam1 raw frame\n(red: detected chart board)",
            transform=ax_v1a.transAxes, ha="center", va="top", fontsize=10)

# (b) chart-board zoom with detected patch ROIs
# green = Stage 1 algorithm-detected; orange dashed = Stage 2 expert-added
ax_v1b = fig.add_axes([_S1_BX, _S1_Y, _S1_THUMB_W, _S1_THUMB_H])
ax_v1b.imshow(chart_zoom, cmap="gray", vmin=0, vmax=255)
if board_box is not None:
    bx0, by0, _, _ = board_box
    for (px, py, pw, ph) in patches_algo:
        rect = Rectangle(((px - bx0) / DS_CHART, (py - by0) / DS_CHART),
                         pw / DS_CHART, ph / DS_CHART,
                         fill=False, edgecolor="lime", linewidth=1.5)
        ax_v1b.add_patch(rect)
    if patch_expert is not None:
        px, py, pw, ph = patch_expert
        rect = Rectangle(((px - bx0) / DS_CHART, (py - by0) / DS_CHART),
                         pw / DS_CHART, ph / DS_CHART,
                         fill=False, edgecolor="orange", linewidth=1.8,
                         linestyle="--")
        ax_v1b.add_patch(rect)
ax_v1b.set_xticks([]); ax_v1b.set_yticks([])
ax_v1b.text(0.5, -0.06,
            "(b) Chart-board zoom\n(green: Stage 1; orange: Stage 2 expert)",
            transform=ax_v1b.transAxes, ha="center", va="top", fontsize=10)

# -------------------------------------------------------------------------
# Stage 2 visual — representative frames per class with counts
# Counts derived from cam1_crack_detectability.csv (n=50):
#   identifiable=N  -> expert-flagged-fail
#   identifiable=Y, n_sq>=10 -> algorithm-confirmed (proxy)
#   identifiable=Y, n_sq<10  -> expert-verified (proxy)
# -------------------------------------------------------------------------
import csv as _csv, re as _re
_CSV = "/Users/lch/home/code/tunnelscanning/01_tunnelscanning/03_src/data/cam1_crack_detectability.csv"
_counts = {"algorithm-confirmed": 0, "expert-verified": 0, "expert-flagged-fail": 0}
with open(_CSV) as _f:
    for _row in _csv.DictReader(_f):
        _m = _re.search(r"n_sq=(\d+)", _row.get("notes", ""))
        _n = int(_m.group(1)) if _m else None
        if _row.get("identifiable") == "N":
            _counts["expert-flagged-fail"] += 1
        elif _n is not None and _n >= 10:
            _counts["algorithm-confirmed"] += 1
        else:
            _counts["expert-verified"] += 1
_total = sum(_counts.values())

# All three thumbnails share d = 3.5 m where possible so the chart appears
# at the same physical scale; expert-verified reuses the (a)/(b) frame to
# anchor the reader's attention back to the workflow example.
CLASSES = [
    ("algorithm-confirmed",  "#7ed47e",
     "/Users/lch/home/code/tunnelscanning/01_tunnelscanning/03_src/data/raw/crack/cam1/cam1_v60_iso200_d35.png"),
    ("expert-verified",      "#7ea3d4",
     RAW),  # same frame as (a)/(b) — the orange dashed ROI in (b) is the
            # expert-added 10th patch that justifies this label
    ("expert-flagged-fail",  "#d47e7e",
     "/Users/lch/home/code/tunnelscanning/01_tunnelscanning/03_src/data/raw/crack/cam1/cam1_v60_iso1600_d25_x.png"),
]


def _board_crop(path, target_aspect=1.5, out_w=600, pad_value=255):
    """Extract chart-board region and PAD (never crop) to a fixed aspect ratio
    so all (c) thumbnails are uniformly sized while showing the full chart.
    Padding uses pad_value (default = 255, matching the chart's white background).
    """
    g = np.array(Image.open(path).convert("L"))
    h_, w_ = g.shape
    _, br = cv2.threshold(g[: int(h_ * 0.75), :], 180, 255, cv2.THRESH_BINARY)
    k = np.ones((15, 15), np.uint8)
    br = cv2.morphologyEx(br, cv2.MORPH_CLOSE, k)
    br = cv2.morphologyEx(br, cv2.MORPH_OPEN, k)
    c, _ = cv2.findContours(br, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if c:
        largest = max(c, key=cv2.contourArea)
        if cv2.contourArea(largest) >= w_ * h_ * 0.02:
            x, y, w, h = cv2.boundingRect(largest)
            crop = g[y:y + h, x:x + w]
        else:
            crop = g[: int(h_ * 0.6), int(w_ * 0.15):int(w_ * 0.85)]
    else:
        crop = g[: int(h_ * 0.6), int(w_ * 0.15):int(w_ * 0.85)]
    # Pad shorter dimension to reach target aspect (no content removed)
    ch, cw = crop.shape
    cur_aspect = cw / max(ch, 1)
    if cur_aspect > target_aspect:
        new_h = int(round(cw / target_aspect))
        pad_top = (new_h - ch) // 2
        pad_bot = new_h - ch - pad_top
        crop = np.pad(crop, ((pad_top, pad_bot), (0, 0)),
                      constant_values=pad_value)
    elif cur_aspect < target_aspect:
        new_w = int(round(ch * target_aspect))
        pad_left = (new_w - cw) // 2
        pad_right = new_w - cw - pad_left
        crop = np.pad(crop, ((0, 0), (pad_left, pad_right)),
                      constant_values=pad_value)
    out_h = int(round(out_w / target_aspect))
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)


# Three thumbnail axes embedded inside Stage 2 box's right half
# Stage 2 box: x=0.4-9.0, y=10.4-14.9; reserve x=4.7-8.9 for visuals.
_thumb_w = 0.135
_thumb_h = 0.16
_gap = 0.006
# Centre the row of thumbnails vertically inside Stage 2 box (with caption below)
_caption_h = 0.025
_block_h = _thumb_h + _caption_h
_S2_CY = (_fy(10.4) + _fy(14.9)) / 2
_thumb_y = _S2_CY - _block_h / 2 + _caption_h
# Centre the row horizontally in the visual region (x=4.7..8.9)
_S2_VIS_X0 = _fx(4.7)
_S2_VIS_W = _fx(8.9) - _fx(4.7)
_total_thumbs_w = 3 * _thumb_w + 2 * _gap
_x0 = _S2_VIS_X0 + (_S2_VIS_W - _total_thumbs_w) / 2
for i, (name, col, path) in enumerate(CLASSES):
    ax_t = fig.add_axes([_x0 + i * (_thumb_w + _gap), _thumb_y,
                         _thumb_w, _thumb_h])
    ax_t.imshow(_board_crop(path, target_aspect=1.0, out_w=600),
                cmap="gray", vmin=0, vmax=255, aspect="auto")
    for spine in ax_t.spines.values():
        spine.set_edgecolor(col)
        spine.set_linewidth(2.5)
    ax_t.set_xticks([]); ax_t.set_yticks([])
    ax_t.text(0.5, -0.06, name,
              transform=ax_t.transAxes, ha="center", va="top",
              fontsize=9.5, color="black")

# (c) caption at the bottom-centre of the row of thumbnails
fig.text(_S2_VIS_X0 + _S2_VIS_W / 2,
         _thumb_y - 0.024,
         "(c) Stage 2 class examples (centre frame = the (a)/(b) example)",
         ha="center", va="top", fontsize=10)

# -------------------------------------------------------------------------
# Stage 3 visual — D(a) matrix (example MDW = 0.3 mm)
# -------------------------------------------------------------------------
# (d) D(a) matrix embedded inside Stage 3 box's right half
# Stage 3 box: x=0.4-9.0, y=6.45-9.65; reserve x=4.7-8.9 for visuals.
_S3_VIS_X0 = _fx(4.7)
_S3_VIS_W = _fx(8.9) - _fx(4.7)
_S3_VIS_H = 0.115
_S3_CY = (_fy(6.45) + _fy(9.65)) / 2
_S3_VIS_Y = _S3_CY - _S3_VIS_H / 2 + 0.012
ax_v3 = fig.add_axes([_S3_VIS_X0, _S3_VIS_Y, _S3_VIS_W, _S3_VIS_H])
ax_v3.set_xlim(0, 11)
ax_v3.set_ylim(0, 4)
ax_v3.axis("off")
fig.text(_S3_VIS_X0 + _S3_VIS_W / 2, _S3_VIS_Y - 0.005,
         "(d) Stage 3 example: $D(a)$ for one condition with MDW = 0.3 mm",
         ha="center", va="top", fontsize=10)

WIDTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MDW_EX = 0.3

ax_v3.text(0.0, 2.7, "$a$ (mm):", ha="left", va="center",
           fontsize=11)
ax_v3.text(0.0, 1.2, "$D(a)$:", ha="left", va="center",
           fontsize=11)

for i, a in enumerate(WIDTHS):
    cx = 1.4 + i * 0.95
    # Width label row
    ax_v3.text(cx, 2.7, f"{a}", ha="center", va="center",
               fontsize=11)
    # D(a) cell
    d_val = 1 if a >= MDW_EX else 0
    cell_color = "#7ed47e" if d_val == 1 else "#d47e7e"
    cell = Rectangle((cx - 0.4, 0.65), 0.8, 1.1,
                     fc=cell_color, ec="black", linewidth=0.8)
    ax_v3.add_patch(cell)
    ax_v3.text(cx, 1.2, str(d_val), ha="center", va="center",
               fontsize=12)

# MDW marker (vertical line at a = MDW)
mdw_x = 1.4 + WIDTHS.index(MDW_EX) * 0.95 - 0.5
ax_v3.plot([mdw_x, mdw_x], [0.5, 3.4], color="black",
           linestyle="--", linewidth=1.0)
ax_v3.text(mdw_x, 3.55, "MDW", ha="center", va="bottom", fontsize=9.5)

# Save
plt.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
plt.savefig(OUT + ".pdf", bbox_inches="tight")
plt.close()

print(f"Saved: {OUT}.png")
