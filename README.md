# Supplementary Material

**Paper**: *A POD-Based Image Quality Acceptance Framework for Crack Detection in High-Speed Mobile Tunnel Inspection*

**Authors**: Chulhee Lee, Donggyou Kim, Dongku Kim, Junbeom An
**Affiliation**: Korea Institute of Civil Engineering and Building Technology (KICT)
**Target Journal**: IEEE Transactions on Instrumentation and Measurement (IEEE TIM)
**Status**: Major revision (manuscript TIM-26-05519) in preparation (2026-07)

---

## Overview

This supplementary package supports reproducibility of the statistical Image Quality (IQ) acceptance framework presented in the paper. It contains the analysis code, processed datasets, and result files used to derive the reported figures, tables, and acceptance thresholds.

The package is organized into two access modes:

| Tier | Content | Location | License |
|------|---------|----------|---------|
| **Tier 1** | Analysis code, processed CSVs, JSON results | This directory (`tier1_code_data/`) | Code: **MIT** / Data: **CC-BY 4.0** |
| Tier 2 | Representative cam1 raw frames (Fig. 6) | Available from the corresponding author on reasonable request (preparation materials in `tier2_on_request/`) | **CC-BY 4.0** upon release |
| Tier 3 | Full 50-condition cam1 raw archive; cam2 real-crack frame archive (50 conditions, 737 frames, revision Secs 3.7/4.7) | Available from the corresponding author on reasonable request | — |

---

## Directory Structure

```
supplementary/
├── README.md                       # This file
├── LICENSE-CODE-MIT.txt            # MIT License (code)
├── LICENSE-DATA-CC-BY-4.0.txt      # CC-BY 4.0 (data)
├── requirements.txt                # Python dependencies (minimum versions)
├── ENVIRONMENT_SNAPSHOT.md         # Exact environment of the published runs (pip freeze, seeds)
│
├── tier1_code_data/
│   ├── code/
│   │   ├── phase1/                 # Phase 1 analysis (v1 paper)
│   │   │   ├── multi_criterion_gate.py
│   │   │   ├── bootstrap_ci.py
│   │   │   ├── gof_calibration.py
│   │   │   ├── pod_ci_band.py
│   │   │   ├── sensitivity_monotonic.py
│   │   │   └── sigma_motion_model.py
│   │   ├── v2_analysis/            # v2 analysis (current IEEE TIM v2 paper)
│   │   │   ├── build_long_v3.py
│   │   │   ├── A1_per_metric_hitmiss.py
│   │   │   ├── A1b_iq_only_100entries.py
│   │   │   ├── A2_stratified_pod_iq.py
│   │   │   ├── A3_basic_pod_width.py
│   │   │   ├── A4_per_width_detection.py
│   │   │   ├── A5_extended_logistic_pod.py
│   │   │   ├── A5b_extended_logistic_with_sigma.py
│   │   │   └── A6_perAxis_thresholds.py
│   │   └── figures/                # Figure generation scripts
│   │       ├── fig01_roc.py ... fig14_representative_imagery.py   # Phase 1 (v1) figures
│   │       └── v2_figures/         # IEEE TIM v2 figures (F5-F12; F6 = Representative Imagery)
│   ├── data/                       # Processed input CSVs
│   │   ├── wide_with_shading.csv   # 50 conditions, wide format
│   │   ├── long_v3.csv             # 500 rows (50 cond x 10 widths), cam1+cam2 mean IQ
│   │   ├── pod_predictions.csv     # 500 rows + probability predictions
│   │   ├── sigma_motion_data.csv   # 53 motion-blur conditions (Phase 1)
│   │   ├── pod_curve_pooled.csv    # Pooled POD curve points
│   │   ├── pod_curve_gate_pass.csv # Gate-pass POD curve points
│   │   ├── stratified_pod_curves.csv # IQ-stratified POD curves (v2)
│   │   └── logistic_predictions.csv
│   └── results_json/               # Result JSONs
│       ├── (Phase 1) bootstrap_ci.json, gate_optimization.json, gof_calibration.json,
│       │            pod_ci_band.json, sensitivity_monotonic.json, sigma_motion_model.json
│       ├── (v2)     A5b_extended_logistic_with_sigma.json, A6_perAxis_thresholds.json,
│       │            basic_pod_width.json, extended_logistic_pod.json, iq_only_100.json,
│       │            per_metric_hitmiss.json, per_width_detection.json, stratified_pod_iq.json
│       └── PHASE1_RESULTS_SUMMARY.md   # Phase 1 summary document
│
└── tier2_on_request/              # Preparation materials for on-request release
    ├── README_TIER2.md             # Release guide + 4-item anonymization checklist
    ├── metadata.json               # Dataset descriptor (used when fulfilling a request)
    └── raw_frames/                 # Placeholder for 4 raw frames (released on request only)
```

---

## Python Environment

- **Python**: 3.10 or newer
- **Install dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

Core libraries: NumPy, SciPy, statsmodels, scikit-learn, pandas, matplotlib.

`requirements.txt` specifies minimum versions only. The exact environment of
the published runs (full `pip freeze`, Python/OS versions, and the fixed seeds
and repetition counts of the revision analyses, e.g. seed 20260511, B = 2000,
R = 200, grouped folds = 5 for `revision/wp9_cluster_inference.py`) is recorded
in `ENVIRONMENT_SNAPSHOT.md`. Bootstrap confidence-interval trailing digits can
drift across library versions; the published JSON files are the canonical
record.

---

## Reproduction Workflow

### A. Reproduce v2 paper results (recommended path)

The IEEE TIM v2 manuscript reports the following pipeline. All scripts assume the working directory is `tier1_code_data/`.

```bash
# 1. Build the v2 long-format dataset (500 rows = 50 conditions x 10 crack widths)
python code/v2_analysis/build_long_v3.py

# 2. Univariate IQ -> detection logistic (Sec 4.3, Fig. F9)
python code/v2_analysis/A1_per_metric_hitmiss.py
python code/v2_analysis/A1b_iq_only_100entries.py

# 3. Stratified POD by IQ median split (Sec 4.4, Fig. F11)
python code/v2_analysis/A2_stratified_pod_iq.py

# 4. Basic POD linear-in-width (Sec 4.4, Fig. F10; a_{90/95} = 1.008 mm)
python code/v2_analysis/A3_basic_pod_width.py

# 5. Per-width detection rate (Sec 4.2, Fig. F8)
python code/v2_analysis/A4_per_width_detection.py

# 6. Nested multivariable logistic M0/M1/M2 with sigma_motion (Sec 4.5, Tables VII/VIII)
python code/v2_analysis/A5_extended_logistic_pod.py
python code/v2_analysis/A5b_extended_logistic_with_sigma.py

# 7. Numerical inversion at a* = 0.5 mm -> per-axis IQ thresholds and their
#    cluster-bootstrap CIs (B = 2000, fixed seed) reported in Table XI of the
#    revised manuscript (Table IX of the original submission; flowchart Fig. 6)
python code/v2_analysis/A6_perAxis_thresholds.py

# 8. Regenerate v2 figures (F5-F12; F6 = Representative Imagery, NEW in v21)
python code/figures/v2_figures/fig05_inversion_flowchart.py
python code/figures/v2_figures/fig06_representative_imagery.py
python code/figures/v2_figures/fig07_iq_distributions.py
python code/figures/v2_figures/fig08_per_width_detection.py
python code/figures/v2_figures/fig09_univariate_forest.py
python code/figures/v2_figures/fig10_pod_pooled.py
python code/figures/v2_figures/fig11_pod_stratified_iq.py
python code/figures/v2_figures/fig12_acceptance_cascade.py
```

### B. Phase 1 (v1) analyses (preserved for cross-reference)

Phase 1 analyses produced the dual-criterion gate (C_M, L_90) and the system-invariant motion-blur ratio eta ~ 0.706. They are preserved in `code/phase1/` for completeness and for the bridge between v1 dual-criterion framing and the v2 5-metric synthesis.

```bash
python code/phase1/multi_criterion_gate.py     # gate_optimization.json
python code/phase1/bootstrap_ci.py             # bootstrap_ci.json
python code/phase1/gof_calibration.py          # gof_calibration.json
python code/phase1/pod_ci_band.py              # pod_ci_band.json
python code/phase1/sensitivity_monotonic.py    # sensitivity_monotonic.json (Delta-AIC = 118.13)
python code/phase1/sigma_motion_model.py       # sigma_motion_model.json (eta = 0.706, tau_eff = 35.3 us)
```

---

## Data Files: Schema Notes

| File | Rows | Schema (key columns) |
|------|------|----------------------|
| `wide_with_shading.csv` | 50 | speed_kmh, iso, distance_m, C_M, L_90, MTF50_H, BEW_H, sigma_motion, identifiable (cam1) |
| `long_v3.csv` | 500 | 50 conditions x 10 widths (0.1-1.0 mm), cam1+cam2 mean IQ + cam1 hit/miss |
| `pod_predictions.csv` | 500 | width_mm, predicted_P(detect), pooled and gate-pass |
| `sigma_motion_data.csv` | 53 | per-condition x camera sigma_motion vs sigma_theory (kinematic v . tau_int / GSD) |
| `stratified_pod_curves.csv` | varies | IQ metric, split (high/low), width_mm, POD |

---

## Revision Addendum (Major Revision, 2026-07)

The major revision of the manuscript added new analyses whose code and results are included in this repository under `tier1_code_data/{code,results_json,data}`:

| Script (`code/`) | Result (`results_json/`, `data/`) | Manuscript location |
|---|---|---|
| `revision/wp3_diagnostics.py` | `revision/wp3_diagnostics.json` | Sec 4.6 — VIF/condition number, cross-validated AUC, calibration (Fig. 14, Table X) |
| `revision/wp3_m0_cv.py` | `revision/wp3_m0_cv.json` | Sec 4.6 — width-only M0 cross-validated baseline |
| `revision/wp2_monotonic_sensitivity.py` | `revision/wp2_monotonic_sensitivity.json` | Secs 3.4, 4.6 — label counts and monotonic-completion sensitivity (Table II) |
| `revision/wp2_threshold_calibration.py` | `revision/wp2_threshold_calibration.json` | Sec 3.4 — visibility-score threshold (s = 0.08) calibration evidence |
| `revision/wp2_borderline_figure.py` | — | Fig. 5 — borderline visibility examples |
| `revision/wp5_psf_uncertainty.py` | `revision/wp5_psf_uncertainty.json` | Secs 4.5–4.6 — isotropic-PSF tests, repeatability, measurement-error Monte Carlo (Figs. 13, 15) |
| `revision/wp4_thresholds_region.py` | `revision/wp4_thresholds_region.json` | Sec 5.2 — threshold diagnostics (C_M sign reversal, MTF50 geometry confounding) and joint acceptance region (Fig. 18). The per-axis inversion points and their bootstrap CIs in Table XI are produced by `v2_analysis/A6_perAxis_thresholds.py` → `results_json/A6_perAxis_thresholds.json` |
| `v2_analysis/A6_physical_range_check.py` | `results_json/A6_physical_range_check.json` | Sec 5.2 / Table XI — seed-fixed re-execution of the A6 bootstrap reporting the fraction of L_90 threshold replicates above the 8-bit ceiling (60/2000 = 3 %) and of C_M replicates outside [0, 1] |
| `revision/wp9_cluster_inference.py` | `results_json/revision/wp9_cluster_inference.json` | Secs 4.5–4.6 — cluster-aware incremental-value analyses: paired condition-cluster bootstrap ΔAUC(M1−M0) = 0.108, 95 % CI [0.04, 0.24]; 200 repeated grouped 5-fold shuffles (mean out-of-sample ΔAUC 0.003, percentiles −0.09 to +0.09); mixed-effects random-intercept sensitivity fit; exchangeable-GEE instability with successful independence-working-correlation fallback (condition-cluster-robust SEs) recorded in the JSON (B = 2000, R = 200, seed 20260511) |
| `revision/wp7_computational_cost.py`, `revision/wp7_exposure_optimized.py` | `revision/wp7_*.json` | Sec 5.3 — exposure-gate runtime measurements |
| `revision/fig_r3_publication.py` | — | Figs. 13 and 15 (split publication versions) |
| `real_crack/cnn_detect.py` | `data/real_crack/cam2_cnn_detection.csv` | Sec 3.7 — SegFormer-B4 per-frame segmentation of the 737 real-crack frames |
| `real_crack/external_validation.py`, `real_crack/external_validation_prep.py` | `data/real_crack/cam2_realcrack_condition_summary.csv`, `data/real_crack/external_validation_table.csv` | Sec 4.7 — condition-level external validation (Fig. 16) |
| `real_crack/threshold_sensitivity.py` | — | Sec 3.7 — decision-rule sensitivity sweeps |
| `real_crack/fig_r1_composite.py` | — | Fig. 16(a) — three-regime overlay composite |

Notes on reproduction: the CNN detector is the publicly released checkpoint `varcoder/segformer-b4-crack-segmentation-dataset` (Hugging Face), used without any adaptation; `real_crack/` scripts additionally require `torch` and `transformers`. The underlying cam2 real-crack frame archive (~47 GB) is not distributable through this repository and is available from the corresponding author upon reasonable request (Tier 3); `data/real_crack/cam2_real_crack_manifest.csv` documents the frame inventory. Figure/table numbers refer to the revised manuscript.

---

## License

- **Code** (all `.py` files): MIT License — see `LICENSE-CODE-MIT.txt`
- **Data** (all `.csv`, `.json`, `.md` results files): Creative Commons Attribution 4.0 International (CC-BY 4.0) — see `LICENSE-DATA-CC-BY-4.0.txt`

---

## Citation

If you use this code or data, please cite the paper:

```
Lee, C., Kim, D., Kim, D., An, J. (2026). A POD-Based Image Quality Acceptance
Framework for Crack Detection in High-Speed Mobile Tunnel Inspection.
IEEE Transactions on Instrumentation and Measurement. [DOI: TBD upon acceptance]
```

This GitHub repository may also be cited directly via its URL and Release tag.

---

## Contact

For Tier 2 representative raw frames or Tier 3 full 50-condition raw image archive, contact the corresponding author at KICT. Questions about reproduction or the analysis pipeline are also welcome.
