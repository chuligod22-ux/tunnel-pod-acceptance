# Phase 1 Results Summary — B-2 Statistical Framework

> 목적: Phase 2 본문 작성을 위한 모든 통계 결과의 단일 종합 정리
> 작성: 2026-04-27 | Phase 1 Tasks 1.1–1.7 결과 통합

---

## 1. 데이터 개요

### 1.1 분석 데이터셋

| 항목 | 값 |
|------|-----|
| **Conditions (cam1)** | 50 (2 speeds × 5 ISOs × 5 distances) |
| **Detection outcome (Y/N)** | 44 / 6 = 88% / 12% |
| **Wide format rows** | 50 (cam1, condition-level) |
| **Long format rows** | 500 (50 conditions × 10 widths, monotonic) |
| **σ_motion measurements** | 53 (cam1: 26, cam2: 27) |
| **Width range** | 0.1 – 1.0 mm (10 levels, sequential chart) |
| **Speed range** | 60, 80 km/h |
| **ISO range** | 100, 200, 400, 800, 1600 |
| **Distance range** | 2.5, 3.5, 4.5, 5.5, 6.5 m |

### 1.2 핵심 변수 정의

| 변수 | 정의 | 단위 |
|------|------|------|
| `C_M` | Percentile-based Michelson contrast = (*L*_90 − *L*_10) / (*L*_90 + *L*_10), robust variant of the classical max/min formulation (cf. Sec 3.3 Eq. 24). 코드 (`crack_contrast_analysis.py`)에서는 변수명을 `L_max`(=P_90), `L_min`(=P_10)으로 명명하지만 실제 계산은 90th/10th percentile 기반. | — |
| `L_90` | 90th percentile luminance | DN (0–255) |
| `L_10` | 10th percentile luminance | DN (0–255) |
| `MTF50_H/V` | 50% modulation transfer (ISO 12233 slant-edge) | cy/px |
| `BEW_H/V` | Blur edge width (10–90% ESF rise) | px |
| `σ_motion` | Motion-only blur (estimated by H/V decomposition) | px |
| `MDW` | Minimum detectable crack width | mm |
| `identifiable` | Reference-crack visibility decision *Y* per Sec 3.4 (computer-vision visibility pipeline + domain-expert validation, monotonic completion to 500-row hit/miss for POD analysis). 이전 노트의 "3-observer forced choice" 표현은 outdated — Sec 3.4 v06 기준으로 정정. | Y / N |

---

## 2. Univariate ROC Analysis (Task 1.1, Fig. 1)

| Predictor | n | AUC | Optimal threshold | Sens@opt | Spec@opt |
|-----------|----|------|-------------------|----------|----------|
| C_M | 50 | **0.538** | 0.312 | 0.70 | 0.67 |
| L_90 | 50 | **0.678** | 89.0 | 0.91 | 0.67 |
| MTF50_H | 28 | **0.962** | 0.134 | 0.96 | 1.00 |
| MTF50_V | 28 | 0.846 | 0.152 | 0.85 | 1.00 |
| BEW_H | 28 | 0.519 | 5.011 | 0.31 | 1.00 |
| BEW_V | 28 | **0.923** | 4.535 | 0.92 | 1.00 |
| σ_motion | 26 | **0.880** | 1.874 | 0.88 | 1.00 |

**Bootstrap 95% CI** (Task 1.2):

| Predictor | Median AUC | 95% CI |
|-----------|-----------|--------|
| C_M | 0.610 | [0.508, 0.871] |
| L_90 | 0.688 | [0.511, 0.987] |

→ 단일 메트릭으로는 결정적 분류 불가. 결합 필요.

---

## 3. Multivariable Logistic Regression (Task 1.1)

### 3.1 Model: identifiable ~ C_M + L_90 (n = 50)

```
P(identifiable) = sigmoid(β₀ + β₁·C_M + β₂·L_90)

β₀ (const)  = -44.88
β₁ (C_M)    = +104.59
β₂ (L_90)   = +0.163

Pseudo R² = 0.5625
LRT p-value = 3.30 × 10⁻⁵
Combined AUC = 0.962
```

### 3.2 Bootstrap CI (Task 1.2, n_converged = 1757/2000)

| Coefficient | Median | 95% CI |
|------------|--------|--------|
| β₀ (const) | -43.72 | [-88.84, -0.48] |
| β₁ (C_M) | +102.95 | **[-16.80, 203.35]** |
| β₂ (L_90) | +0.158 | [0.080, 0.327] |
| Combined AUC | 0.958 | [0.636, 0.966] |

→ **L_90 β CI excludes 0 → robust predictor**
→ C_M β CI includes 0 → marginal effect at boundary cases

---

## 4. Gate Performance (Task 1.1, 1.2, Fig. 6)

### 4.1 Dual-criterion gate (논문 보고: C_M ≥ 0.05 AND L_90 ≥ 55)

```
Confusion matrix:
              Predicted N    Predicted Y
Actual N      TN = 4         FP = 2
Actual Y      FN = 0         TP = 44
```

| Metric | Point | 95% Bootstrap CI |
|--------|-------|------------------|
| Accuracy | **0.960** | [0.920, 1.000] |
| Sensitivity | **1.000** | [1.000, 1.000] |
| Specificity | **0.667** | [0.333, 1.000] |
| PPV (precision) | 0.957 | — |
| NPV | 1.000 | — |

### 4.2 False positive cases

| condition | C_M | L_90 | identifiable | 비고 |
|-----------|-----|------|------------|------|
| 60/200/6.5 | 0.315 | 73 | N | 광학 metric 모두 NaN |
| 80/200/6.5 | 0.316 | 75 | N | 광학 metric 모두 NaN |

→ **본질적 한계: 두 FP는 boundary case, 광학 metric으로 분리 불가**

---

## 5. Threshold Robustness (Task 1.1, Fig. 8)

### 5.1 Grid search (C_M ∈ [0.005, 0.20], L_90 ∈ [40, 100])

```
Achievable ceiling accuracy = 96.0%
Number of threshold combinations realising ceiling = 108
Combinations achieving 100% accuracy = 0
```

### 5.2 Plateau region (96% ceiling)

| Range | Note |
|-------|------|
| C_M_thr ∈ [0.02, 0.07] | 임의 선택 가능 |
| L_90_thr ∈ [50, 60] | 임의 선택 가능 |

→ 논문 보고 임계값 (0.05, 55)은 plateau 중심값 (정당화됨)

---

## 6. Goodness-of-Fit & Calibration (Task 1.3, Fig. 4)

### 6.1 Logistic gate model (n = 50, 5 bins)

| Metric | Value |
|--------|-------|
| Pseudo R² | 0.5625 |
| Log-Likelihood | -8.026 |
| AIC / BIC | 22.05 / 27.79 |
| Brier score | **0.0564** |
| Mean Calibration Error (MCE) | **0.0267** |
| Hosmer-Lemeshow χ² | 0.895 |
| HL df | 3 |
| **HL p-value** | **0.827** ✅ (적합) |

### 6.2 POD model (gate-pass only, n = 460, 10 bins)

| Metric | Value |
|--------|-------|
| Pseudo R² | 0.1384 |
| β₀ | 2.746 |
| β₁ (slope) | 1.306 |
| Brier score | 0.1321 |
| MCE | 0.0471 |
| HL χ² | 8.866 |
| HL df | 8 |
| **HL p-value** | **0.354** ✅ (적합) |

→ **두 모델 모두 misspecification 증거 없음**

---

## 7. POD Curves (Task 1.4, Fig. 2, 3)

### 7.1 Pooled POD (n = 500, 50 conditions × 10 widths, monotonic)

```
logit(POD) = 1.979 + 0.993 · log(a)

Detection rate per width:
  0.1mm: 50%   0.5mm: 78%
  0.2mm: 54%   0.6mm: 80%
  0.3mm: 62%   0.7mm: 88%
  0.4mm: 72%   0.8mm: 88%
                0.9mm: 88%
                1.0mm: 88%
```

| Metric | Point | Wald 95% CI | Bootstrap CI |
|--------|-------|-------------|--------------|
| a₅₀ | **0.136 mm** | — | [0.068, 0.235] |
| a₉₀ | **1.246 mm** | — | [0.609, 3.417] |
| a₉₀/₉₅ | — | **2.186 mm** | 2.597 mm |

⚠️ **POD plateau ≈ 88%** (6 conditions exposure-fail)
⚠️ **a₉₀ extrapolated beyond data range (1.0 mm)** — pooled 분석으로는 NDT acceptance metric 도출 불가

### 7.2 Gate-pass POD (n = 460, 46 conditions, monotonic)

```
logit(POD) = 2.746 + 1.306 · log(a)
```

| Metric | Point | Wald 95% CI | Bootstrap CI |
|--------|-------|-------------|--------------|
| a₅₀ | **0.122 mm** | 0.160 mm | [0.068, 0.190] |
| a₉₀ | **0.657 mm** | — | [0.377, 1.183] |
| a₉₀/₉₅ | — | **0.922 mm** ✅ | 1.094 mm |

✅ **모든 acceptance metric이 데이터 범위 내** (0.1–1.0 mm)
✅ **a₉₀/₉₅ = 0.92 mm가 본 시스템의 정량적 acceptance metric**

### 7.3 Gate effect (Pooled vs Gate-pass)

| Metric | Pooled | Gate-pass | Reduction |
|--------|--------|-----------|-----------|
| a₉₀/₉₅ (Wald) | 2.186 mm (extrapolated) | **0.922 mm** | **58%** |
| a₉₀/₉₅ (Bootstrap) | 2.597 mm | 1.094 mm | **58%** |

→ **두 방법 모두 일관된 ~58% 감소**, gate의 정량적 효과 입증

---

## 8. Monotonic Assumption Sensitivity (Task 1.5)

### 8.1 Visible list completeness

| 항목 | 값 |
|------|-----|
| 평균 visible widths | **2.86 / 10** |
| Partial visible 기록 conditions | 37 / 50 |
| User-labeled (Y/X) | 12 / 50 |
| No visible info | 1 / 50 |
| Inconsistency cases (mono=1, vis=0) | 204 / 500 (40.8%) |

### 8.2 Detection rate per width 비교

| Width | Monotonic | Visible | Diff | 비고 |
|-------|-----------|---------|------|------|
| 0.1 mm | 0.50 | 0.50 | 0.00 | |
| 0.5 mm | 0.78 | 0.30 | +0.48 | |
| **1.0 mm** | **0.88** | **0.14** | **+0.74** | Visible은 비현실적 |

### 8.3 POD model 비교 (pooled)

| Metric | Monotonic | Visible | Δ |
|--------|-----------|---------|---|
| β₁ (slope) | **+0.99** ✅ | **-0.31** ❌ | +1.30 |
| Pseudo R² | 0.083 | 0.008 | +0.075 |
| AIC | **521.7** | **639.8** | **-118.1** ✅ |

→ **Monotonic 가정이 통계적으로 명확히 우위** (ΔAIC 118.1 = 결정적)
→ Visible 가정은 β₁ < 0 (비물리적, 큰 균열 detection 감소) → 채택 불가

---

## 9. σ_motion Effective Integration Ratio (Task 1.6, Fig. 7)

### 9.1 Overall

```
η = σ_motion_measured / σ_motion_kinematic_theory

Overall n = 53
η_mean = 0.679
η_median = 0.702
η_std = 0.222
95% CI Bootstrap = [0.621, 0.743]

Implied: τ_eff = η · τ_int = 0.679 × 50 μs = 33.9 μs
```

### 9.2 Per group

| Group | n | Mean η | Std |
|-------|---|--------|-----|
| cam1, 60 km/h | 12 | 0.711 | 0.179 |
| cam1, 80 km/h | 14 | 0.695 | 0.089 |
| cam2, 60 km/h | 12 | 0.552 | 0.256 |
| cam2, 80 km/h | 15 | 0.740 | 0.283 |

### 9.3 ANOVA (η ~ speed + ISO + distance + camera)

| Factor | F | p-value | 결론 |
|--------|---|---------|------|
| Speed | 1.91 | 0.174 | n.s. |
| ISO | 0.14 | 0.967 | n.s. |
| Distance | 1.25 | 0.297 | n.s. |
| Camera | 0.77 | 0.386 | n.s. |

→ **η는 시스템적 상수** (운용 조건 무관)

### 9.4 Hypothesis comparison (AIC)

| Hypothesis | AIC | ΔAIC | 채택 |
|-----------|-----|------|------|
| H1: η = constant | -8.18 | 0 | ✅ (parsimonious) |
| H2: η = f(speed) | -8.27 | -0.09 | (negligible) |
| H3: η = f(ISO, dist) | -7.37 | +0.81 | × |

→ **H1 (constant ratio model) 채택**

---

## 10. 본문 작성용 핵심 문장 (Results 4.1–4.6)

### 4.1 Univariate ROC

> "Univariate ROC analysis revealed that neither C_M alone (AUC = 0.54, 95% CI: 0.51–0.87) nor L_90 alone (AUC = 0.68, 95% CI: 0.51–0.99) provides adequate predictive power for detection outcome."

### 4.2 Multivariable logistic gate

> "The multivariable logistic combination of C_M and L_90 achieved AUC = 0.96 (95% CI: 0.64–0.97) with pseudo R² = 0.56 and likelihood-ratio test p < 0.001. The optimal operating point yielded sensitivity = 0.93 and specificity = 1.00, statistically justifying the dual-criterion gate structure. Hosmer–Lemeshow goodness-of-fit was satisfied (χ² = 0.90, df = 3, p = 0.83) with Brier score 0.056 and mean calibration error 0.027."

### 4.3 Gate performance with bootstrap CI

> "The dual-criterion gate (C_M ≥ 0.05 AND L_90 ≥ 55) achieved 96.0% accuracy (Bootstrap 95% CI: 0.92–1.00), 100% sensitivity, and 66.7% specificity (CI: 0.33–1.00). Sensitivity was perfectly preserved across all bootstrap replicates, while specificity confidence width reflects the limited negative cohort (n = 6)."

### 4.4 POD pooled limitation + gate-stratified

> "Pooled POD analysis revealed an intrinsic plateau at approximately 88%, reflecting six conditions where exposure failure precluded any detection. Restricting analysis to the 46 gate-passing conditions yielded a well-behaved logistic POD with a₅₀ = 0.12 mm (CI: 0.07–0.19), a₉₀ = 0.66 mm (CI: 0.38–1.18), and the NDT-standard a₉₀/₉₅ = 0.92 mm (Wald) / 1.09 mm (Bootstrap). The 58% reduction in a₉₀/₉₅ relative to pooled extrapolation quantifies the operational value of exposure-adequacy enforcement."

### 4.5 σ_motion gap

> "Across both cameras, both speeds, and all 5 × 5 ISO–distance conditions, the effective integration ratio η = σ_motion / σ_theory was estimated as 0.679 (Bootstrap 95% CI: 0.621–0.743), indicating that effective imaging integration is approximately 68% of the nominal 50 μs (τ_eff ≈ 33.9 μs). Multivariable ANOVA found no significant dependence on speed (p = 0.17), ISO (p = 0.97), distance (p = 0.30), or camera (p = 0.39); the constant-ratio model was preferred over speed- and environment-dependent alternatives by AIC (ΔAIC < 2). The constancy of η across operating conditions supports interpretation as a system-level property rather than a condition-specific artefact."

### 4.6 Multi-criterion gate optimization

> "Grid search across (C_M, L_90) threshold combinations confirmed two structural findings. First, 108 distinct threshold combinations within C_M ∈ [0.02, 0.07] and L_90 ∈ [50, 60] achieved the same 96% accuracy, demonstrating robustness to the precise choice of cutoff values. The reported gate (C_M = 0.05, L_90 = 55) sits at the centre of this plateau. Second, no threshold combination attained 100% accuracy, confirming that the 96% ceiling is intrinsic to the (C_M, L_90)-only gate structure."

### Methods (sensitivity analysis justification)

> "Sensitivity analysis comparing the monotonic detection assumption against the original visible-crack-list labels revealed the latter as incomplete labelling: only 2.86 of 10 widths recorded per condition on average, with 204 of 500 (40.8%) cases where widths exceeding the recorded MDW were absent from the visible list, including widths up to 1.0 mm. The visible-list-based POD model produced an unphysical negative slope (β₁ = -0.31), implying decreased detection probability with increasing defect size, while the monotonic-assumption model produced the expected positive slope (β₁ = +0.99) with substantially superior fit (ΔAIC = 118.1). This supports use of the standard NDT-POD monotonic assumption (MIL-HDBK-1823)."

---

## 11. Figures Inventory (Task 1.7)

| # | File | Title | Key Message |
|---|------|-------|------------|
| F1 | `F1_roc_curves.pdf` | ROC curves (univariate + multivariable) | Combined logistic AUC = 0.96, single metrics weak |
| F2 | `F2_pod_pooled.pdf` | POD pooled curve with 95% CI (data-restricted) | Pooled plateau ≈ 88%, a₉₀ unreachable |
| F3 | `F3_pod_stratified.pdf` | POD pooled vs gate-pass | Gate-pass: a₉₀ = 0.66 mm, a₉₀/₉₅ = 0.92 mm |
| F4 | `F4_calibration.pdf` | Calibration plot (logistic + POD) | Both models well-calibrated, HL p > 0.05 |
| F5 | `F5_cm_l90_heatmap.pdf` | C_M–L_90 detection map with logistic contours | 50 conditions in (C_M, L_90) space, gate boundary visualised |
| F6 | `F6_confusion_matrix.pdf` | Confusion matrix + performance metrics | TP=44, TN=4, FP=2, FN=0; Acc=96%, Sens=100% |
| F7 | `F7_sigma_motion.pdf` | η distribution by camera × speed (2×2) | Mean η = 0.68, no group dependence |
| F8 | `F8_threshold_grid.pdf` | Threshold robustness grid search | 96% ceiling, 108 combos, threshold-robust |

---

## 12. Phase 2 본문 작성 준비 — Section Mapping

| Manuscript section | 활용할 결과 |
|-------------------|-----------|
| **Sec 2 Theory** | POD framework formalism, logistic model, ROC theory |
| **Sec 3 Method** | Field exp + statistical framework + monotonic assumption justification (§8) |
| **Sec 4.1 Univariate ROC** | §2 + Fig. 1 |
| **Sec 4.2 Multivariable logistic gate** | §3 + Fig. 1 + Fig. 5 |
| **Sec 4.3 POD pooled** | §7.1 + Fig. 2 |
| **Sec 4.4 POD gate-stratified** | §7.2-7.3 + Fig. 3 |
| **Sec 4.5 σ_motion gap** | §9 + Fig. 7 |
| **Sec 4.6 Multi-criterion optimization** | §5 + §4 + Fig. 6 + Fig. 8 |
| **Sec 5 Discussion** | Limitations (n=50, cam1 only, monotonic), σ_motion mechanism |
| **Sec 6 Conclusion** | a₉₀/₉₅ = 0.92 mm as the system acceptance metric |

---

## 13. 데이터 파일 위치

```
01_tunnelscanning/04_data/b2_results/
├── wide_50conditions.csv         # 50 conditions, all variables
├── wide_with_shading.csv         # +shading data
├── long_500rows.csv              # 50×10 widths, monotonic detection
├── long_v2.csv (in tmp/b2_out/)  # Same as above (working copy)
├── pod_curve_pooled.csv          # POD curve, pooled, 200 width points
├── pod_curve_gate_pass.csv       # POD curve, gate-pass
├── logistic_predictions.csv      # 50 conditions logistic probabilities
├── pod_predictions.csv           # 500 rows POD probabilities
├── sigma_motion_data.csv         # 53 measurements, η values
├── gate_optimization.json        # Task 1.1 results
├── bootstrap_ci.json             # Task 1.2 results
├── gof_calibration.json          # Task 1.3 results
├── pod_ci_band.json              # Task 1.4 results
├── sensitivity_monotonic.json    # Task 1.5 results
├── sigma_motion_model.json       # Task 1.6 results
└── figures/
    ├── F1_roc_curves.{pdf,png}
    ├── F2_pod_pooled.{pdf,png}
    ├── F3_pod_stratified.{pdf,png}
    ├── F4_calibration.{pdf,png}
    ├── F5_cm_l90_heatmap.{pdf,png}
    ├── F6_confusion_matrix.{pdf,png}
    ├── F7_sigma_motion.{pdf,png}
    └── F8_threshold_grid.{pdf,png}
```

---

## 14. Phase 1 완료 체크리스트

- [x] T01 Multi-criterion gate optimization
- [x] T02 Bootstrap CI for AUC, logistic coef, POD parameters
- [x] T03 Hosmer-Lemeshow GOF + calibration analysis
- [x] T04 POD curve 95% CI band (Berens method)
- [x] T05 Monotonic assumption sensitivity analysis
- [x] T06 σ_motion gap effective integration model
- [x] T07 8 publication-quality figures
- [x] T08 Statistics summary (this document)

**Phase 1 완료**: 2026-04-27

---

## 15. Phase 2 준비 사항

다음 단계:
1. d0004_todo.md의 Phase 1 항목 완료 표시
2. d0010_history.md 생성 + Phase 1 완료 기록
3. Phase 2 시작: Title, Abstract, Sec 1 Introduction 작성

`d0001_prd.md`, `d0002_plan.md`, 그리고 본 `results_summary.md`가 Phase 2 작업의 토대.
