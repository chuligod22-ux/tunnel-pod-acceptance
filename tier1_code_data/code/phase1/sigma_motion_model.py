"""
Phase 1 Task 1.6: σ_motion Gap — Effective Integration Ratio Model
================================================================
관찰: σ_motion 측정값이 kinematic 예측 대비 일관된 ~30% gap

분석:
  1. η = σ_motion_measured / σ_motion_theory (effective integration ratio)
  2. η의 분포 + 95% CI (per speed, ISO, distance)
  3. η가 다른 변수(ISO, distance)에 의존하는지 ANOVA
  4. 통합 모델: η constant vs covariate-dependent
  5. 물리적 해석 + alternative hypotheses

Theory:
  Kinematic blur (top-hat exposure):
    σ_kinematic = (v · τ_int) / (GSD · 2√3)  [or similar]
  본 데이터의 theory_blur = v · τ_int / GSD (px) — 직접 비교용
  σ_motion / theory_blur ≈ 0.7 일관 → effective τ_eff ≈ 0.7·τ_int

Hypothesis:
  H1: η = constant (≈0.70)        — single ratio model
  H2: η = f(speed)                  — speed-dependent
  H3: η = f(ISO, distance)          — environment-dependent
  H4: σ_motion = α · σ_theory^β     — power-law
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

ROOT = Path("/Users/lch/home/code/tunnelscanning")
DATA = ROOT / "01_tunnelscanning/03_src/data"
OUT = ROOT / "01_tunnelscanning/04_data/b2_results"

# combined_analysis.csv: both cameras
df = pd.read_csv(DATA / "combined_analysis.csv")
df = df.dropna(subset=['sigma_motion','blur_err_pct']).copy()
# Compute theory_blur from kinematics: v(m/s) * τ_int(s) / GSD(m) [px]
TAU_INT = 50e-6   # 50 μs nominal integration
GSD_M = 0.3e-3    # 0.3 mm/px
df['v_ms'] = df['speed'] * 1000.0 / 3600.0  # km/h → m/s
df['theory_blur'] = df['v_ms'] * TAU_INT / GSD_M  # px
df['eta'] = df['sigma_motion'] / df['theory_blur']  # effective integration ratio
df['eta_pct_dev'] = (df['eta'] - 1) * 100  # % deviation from kinematic (negative)
print(f"  Theory blur: 60 km/h → {df[df['speed']==60]['theory_blur'].iloc[0]:.3f} px,  80 km/h → {df[df['speed']==80]['theory_blur'].iloc[0]:.3f} px")

print("="*80)
print("[A] Effective integration ratio η = σ_motion / σ_theory_kinematic")
print("="*80)
print(f"\n  Total measurements: {len(df)}")
print(f"  Cameras: {df['cam'].value_counts().to_dict()}")
print(f"  Speeds:  {df['speed'].value_counts().to_dict()}")

# Overall summary
print(f"\n  η overall: mean={df['eta'].mean():.4f}, median={df['eta'].median():.4f}, "
      f"std={df['eta'].std():.4f}, n={len(df)}")
ci_lo, ci_hi = stats.t.interval(0.95, len(df)-1, df['eta'].mean(), stats.sem(df['eta']))
print(f"  η 95% CI (t-distribution): [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Implied effective integration time: τ_eff = {df['eta'].mean():.3f} × τ_int")
print(f"  → if τ_int = 50 μs, τ_eff ≈ {df['eta'].mean()*50:.1f} μs")

# Per speed
print("\n  η by speed:")
print(f"  {'speed':>6} {'n':>4} {'mean':>8} {'std':>8} {'95%CI':>20}")
for spd, sub in df.groupby('speed'):
    m = sub['eta'].mean(); s = sub['eta'].std()
    lo, hi = stats.t.interval(0.95, len(sub)-1, m, stats.sem(sub['eta']))
    print(f"  {spd:>6} {len(sub):>4} {m:>8.4f} {s:>8.4f}  [{lo:.4f}, {hi:.4f}]")

# Per camera
print("\n  η by camera:")
for cam, sub in df.groupby('cam'):
    m = sub['eta'].mean(); s = sub['eta'].std()
    print(f"  {cam}: n={len(sub)}, mean={m:.4f}, std={s:.4f}")

# Per distance
print("\n  η by distance:")
for d, sub in df.groupby('dist'):
    m = sub['eta'].mean(); s = sub['eta'].std()
    print(f"  d={d}m: n={len(sub)}, mean={m:.4f}, std={s:.4f}")

# Per ISO
print("\n  η by ISO:")
for iso, sub in df.groupby('iso'):
    m = sub['eta'].mean(); s = sub['eta'].std()
    print(f"  ISO {iso}: n={len(sub)}, mean={m:.4f}, std={s:.4f}")

# ===========================================================================
# B) ANOVA: η ~ speed + ISO + distance + camera
# ===========================================================================
print("\n" + "="*80)
print("[B] ANOVA: η ~ speed + ISO + distance + camera")
print("="*80)

import statsmodels.formula.api as smf
df_an = df.copy()
df_an['speed_c'] = df_an['speed'].astype('category')
df_an['iso_c'] = df_an['iso'].astype('category')
df_an['dist_c'] = df_an['dist'].astype('category')
df_an['cam_c'] = df_an['cam'].astype('category')

try:
    model = smf.ols('eta ~ C(speed_c) + C(iso_c) + C(dist_c) + C(cam_c)', data=df_an).fit()
    print(model.summary())
    print("\n  ANOVA Type II:")
    anova = sm.stats.anova_lm(model, typ=2)
    print(anova)
except Exception as e:
    print(f"ANOVA failed: {e}")

# ===========================================================================
# C) Hypothesis comparison via AIC
# ===========================================================================
print("\n" + "="*80)
print("[C] HYPOTHESIS COMPARISON (AIC)")
print("="*80)

models = {}
# H1: constant
y = df['eta'].values
X1 = np.ones((len(y),1))
m1 = sm.OLS(y, X1).fit()
models['H1_constant'] = {'aic': m1.aic, 'r2': m1.rsquared, 'params': m1.params.tolist()}
print(f"  H1 (constant η): AIC={m1.aic:.2f}, R²={m1.rsquared:.4f}, η_hat={m1.params[0]:.4f}")

# H2: η = f(speed)
X2 = sm.add_constant(df['speed'].values.astype(float))
m2 = sm.OLS(y, X2).fit()
models['H2_speed'] = {'aic': m2.aic, 'r2': m2.rsquared, 'params': m2.params.tolist()}
print(f"  H2 (speed-dep): AIC={m2.aic:.2f}, R²={m2.rsquared:.4f}, "
      f"η = {m2.params[0]:.4f} + {m2.params[1]:.6f}·speed")

# H3: η = f(ISO, distance)
X3 = sm.add_constant(df[['iso','dist']].astype(float))
m3 = sm.OLS(y, X3).fit()
models['H3_iso_dist'] = {'aic': m3.aic, 'r2': m3.rsquared, 'params': m3.params.tolist()}
print(f"  H3 (ISO+dist): AIC={m3.aic:.2f}, R²={m3.rsquared:.4f}")

# H4: power-law σ = α · σ_theory^β
log_y_pl = np.log(df['sigma_motion'].values)
log_x_pl = np.log(df['theory_blur'].values)
X4 = sm.add_constant(log_x_pl)
m4 = sm.OLS(log_y_pl, X4).fit()
alpha = float(np.exp(m4.params[0])); beta = float(m4.params[1])
models['H4_power_law'] = {'aic': m4.aic, 'r2': m4.rsquared, 'alpha': alpha, 'beta': beta}
print(f"  H4 (power-law): AIC(log)={m4.aic:.2f}, R²={m4.rsquared:.4f}")
print(f"     σ_motion = {alpha:.4f} · σ_theory^{beta:.4f}")
print(f"     If β≈1, η = α (constant ratio); β≠1 means non-linear")

# ===========================================================================
# D) Best model + interpretation
# ===========================================================================
print("\n" + "="*80)
print("[D] BEST MODEL & INTERPRETATION")
print("="*80)

best = min(['H1_constant','H2_speed','H3_iso_dist'], key=lambda k: models[k]['aic'])
print(f"\n  Best AIC (excluding H4 — different scale): {best}")
delta_aic = {k: models[k]['aic'] - models['H1_constant']['aic'] for k in ['H1_constant','H2_speed','H3_iso_dist']}
print(f"  ΔAIC vs H1: {delta_aic}")

if abs(delta_aic['H2_speed']) < 2 and abs(delta_aic['H3_iso_dist']) < 2:
    print("  → H1 (constant) is preferred (parsimony, ΔAIC < 2)")
    print(f"  → η = {df['eta'].mean():.4f} ± {df['eta'].std():.4f} (mean ± SD)")
elif delta_aic['H2_speed'] < -2:
    print("  → H2 (speed-dependent) is preferred")
elif delta_aic['H3_iso_dist'] < -2:
    print("  → H3 (ISO+distance) is preferred")
else:
    print("  → Mixed evidence; constant model is parsimonious")

# Power-law check
ci_b = m4.conf_int()
ci_b_lo = ci_b[1,0] if hasattr(ci_b, 'shape') else ci_b.iloc[1,0]
ci_b_hi = ci_b[1,1] if hasattr(ci_b, 'shape') else ci_b.iloc[1,1]
print(f"\n  Power-law slope β = {beta:.4f} (95% CI: [{ci_b_lo:.4f}, {ci_b_hi:.4f}])")
if 0.95 < beta < 1.05:
    print(f"  → β ≈ 1 confirms ratio model (η is roughly constant)")

# ===========================================================================
# E) Bootstrap CI for η
# ===========================================================================
print("\n" + "="*80)
print("[E] Bootstrap 95% CI for η (constant model)")
print("="*80)

N_BOOT = 5000
RNG = np.random.default_rng(42)
eta_boot = []
for _ in range(N_BOOT):
    idx = RNG.choice(len(df), size=len(df), replace=True)
    eta_boot.append(df['eta'].iloc[idx].mean())

print(f"  Bootstrap mean η: median={np.median(eta_boot):.4f}, "
      f"95% CI=[{np.percentile(eta_boot, 2.5):.4f}, {np.percentile(eta_boot, 97.5):.4f}]")

# ===========================================================================
# Save
# ===========================================================================
results = {
    'overall': {
        'n': int(len(df)),
        'eta_mean': float(df['eta'].mean()),
        'eta_median': float(df['eta'].median()),
        'eta_std': float(df['eta'].std()),
        'eta_ci95_t': [float(ci_lo), float(ci_hi)],
        'eta_bootstrap_median': float(np.median(eta_boot)),
        'eta_bootstrap_ci95': [float(np.percentile(eta_boot, 2.5)),
                                float(np.percentile(eta_boot, 97.5))],
        'tau_eff_mu_50us_int': float(df['eta'].mean() * 50),
    },
    'by_speed': {
        int(spd): {
            'n': int(len(sub)),
            'eta_mean': float(sub['eta'].mean()),
            'eta_std': float(sub['eta'].std()),
            'eta_ci95': list(stats.t.interval(0.95, len(sub)-1, sub['eta'].mean(), stats.sem(sub['eta']))),
        } for spd, sub in df.groupby('speed')
    },
    'by_camera': {
        cam: {
            'n': int(len(sub)),
            'eta_mean': float(sub['eta'].mean()),
            'eta_std': float(sub['eta'].std()),
        } for cam, sub in df.groupby('cam')
    },
    'hypothesis_comparison': models,
    'best_model': best,
    'power_law': {
        'alpha': alpha, 'beta': beta,
        'beta_ci95': [float(ci_b_lo), float(ci_b_hi)],
    },
}
with open(OUT / "sigma_motion_model.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save analyzed data for plotting
df[['cam','speed','iso','dist','sigma_motion','theory_blur','eta','eta_pct_dev','blur_err_pct']].to_csv(
    OUT / "sigma_motion_data.csv", index=False)

print(f"\nOutputs:")
print(f"  - {OUT/'sigma_motion_model.json'}")
print(f"  - {OUT/'sigma_motion_data.csv'}")
