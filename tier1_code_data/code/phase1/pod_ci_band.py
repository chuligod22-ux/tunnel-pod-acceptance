"""
Phase 1 Task 1.4: POD 95% CI Band (Berens method)
================================================================
대상: POD pooled + POD gate-pass

방법:
  1. Width grid (0.05 ~ 2.0 mm, 200 points, log-uniform)
  2. POD 점추정: P(a) = sigmoid(β₀ + β₁·log(a))
  3. Wald 95% CI (logit scale, parametric):
     SE(η)² = Var(β₀) + log(a)² · Var(β₁) + 2·log(a)·Cov(β₀,β₁)
     CI(P) = sigmoid(η ± 1.96·SE(η))
  4. Bootstrap 95% CI (cluster bootstrap by condition, 2000 iter)
  5. a90/95 정식 도출:
     - From Wald: lower CL of η at POD=0.9 → upper CL of a
     - From Bootstrap: 95th percentile of a90 samples
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

ROOT = Path("/Users/lch/home/code/tunnelscanning")
LONG = ROOT / "tmp/b2_out/long_v2.csv"
OUT = ROOT / "01_tunnelscanning/04_data/b2_results"

N_BOOT = 2000
RNG = np.random.default_rng(42)

df_l = pd.read_csv(LONG)
df_l['gate'] = ((df_l['C_M']>=0.05) & (df_l['L_90']>=55)).astype(int)

# Width grid (log-uniform)
W_GRID = np.exp(np.linspace(np.log(0.05), np.log(2.0), 200))

def fit_pod_logit(detected, log_w):
    X = sm.add_constant(log_w)
    res = sm.Logit(detected, X).fit(disp=False, maxiter=200)
    return res

def cluster_bootstrap_indices(df, group_cols, rng):
    groups = df.groupby(group_cols).indices
    keys = list(groups.keys())
    pos = rng.integers(0, len(keys), size=len(keys))
    return np.concatenate([groups[keys[i]] for i in pos])

def wald_pod_ci(beta, cov, log_a_grid):
    """Berens-style Wald 95% CI on logit scale."""
    b0, b1 = beta
    var_b0 = cov[0,0]; var_b1 = cov[1,1]; cov_b01 = cov[0,1]
    eta = b0 + b1 * log_a_grid
    var_eta = var_b0 + log_a_grid**2 * var_b1 + 2 * log_a_grid * cov_b01
    se_eta = np.sqrt(np.maximum(var_eta, 0))
    pod = 1.0 / (1.0 + np.exp(-eta))
    pod_lo = 1.0 / (1.0 + np.exp(-(eta - 1.96 * se_eta)))
    pod_hi = 1.0 / (1.0 + np.exp(-(eta + 1.96 * se_eta)))
    return pod, pod_lo, pod_hi, eta, se_eta

def find_a_at_pod(beta, target_pod):
    """Solve P = sigmoid(b0 + b1*log(a)) for a."""
    b0, b1 = beta
    if b1 == 0: return float('nan')
    return float(np.exp((np.log(target_pod/(1-target_pod)) - b0) / b1))

def find_a_wald_upper(beta, cov, target_pod, conf=0.95):
    """
    Berens a_p/conf: at POD=p, find largest a such that lower CL of POD ≥ p.
    Equivalent to: a_p,conf = exp((logit(p) - b0 + 1.96*SE(η)) / b1) under linearization.

    We solve numerically: find a* such that pod_lower(a*) = target_pod.
    """
    log_a_grid = np.linspace(np.log(0.01), np.log(50), 5000)
    pod, pod_lo, pod_hi, _, _ = wald_pod_ci(beta, cov, log_a_grid)
    # Find smallest a where pod_lower >= target
    above = pod_lo >= target_pod
    if not np.any(above): return float('nan')
    idx = np.argmax(above)  # first True
    return float(np.exp(log_a_grid[idx]))

results = {}

for label, df_sub in [('pooled', df_l), ('gate_pass', df_l[df_l['gate']==1])]:
    print("="*80)
    print(f"[POD] {label.upper()} (n={len(df_sub)})")
    print("="*80)

    y = df_sub['detected'].values
    log_w = np.log(df_sub['width_mm'].values)

    # --- Point estimate
    res = fit_pod_logit(y, log_w)
    beta = res.params
    cov = res.cov_params()
    print(f"\n  Point estimate:  β₀={beta[0]:.3f}, β₁={beta[1]:.3f}")
    print(f"  Var(β₀)={cov[0,0]:.4f}, Var(β₁)={cov[1,1]:.4f}, Cov={cov[0,1]:.4f}")

    a50_pt = find_a_at_pod(beta, 0.5)
    a90_pt = find_a_at_pod(beta, 0.9)
    print(f"  a50 (point) = {a50_pt:.4f} mm")
    print(f"  a90 (point) = {a90_pt:.4f} mm")

    # --- Wald CI band
    pod_grid, pod_lo_w, pod_hi_w, eta_grid, se_grid = wald_pod_ci(beta, cov, np.log(W_GRID))
    a90_95_wald = find_a_wald_upper(beta, cov, 0.9, conf=0.95)
    a50_95_wald = find_a_wald_upper(beta, cov, 0.5, conf=0.95)
    print(f"\n  Wald 95% CI band computed across width grid (n={len(W_GRID)})")
    print(f"  a50/95 (Wald, lower CL of POD=0.5 method) = {a50_95_wald:.4f} mm")
    print(f"  a90/95 (Wald, NDT standard) = {a90_95_wald:.4f} mm  ← PRIMARY METRIC")

    # --- Bootstrap CI band
    print(f"\n  Bootstrap CI band (cluster bootstrap, {N_BOOT} iter)...")
    boot_pods = np.zeros((N_BOOT, len(W_GRID)))
    a50_boot, a90_boot = [], []
    n_conv = 0
    for i in range(N_BOOT):
        idx = cluster_bootstrap_indices(df_sub, ['speed','iso','dist'], RNG)
        sub = df_sub.iloc[idx]
        if sub['detected'].nunique() < 2:
            boot_pods[i] = np.nan
            continue
        try:
            r = fit_pod_logit(sub['detected'].values, np.log(sub['width_mm'].values))
            if not r.mle_retvals.get('converged', True):
                boot_pods[i] = np.nan
                continue
            b = r.params
            eta = b[0] + b[1]*np.log(W_GRID)
            boot_pods[i] = 1.0/(1.0 + np.exp(-eta))
            a50_boot.append(find_a_at_pod(b, 0.5))
            a90_boot.append(find_a_at_pod(b, 0.9))
            n_conv += 1
        except Exception:
            boot_pods[i] = np.nan
    boot_pods = boot_pods[~np.isnan(boot_pods).any(axis=1)]
    pod_lo_b = np.percentile(boot_pods, 2.5, axis=0)
    pod_hi_b = np.percentile(boot_pods, 97.5, axis=0)
    pod_med_b = np.percentile(boot_pods, 50, axis=0)

    a90_95_boot = float(np.percentile(a90_boot, 95)) if a90_boot else float('nan')
    a90_med_boot = float(np.median(a90_boot)) if a90_boot else float('nan')
    a90_lo_boot = float(np.percentile(a90_boot, 2.5)) if a90_boot else float('nan')
    a90_hi_boot = float(np.percentile(a90_boot, 97.5)) if a90_boot else float('nan')

    print(f"  n_converged: {n_conv}/{N_BOOT}")
    print(f"  Bootstrap a90: median={a90_med_boot:.3f}  95% CI=[{a90_lo_boot:.3f}, {a90_hi_boot:.3f}]")
    print(f"  a90/95 (Bootstrap, 95th percentile of a90) = {a90_95_boot:.3f} mm")

    # --- Save curve data for plotting
    curve_df = pd.DataFrame({
        'width_mm': W_GRID,
        'pod_point': pod_grid,
        'pod_wald_lo': pod_lo_w,
        'pod_wald_hi': pod_hi_w,
        'pod_boot_med': pod_med_b,
        'pod_boot_lo': pod_lo_b,
        'pod_boot_hi': pod_hi_b,
    })
    curve_df.to_csv(OUT / f"pod_curve_{label}.csv", index=False)
    print(f"\n  Saved: {OUT/f'pod_curve_{label}.csv'}")

    results[label] = {
        'n': int(len(df_sub)),
        'beta_0_point': float(beta[0]),
        'beta_1_point': float(beta[1]),
        'cov_matrix': cov.tolist(),
        'a50_point_mm': float(a50_pt),
        'a90_point_mm': float(a90_pt),
        'a50_95_wald_mm': float(a50_95_wald),
        'a90_95_wald_mm': float(a90_95_wald),
        'a90_bootstrap': {
            'median': a90_med_boot,
            'ci_lower': a90_lo_boot,
            'ci_upper': a90_hi_boot,
            'a90_95': a90_95_boot,
        },
        'n_bootstrap_converged': n_conv,
    }

# Comparison summary
print("\n" + "="*80)
print("SUMMARY: a90/95 Comparison (NDT acceptance metric)")
print("="*80)
print(f"\n  {'Model':>15} {'a90 (point)':>12} {'a90/95 (Wald)':>14} {'a90/95 (Boot)':>14}")
for label in ['pooled', 'gate_pass']:
    r = results[label]
    print(f"  {label:>15} {r['a90_point_mm']:>12.3f} {r['a90_95_wald_mm']:>14.3f} {r['a90_bootstrap']['a90_95']:>14.3f}")

print(f"\n  Effect of gating: a90/95 reduced from {results['pooled']['a90_95_wald_mm']:.3f} to {results['gate_pass']['a90_95_wald_mm']:.3f} mm")
print(f"  (Wald-based, NDT-style 90% POD with 95% upper CL)")

# Save
with open(OUT / "pod_ci_band.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nOutput: {OUT/'pod_ci_band.json'}")
print(f"        {OUT/'pod_curve_pooled.csv'}")
print(f"        {OUT/'pod_curve_gate_pass.csv'}")
