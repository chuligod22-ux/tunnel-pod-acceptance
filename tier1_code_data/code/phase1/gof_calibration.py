"""
Phase 1 Task 1.3: Hosmer-Lemeshow GOF + Calibration Analysis
================================================================
대상:
  A. Logistic regression (condition-level, n=50)
     - Hosmer-Lemeshow chi-square test (5 bins, 작은 표본 고려)
     - Brier score
     - Calibration curve (binned + smoothed)
  B. POD model (long-format, n=500, monotonic)
     - HL test (10 bins)
     - Brier score
     - Calibration curve
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

ROOT = Path("/Users/lch/home/code/tunnelscanning")
WIDE = ROOT / "tmp/b2_out/wide_v2.csv"
LONG = ROOT / "tmp/b2_out/long_v2.csv"
OUT = ROOT / "01_tunnelscanning/04_data/b2_results"

df_w = pd.read_csv(WIDE)
df_l = pd.read_csv(LONG)
y_w = df_w['identifiable_bin'].values
y_l = df_l['detected'].values

results = {}

# ---------------------------------------------------------------------------
# Helper: Hosmer-Lemeshow test
# ---------------------------------------------------------------------------
def hosmer_lemeshow(y, p, n_bins=10):
    """
    HL chi-square goodness-of-fit test for logistic regression.
    Returns: chi2, df, p_value, bin_table
    """
    df_hl = pd.DataFrame({'y': y, 'p': p})
    # Quantile-based bins
    try:
        df_hl['bin'] = pd.qcut(df_hl['p'], q=n_bins, duplicates='drop')
    except ValueError:
        return None
    g = df_hl.groupby('bin', observed=True).agg(
        n=('y','count'),
        observed_pos=('y','sum'),
        expected_pos=('p','sum'),
    )
    g['observed_neg'] = g['n'] - g['observed_pos']
    g['expected_neg'] = g['n'] - g['expected_pos']
    # Chi-square
    chi2 = 0
    for _, row in g.iterrows():
        if row['expected_pos'] > 0:
            chi2 += (row['observed_pos'] - row['expected_pos'])**2 / row['expected_pos']
        if row['expected_neg'] > 0:
            chi2 += (row['observed_neg'] - row['expected_neg'])**2 / row['expected_neg']
    df_chi = len(g) - 2  # HL convention
    if df_chi <= 0:
        return None
    p_val = 1 - stats.chi2.cdf(chi2, df_chi)
    return {
        'chi2': float(chi2),
        'df': int(df_chi),
        'p_value': float(p_val),
        'n_bins_used': len(g),
        'bin_table': g.reset_index().assign(
            bin=lambda d: d['bin'].astype(str)
        ).to_dict(orient='records'),
    }

# ---------------------------------------------------------------------------
# Helper: Calibration curve (binned)
# ---------------------------------------------------------------------------
def calibration_binned(y, p, n_bins=10):
    """Returns mean predicted prob and observed freq per bin."""
    df_c = pd.DataFrame({'y': y, 'p': p})
    try:
        df_c['bin'] = pd.qcut(df_c['p'], q=n_bins, duplicates='drop')
    except ValueError:
        return None
    g = df_c.groupby('bin', observed=True).agg(
        n=('y','count'),
        mean_pred=('p','mean'),
        observed_freq=('y','mean'),
    ).reset_index().drop(columns='bin')
    return g.to_dict(orient='records')

# ===========================================================================
# A) Logistic (condition-level, n=50, identifiable ~ C_M + L_90)
# ===========================================================================
print("="*80)
print("[A] LOGISTIC REGRESSION (n=50, identifiable ~ C_M + L_90)")
print("="*80)

X = sm.add_constant(df_w[['C_M','L_90']].astype(float))
res = sm.Logit(y_w, X).fit(disp=False, maxiter=200)
prob = res.predict(X)

print(f"\n  Pseudo R² = {res.prsquared:.4f}")
print(f"  Log-Likelihood = {res.llf:.3f}")
print(f"  AIC = {res.aic:.2f}, BIC = {res.bic:.2f}")
print(f"  LRT p-value = {res.llr_pvalue:.2e}")

# HL test (small sample → 5 bins)
print("\n  Hosmer-Lemeshow (n_bins=5, 작은 표본 보정):")
hl_5 = hosmer_lemeshow(y_w, prob, n_bins=5)
if hl_5:
    print(f"    χ² = {hl_5['chi2']:.3f}, df = {hl_5['df']}, p = {hl_5['p_value']:.4f}")
    print(f"    → {'적합 가설 채택 (모델 적합)' if hl_5['p_value']>0.05 else 'reject 적합 (모델 부적합)'}")
    print("\n    Bin table:")
    print(f"    {'bin':>20} {'n':>4} {'obs_pos':>8} {'exp_pos':>8} {'obs_neg':>8} {'exp_neg':>8}")
    for r in hl_5['bin_table']:
        print(f"    {r['bin']:>20} {r['n']:>4} {r['observed_pos']:>8.1f} {r['expected_pos']:>8.2f} {r['observed_neg']:>8.1f} {r['expected_neg']:>8.2f}")

# Brier score
brier_log = brier_score_loss(y_w, prob)
print(f"\n  Brier score = {brier_log:.4f}  (낮을수록 better, 0이 perfect)")

# Calibration
print("\n  Calibration (n_bins=5):")
cal_5 = calibration_binned(y_w, prob, n_bins=5)
if cal_5:
    print(f"    {'n':>4} {'mean_pred':>10} {'obs_freq':>10}")
    for r in cal_5:
        print(f"    {r['n']:>4} {r['mean_pred']:>10.3f} {r['observed_freq']:>10.3f}")

# Mean calibration error (MCE)
if cal_5:
    mce = np.mean([abs(r['mean_pred'] - r['observed_freq']) for r in cal_5])
    print(f"\n  Mean Calibration Error (MCE) = {mce:.4f}")

results['logistic_gof'] = {
    'pseudo_R2': float(res.prsquared),
    'log_likelihood': float(res.llf),
    'aic': float(res.aic),
    'bic': float(res.bic),
    'llr_pvalue': float(res.llr_pvalue),
    'brier_score': float(brier_log),
    'hl_test_5bin': hl_5,
    'calibration_5bin': cal_5,
    'mean_calibration_error': float(mce) if cal_5 else None,
}

# ===========================================================================
# B) POD model (long, n=500, detected ~ log(width))
# ===========================================================================
print("\n" + "="*80)
print("[B] POD MODEL (n=500, detected ~ log(width))")
print("="*80)

X_pod = sm.add_constant(np.log(df_l['width_mm'].values))
res_pod = sm.Logit(y_l, X_pod).fit(disp=False, maxiter=200)
prob_pod = res_pod.predict(X_pod)

print(f"\n  Pseudo R² = {res_pod.prsquared:.4f}")
print(f"  Log-Likelihood = {res_pod.llf:.3f}")
print(f"  AIC = {res_pod.aic:.2f}, BIC = {res_pod.bic:.2f}")
print(f"  Coefficients: β₀={res_pod.params[0]:.3f}, β₁={res_pod.params[1]:.3f}")

# HL test (n=500, 10 bins)
print("\n  Hosmer-Lemeshow (n_bins=10):")
hl_10 = hosmer_lemeshow(y_l, prob_pod, n_bins=10)
if hl_10:
    print(f"    χ² = {hl_10['chi2']:.3f}, df = {hl_10['df']}, p = {hl_10['p_value']:.4f}")
    print(f"    → {'적합 가설 채택 (모델 적합)' if hl_10['p_value']>0.05 else 'reject 적합 (모델 부적합)'}")

brier_pod = brier_score_loss(y_l, prob_pod)
print(f"\n  Brier score = {brier_pod:.4f}")

cal_10 = calibration_binned(y_l, prob_pod, n_bins=10)
if cal_10:
    print("\n  Calibration (n_bins=10):")
    print(f"    {'n':>4} {'mean_pred':>10} {'obs_freq':>10}")
    for r in cal_10:
        print(f"    {r['n']:>4} {r['mean_pred']:>10.3f} {r['observed_freq']:>10.3f}")
    mce_pod = np.mean([abs(r['mean_pred'] - r['observed_freq']) for r in cal_10])
    print(f"\n  Mean Calibration Error (MCE) = {mce_pod:.4f}")
else:
    mce_pod = None

results['pod_gof'] = {
    'pseudo_R2': float(res_pod.prsquared),
    'log_likelihood': float(res_pod.llf),
    'aic': float(res_pod.aic),
    'bic': float(res_pod.bic),
    'beta_0': float(res_pod.params[0]),
    'beta_1': float(res_pod.params[1]),
    'brier_score': float(brier_pod),
    'hl_test_10bin': hl_10,
    'calibration_10bin': cal_10,
    'mean_calibration_error': float(mce_pod) if mce_pod else None,
}

# ===========================================================================
# C) POD stratified (gate pass) — 추가 GOF
# ===========================================================================
print("\n" + "="*80)
print("[C] POD MODEL — Gate Pass Only (n=460)")
print("="*80)

df_l['gate'] = ((df_l['C_M']>=0.05) & (df_l['L_90']>=55)).astype(int)
df_pass = df_l[df_l['gate']==1]
y_pass = df_pass['detected'].values
X_pass = sm.add_constant(np.log(df_pass['width_mm'].values))
res_pass = sm.Logit(y_pass, X_pass).fit(disp=False, maxiter=200)
prob_pass = res_pass.predict(X_pass)

print(f"\n  Pseudo R² = {res_pass.prsquared:.4f}")
print(f"  Coefficients: β₀={res_pass.params[0]:.3f}, β₁={res_pass.params[1]:.3f}")

hl_pass = hosmer_lemeshow(y_pass, prob_pass, n_bins=10)
if hl_pass:
    print(f"\n  HL: χ² = {hl_pass['chi2']:.3f}, df = {hl_pass['df']}, p = {hl_pass['p_value']:.4f}")

brier_pass = brier_score_loss(y_pass, prob_pass)
print(f"  Brier score = {brier_pass:.4f}")

cal_pass = calibration_binned(y_pass, prob_pass, n_bins=10)
mce_pass = np.mean([abs(r['mean_pred'] - r['observed_freq']) for r in cal_pass]) if cal_pass else None
print(f"  Mean Calibration Error = {mce_pass:.4f}" if mce_pass else "  MCE: n/a")

results['pod_gate_pass_gof'] = {
    'pseudo_R2': float(res_pass.prsquared),
    'beta_0': float(res_pass.params[0]),
    'beta_1': float(res_pass.params[1]),
    'brier_score': float(brier_pass),
    'hl_test_10bin': hl_pass,
    'calibration_10bin': cal_pass,
    'mean_calibration_error': float(mce_pass) if mce_pass else None,
}

# Save
with open(OUT / "gof_calibration.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Calibration data for plotting
df_w['logistic_prob'] = prob
df_w[['speed','iso','dist','C_M','L_90','identifiable_bin','logistic_prob']].to_csv(
    OUT / "logistic_predictions.csv", index=False)

df_l['pod_prob_pooled'] = prob_pod
df_l.to_csv(OUT / "pod_predictions.csv", index=False)

print("\n" + "="*80)
print("Output files:")
print(f"  - {OUT/'gof_calibration.json'}")
print(f"  - {OUT/'logistic_predictions.csv'}")
print(f"  - {OUT/'pod_predictions.csv'}")
print("="*80)
