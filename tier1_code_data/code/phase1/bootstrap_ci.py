"""
Phase 1 Task 1.2: Bootstrap Confidence Intervals
================================================================
대상:
  1. ROC AUC (univariate + multivariable logistic)
  2. Logistic regression coefficients (β_const, β_C_M, β_L_90)
  3. Gate accuracy / sensitivity / specificity
  4. POD parameters (a50, a90)
  5. Stratified POD (gate pass / fail)

방법: Stratified bootstrap (positive/negative class 비율 유지), 2000 iterations
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

ROOT = Path("/Users/lch/home/code/tunnelscanning")
WIDE = ROOT / "tmp/b2_out/wide_v2.csv"
LONG = ROOT / "tmp/b2_out/long_v2.csv"
OUT = ROOT / "01_tunnelscanning/04_data/b2_results"

N_BOOT = 2000
RNG = np.random.default_rng(42)

df_w = pd.read_csv(WIDE)
df_l = pd.read_csv(LONG)
y = df_w['identifiable_bin'].values

# ---------------------------------------------------------------------------
# Stratified bootstrap helper
# ---------------------------------------------------------------------------
def stratified_bootstrap_indices(y, rng):
    pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
    return np.concatenate([
        rng.choice(pos, size=len(pos), replace=True),
        rng.choice(neg, size=len(neg), replace=True),
    ])

def cluster_bootstrap_indices(df, group_col, rng):
    """Cluster bootstrap (resample by condition, keep all widths within)"""
    groups = df.groupby(group_col).indices
    keys = list(groups.keys())
    sampled_pos = rng.integers(0, len(keys), size=len(keys))
    idx = np.concatenate([groups[keys[i]] for i in sampled_pos])
    return idx

def ci_from_samples(samples, alpha=0.05):
    samples = np.asarray(samples)
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0: return (np.nan, np.nan, np.nan)
    return (
        float(np.median(samples)),
        float(np.percentile(samples, 100*alpha/2)),
        float(np.percentile(samples, 100*(1-alpha/2))),
    )

results = {}

# ===========================================================================
# 1) ROC AUC bootstrap (univariate)
# ===========================================================================
print("="*80)
print("[1] ROC AUC Bootstrap CI (univariate)")
print("="*80)

auc_results = {}
for p in ['C_M', 'L_90']:
    samples = []
    x_full = df_w[p].astype(float).values
    for _ in range(N_BOOT):
        idx = stratified_bootstrap_indices(y, RNG)
        try:
            a = roc_auc_score(y[idx], x_full[idx])
            # 두 방향 모두 시도, 큰 값 채택 (predictor sign 미정)
            a2 = roc_auc_score(y[idx], -x_full[idx])
            samples.append(max(a, a2))
        except: continue
    med, lo, hi = ci_from_samples(samples)
    print(f"  {p:>6}: AUC = {med:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n_valid={len(samples)}")
    auc_results[p] = {'median': med, 'ci_lower': lo, 'ci_upper': hi, 'n_valid': len(samples)}

results['auc_univariate'] = auc_results

# ===========================================================================
# 2) Multivariable logistic AUC + coefficient bootstrap
# ===========================================================================
print("\n" + "="*80)
print("[2] Multivariable Logistic Regression Bootstrap CI")
print("="*80)

X_full = sm.add_constant(df_w[['C_M', 'L_90']].astype(float)).values
auc_combined_samples = []
const_samples, cm_samples, l90_samples = [], [], []
n_converged = 0

for _ in range(N_BOOT):
    idx = stratified_bootstrap_indices(y, RNG)
    try:
        res_b = sm.Logit(y[idx], X_full[idx]).fit(disp=False, maxiter=200)
        if not res_b.mle_retvals.get('converged', True): continue
        n_converged += 1
        prob = res_b.predict(X_full)  # predict on full dataset (out-of-bag-ish)
        auc_combined_samples.append(roc_auc_score(y, prob))
        const_samples.append(res_b.params[0])
        cm_samples.append(res_b.params[1])
        l90_samples.append(res_b.params[2])
    except Exception:
        continue

print(f"  n_converged: {n_converged}/{N_BOOT}")
print(f"  AUC (combined, in-sample evaluated):  median={np.median(auc_combined_samples):.3f}  95% CI=[{np.percentile(auc_combined_samples, 2.5):.3f}, {np.percentile(auc_combined_samples, 97.5):.3f}]")
print(f"  β_const:  median={np.median(const_samples):.2f}  95% CI=[{np.percentile(const_samples, 2.5):.2f}, {np.percentile(const_samples, 97.5):.2f}]")
print(f"  β_C_M  :  median={np.median(cm_samples):.2f}  95% CI=[{np.percentile(cm_samples, 2.5):.2f}, {np.percentile(cm_samples, 97.5):.2f}]")
print(f"  β_L_90 :  median={np.median(l90_samples):.4f}  95% CI=[{np.percentile(l90_samples, 2.5):.4f}, {np.percentile(l90_samples, 97.5):.4f}]")

results['logistic_bootstrap'] = {
    'n_converged': n_converged,
    'auc_combined': dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(auc_combined_samples))),
    'beta_const':   dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(const_samples))),
    'beta_C_M':     dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(cm_samples))),
    'beta_L_90':    dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(l90_samples))),
}

# ===========================================================================
# 3) Dual-criterion gate metrics bootstrap
# ===========================================================================
print("\n" + "="*80)
print("[3] Dual-criterion Gate Metrics Bootstrap CI")
print("="*80)

acc_samples, sens_samples, spec_samples = [], [], []
for _ in range(N_BOOT):
    idx = stratified_bootstrap_indices(y, RNG)
    yb = y[idx]
    pred = ((df_w['C_M'].iloc[idx]>=0.05) & (df_w['L_90'].iloc[idx]>=55)).astype(int).values
    cm_b = confusion_matrix(yb, pred, labels=[0,1])
    if cm_b.shape != (2,2): continue
    tn, fp, fn, tp = cm_b.ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    sens = tp/(tp+fn) if (tp+fn)>0 else np.nan
    spec = tn/(tn+fp) if (tn+fp)>0 else np.nan
    acc_samples.append(acc); sens_samples.append(sens); spec_samples.append(spec)

for name, s in [('Accuracy', acc_samples), ('Sensitivity', sens_samples), ('Specificity', spec_samples)]:
    med, lo, hi = ci_from_samples(s)
    print(f"  {name:>12}: {med:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

results['gate_metrics_bootstrap'] = {
    'accuracy':    dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(acc_samples))),
    'sensitivity': dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(sens_samples))),
    'specificity': dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(spec_samples))),
}

# ===========================================================================
# 4) POD curve parameters (pooled, monotonic)
# ===========================================================================
print("\n" + "="*80)
print("[4] POD Curve Parameters Bootstrap CI (pooled, cluster bootstrap)")
print("="*80)

a50_samples, a90_samples, b0_samples, b1_samples = [], [], [], []
n_pod_conv = 0
for _ in range(N_BOOT):
    # Cluster bootstrap by condition (resample 50 conditions with replacement)
    idx = cluster_bootstrap_indices(df_l, ['speed','iso','dist'], RNG)
    sub = df_l.iloc[idx]
    if sub['detected'].nunique() < 2: continue
    try:
        Xp = sm.add_constant(np.log(sub['width_mm'].values))
        rs = sm.Logit(sub['detected'].values, Xp).fit(disp=False, maxiter=200)
        if not rs.mle_retvals.get('converged', True): continue
        b0, b1 = rs.params
        if b1 == 0: continue
        a50 = float(np.exp((np.log(0.5/0.5) - b0)/b1))
        a90 = float(np.exp((np.log(0.9/0.1) - b0)/b1))
        a50_samples.append(a50); a90_samples.append(a90)
        b0_samples.append(b0); b1_samples.append(b1)
        n_pod_conv += 1
    except: continue

print(f"  n_converged: {n_pod_conv}/{N_BOOT}")
print(f"  β₀ (intercept): median={np.median(b0_samples):.3f}  95% CI=[{np.percentile(b0_samples, 2.5):.3f}, {np.percentile(b0_samples, 97.5):.3f}]")
print(f"  β₁ (slope)   : median={np.median(b1_samples):.3f}  95% CI=[{np.percentile(b1_samples, 2.5):.3f}, {np.percentile(b1_samples, 97.5):.3f}]")
print(f"  a50 (mm)    : median={np.median(a50_samples):.3f}  95% CI=[{np.percentile(a50_samples, 2.5):.3f}, {np.percentile(a50_samples, 97.5):.3f}]")
print(f"  a90 (mm)    : median={np.median(a90_samples):.3f}  95% CI=[{np.percentile(a90_samples, 2.5):.3f}, {np.percentile(a90_samples, 97.5):.3f}]")
print(f"  → a90/95 (NDT-style 90% POD with 95% upper CL): {np.percentile(a90_samples, 95):.3f} mm")

results['pod_pooled_bootstrap'] = {
    'n_converged': n_pod_conv,
    'beta_0': dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(b0_samples))),
    'beta_1': dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(b1_samples))),
    'a50_mm': dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(a50_samples))),
    'a90_mm': dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(a90_samples))),
    'a90_95_mm': float(np.percentile(a90_samples, 95)),
}

# ===========================================================================
# 5) Stratified POD (gate pass)
# ===========================================================================
print("\n" + "="*80)
print("[5] POD Stratified by Gate (gate=pass only, gate=fail uniformly 0)")
print("="*80)

df_l['gate'] = ((df_l['C_M']>=0.05) & (df_l['L_90']>=55)).astype(int)
df_l_pass = df_l[df_l['gate']==1]
df_l_fail = df_l[df_l['gate']==0]
print(f"  gate pass rows: {len(df_l_pass)} (detection rate: {df_l_pass['detected'].mean():.3f})")
print(f"  gate fail rows: {len(df_l_fail)} (detection rate: {df_l_fail['detected'].mean():.3f})")

a50p_samples, a90p_samples = [], []
b0p_samples, b1p_samples = [], []
n_pod_pass_conv = 0
for _ in range(N_BOOT):
    idx = cluster_bootstrap_indices(df_l_pass, ['speed','iso','dist'], RNG)
    sub = df_l_pass.iloc[idx]
    if sub['detected'].nunique() < 2: continue
    try:
        Xp = sm.add_constant(np.log(sub['width_mm'].values))
        rs = sm.Logit(sub['detected'].values, Xp).fit(disp=False, maxiter=200)
        if not rs.mle_retvals.get('converged', True): continue
        b0, b1 = rs.params
        if b1 == 0: continue
        a50 = float(np.exp((np.log(0.5/0.5) - b0)/b1))
        a90 = float(np.exp((np.log(0.9/0.1) - b0)/b1))
        a50p_samples.append(a50); a90p_samples.append(a90)
        b0p_samples.append(b0); b1p_samples.append(b1)
        n_pod_pass_conv += 1
    except: continue

print(f"  n_converged (gate pass): {n_pod_pass_conv}/{N_BOOT}")
print(f"  β₀ (intercept): median={np.median(b0p_samples):.3f}  95% CI=[{np.percentile(b0p_samples, 2.5):.3f}, {np.percentile(b0p_samples, 97.5):.3f}]")
print(f"  β₁ (slope)   : median={np.median(b1p_samples):.3f}  95% CI=[{np.percentile(b1p_samples, 2.5):.3f}, {np.percentile(b1p_samples, 97.5):.3f}]")
print(f"  a50 (mm)    : median={np.median(a50p_samples):.3f}  95% CI=[{np.percentile(a50p_samples, 2.5):.3f}, {np.percentile(a50p_samples, 97.5):.3f}]")
print(f"  a90 (mm)    : median={np.median(a90p_samples):.3f}  95% CI=[{np.percentile(a90p_samples, 2.5):.3f}, {np.percentile(a90p_samples, 97.5):.3f}]")
print(f"  → a90/95 (gate pass only): {np.percentile(a90p_samples, 95):.3f} mm")

results['pod_gate_pass_bootstrap'] = {
    'n_converged': n_pod_pass_conv,
    'beta_0': dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(b0p_samples))),
    'beta_1': dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(b1p_samples))),
    'a50_mm': dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(a50p_samples))),
    'a90_mm': dict(zip(['median','ci_lower','ci_upper'], ci_from_samples(a90p_samples))),
    'a90_95_mm': float(np.percentile(a90p_samples, 95)),
}

results['pod_gate_fail'] = {
    'note': 'All gate-fail conditions have detection_rate=0 (no variability for POD fitting).',
    'n_rows': len(df_l_fail),
    'detection_rate': float(df_l_fail['detected'].mean()),
}

# ===========================================================================
# Save
# ===========================================================================
with open(OUT / "bootstrap_ci.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print(f"Saved: {OUT/'bootstrap_ci.json'}")
print("="*80)
