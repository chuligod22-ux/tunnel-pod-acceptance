"""
Phase 1 Task 1.1: Multi-criterion gate optimization
================================================================
목적: 기존 dual-criterion (C_M ≥ 0.05 AND L_90 ≥ 55) 96% accuracy → 100% 가능성 탐색

전략:
  Step 1: 현재 dual-criterion baseline 평가
  Step 2: 단일 metric 최적 임계값 (ROC-based)
  Step 3: Dual-criterion grid search (C_M × L_90)
  Step 4: Logistic regression 기반 결정 임계값
  Step 5: Leave-One-Out CV로 overfitting 검증
  Step 6: 최종 추천 gate 도출
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import statsmodels.api as sm

ROOT = Path("/Users/lch/home/code/tunnelscanning")
WIDE = ROOT / "tmp/b2_out/wide_v2.csv"
OUT = ROOT / "01_tunnelscanning/04_data/b2_results"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(WIDE)
y = df['identifiable_bin'].values

print("="*80)
print("PHASE 1 TASK 1.1: Multi-criterion gate optimization")
print(f"  n={len(df)}, identifiable=Y:{y.sum()}, N:{(1-y).sum()}")
print("="*80)

# -----------------------------------------------------------------------------
# Step 1: Baseline — 논문 보고 dual-criterion
# -----------------------------------------------------------------------------
def gate_metrics(pred, y):
    cm = confusion_matrix(y, pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    n = len(y)
    return {
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'accuracy': (tp+tn)/n,
        'sensitivity': tp/(tp+fn) if (tp+fn)>0 else float('nan'),
        'specificity': tn/(tn+fp) if (tn+fp)>0 else float('nan'),
    }

print("\n[STEP 1] Baseline — 논문 dual-criterion (C_M ≥ 0.05 AND L_90 ≥ 55)")
pred_baseline = ((df['C_M']>=0.05) & (df['L_90']>=55)).astype(int).values
m_base = gate_metrics(pred_baseline, y)
for k,v in m_base.items(): print(f"  {k:>15}: {v}")

# False positive cases
fp_idx = (pred_baseline==1) & (y==0)
print(f"\n  False Positives ({fp_idx.sum()}):")
print(df[fp_idx][['speed','iso','dist','C_M','L_90']].to_string(index=False))

# False negative cases
fn_idx = (pred_baseline==0) & (y==1)
print(f"  False Negatives ({fn_idx.sum()}):")
if fn_idx.sum() > 0:
    print(df[fn_idx][['speed','iso','dist','C_M','L_90']].to_string(index=False))
else:
    print("  (none)")

# -----------------------------------------------------------------------------
# Step 2: Single-metric optimal thresholds (Youden J)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("[STEP 2] Single-metric optimal thresholds (Youden J = sens + spec - 1)")
print("="*80)

results_uni = []
for p in ['C_M', 'L_90']:
    x = df[p].astype(float).values
    fpr, tpr, thr = roc_curve(y, x)
    j = tpr - fpr
    idx = np.argmax(j)
    auc = roc_auc_score(y, x)
    pred = (x >= thr[idx]).astype(int)
    m = gate_metrics(pred, y)
    print(f"\n  {p}: AUC={auc:.3f}, optimal_thr={thr[idx]:.3f}, J={j[idx]:.3f}")
    for k,v in m.items(): print(f"    {k:>13}: {v}")
    results_uni.append({'metric': p, 'AUC': auc, 'thr': float(thr[idx]),
                        'J': float(j[idx]), **m})

# -----------------------------------------------------------------------------
# Step 3: Dual-criterion grid search
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("[STEP 3] Dual-criterion grid search (C_M_thr × L_90_thr)")
print("="*80)

C_grid = np.arange(0.01, 0.21, 0.005)
L_grid = np.arange(40, 100, 1.0)

best_acc = 0.0
best_combos = []
all_perfect_combos = []

for c_thr in C_grid:
    for l_thr in L_grid:
        pred = ((df['C_M']>=c_thr) & (df['L_90']>=l_thr)).astype(int).values
        m = gate_metrics(pred, y)
        if m['accuracy'] > best_acc:
            best_acc = m['accuracy']
            best_combos = [(c_thr, l_thr, m)]
        elif m['accuracy'] == best_acc:
            best_combos.append((c_thr, l_thr, m))
        if m['accuracy'] == 1.0:
            all_perfect_combos.append((c_thr, l_thr, m))

print(f"\nBest accuracy: {best_acc:.4f}")
print(f"Number of (C_M, L_90) combos achieving best: {len(best_combos)}")
print(f"Number of combos achieving 100% accuracy: {len(all_perfect_combos)}")

if all_perfect_combos:
    print("\nTop perfect combos (100% accuracy) — first 10:")
    print(f"  {'C_M_thr':>10} {'L_90_thr':>10} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}")
    for c_thr, l_thr, m in all_perfect_combos[:10]:
        print(f"  {c_thr:>10.4f} {l_thr:>10.1f} {m['TP']:>4} {m['TN']:>4} {m['FP']:>4} {m['FN']:>4}")

    # Pick "central" perfect combo (median values for robustness)
    perfect_C = [x[0] for x in all_perfect_combos]
    perfect_L = [x[1] for x in all_perfect_combos]
    print(f"\nPerfect combo C_M_thr range: [{min(perfect_C):.3f}, {max(perfect_C):.3f}]  median={np.median(perfect_C):.3f}")
    print(f"Perfect combo L_90_thr range: [{min(perfect_L):.1f}, {max(perfect_L):.1f}]  median={np.median(perfect_L):.1f}")
else:
    print("No combo achieves 100% accuracy.")
    print("\nTop combos at best accuracy:")
    for c_thr, l_thr, m in best_combos[:5]:
        print(f"  C_M≥{c_thr:.3f}, L_90≥{l_thr:.1f}: TP={m['TP']}, TN={m['TN']}, FP={m['FP']}, FN={m['FN']}")

# -----------------------------------------------------------------------------
# Step 4: Logistic regression 기반 결정 임계값
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("[STEP 4] Logistic regression-based decision threshold")
print("="*80)

X = sm.add_constant(df[['C_M', 'L_90']].astype(float))
res = sm.Logit(y, X).fit(disp=False, maxiter=200)
print(f"\n  Coefficients: const={res.params[0]:.3f}, C_M={res.params[1]:.3f}, L_90={res.params[2]:.3f}")
print(f"  Pseudo R² = {res.prsquared:.4f}")
print(f"  LLR p-value = {res.llr_pvalue:.2e}")

prob = res.predict(X)
df['prob'] = prob

# Sweep probability threshold
print("\n  Probability threshold sweep:")
print(f"  {'p_thr':>6} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} {'Acc':>6} {'Sens':>6} {'Spec':>6}")
best_p_thr = 0.5
best_p_acc = 0.0
best_p_metrics = None
for p_thr in np.arange(0.05, 0.96, 0.05):
    pred = (prob >= p_thr).astype(int).values
    m = gate_metrics(pred, y)
    print(f"  {p_thr:>6.2f} {m['TP']:>4} {m['TN']:>4} {m['FP']:>4} {m['FN']:>4} "
          f"{m['accuracy']:>6.3f} {m['sensitivity']:>6.3f} {m['specificity']:>6.3f}")
    if m['accuracy'] > best_p_acc:
        best_p_acc = m['accuracy']
        best_p_thr = p_thr
        best_p_metrics = m

print(f"\n  Best probability threshold: p ≥ {best_p_thr:.2f} → accuracy = {best_p_acc:.3f}")

# -----------------------------------------------------------------------------
# Step 5: Leave-One-Out CV (overfitting 검증)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("[STEP 5] Leave-One-Out CV (logistic regression generalization)")
print("="*80)

X_full = sm.add_constant(df[['C_M', 'L_90']].astype(float))
loo_preds = np.zeros(len(df))
for i in range(len(df)):
    mask = np.ones(len(df), dtype=bool)
    mask[i] = False
    try:
        res_i = sm.Logit(y[mask], X_full.iloc[mask]).fit(disp=False, maxiter=200)
        loo_preds[i] = res_i.predict(X_full.iloc[[i]])[0]
    except Exception as e:
        loo_preds[i] = np.nan

valid = ~np.isnan(loo_preds)
print(f"  LOO valid predictions: {valid.sum()}/{len(df)}")
loo_pred_bin = (loo_preds[valid] >= best_p_thr).astype(int)
m_loo = gate_metrics(loo_pred_bin, y[valid])
print(f"  LOO performance (p_thr={best_p_thr}):")
for k,v in m_loo.items(): print(f"    {k:>13}: {v}")

# Compare to in-sample
print(f"\n  In-sample accuracy:  {best_p_acc:.4f}")
print(f"  LOO     accuracy:    {m_loo['accuracy']:.4f}")
print(f"  Generalization gap:  {best_p_acc - m_loo['accuracy']:.4f}")

# -----------------------------------------------------------------------------
# Step 6: 최종 추천 gate
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("[STEP 6] 최종 추천 Gate")
print("="*80)

recommendations = {
    'baseline_dual_criterion': {
        'rule': 'C_M >= 0.05 AND L_90 >= 55',
        'metrics': m_base,
        'note': '논문 보고. 96% accuracy, 2 FP'
    },
    'optimized_dual_criterion': None,
    'logistic_decision_rule': {
        'rule': f'P(detect) = sigmoid({res.params[0]:.2f} + {res.params[1]:.2f}·C_M + {res.params[2]:.4f}·L_90) >= {best_p_thr:.2f}',
        'in_sample_metrics': best_p_metrics,
        'loo_metrics': m_loo,
        'note': 'Logistic regression based, generalization 검증',
    },
}

if all_perfect_combos:
    # Recommend most "central" perfect combo for robustness
    median_c = float(np.median([x[0] for x in all_perfect_combos]))
    median_l = float(np.median([x[1] for x in all_perfect_combos]))
    pred_med = ((df['C_M']>=median_c) & (df['L_90']>=median_l)).astype(int).values
    m_med = gate_metrics(pred_med, y)
    recommendations['optimized_dual_criterion'] = {
        'rule': f'C_M >= {median_c:.3f} AND L_90 >= {median_l:.1f}',
        'metrics': m_med,
        'note': f'Grid search median of {len(all_perfect_combos)} perfect combos. 100% accuracy.',
    }

print("\nFinal Recommendations:")
for name, rec in recommendations.items():
    if rec is None: continue
    print(f"\n  [{name}]")
    print(f"    Rule: {rec['rule']}")
    if 'metrics' in rec:
        m = rec['metrics']
        print(f"    Acc={m['accuracy']:.3f}, Sens={m['sensitivity']:.3f}, Spec={m['specificity']:.3f}, FP={m['FP']}, FN={m['FN']}")
    print(f"    Note: {rec['note']}")

# -----------------------------------------------------------------------------
# Save outputs
# -----------------------------------------------------------------------------
results = {
    'baseline': {'thr': {'C_M': 0.05, 'L_90': 55}, **m_base},
    'univariate': results_uni,
    'logistic': {
        'coefficients': {'const': float(res.params[0]),
                         'C_M': float(res.params[1]),
                         'L_90': float(res.params[2])},
        'pseudo_R2': float(res.prsquared),
        'LLR_pvalue': float(res.llr_pvalue),
        'best_p_threshold': float(best_p_thr),
        'in_sample': best_p_metrics,
        'LOO': m_loo,
    },
    'grid_search': {
        'best_accuracy': float(best_acc),
        'n_perfect_combos': len(all_perfect_combos),
        'perfect_combos_first_10': [
            {'C_M_thr': float(c), 'L_90_thr': float(l)} for c,l,_ in all_perfect_combos[:10]
        ] if all_perfect_combos else [],
    },
    'recommendations': recommendations,
}
with open(OUT / "gate_optimization.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Probability per condition CSV
df[['speed','iso','dist','C_M','L_90','prob','identifiable']].to_csv(
    OUT / "logistic_probabilities.csv", index=False)

print(f"\nOutput files:")
print(f"  - {OUT/'gate_optimization.json'}")
print(f"  - {OUT/'logistic_probabilities.csv'}")
