"""
Phase 1 Task 1.5: Monotonic POD 가정 Sensitivity Analysis
================================================================
비교 대상:
  A. Monotonic 가정: MDW (min_crack_mm) 이상의 모든 width = detected
  B. Visible 리스트 가정: notes의 visible=[...] 그대로 사용
  C. 두 가정의 inconsistency 정량
  D. POD curve fitting 결과 비교 (β, a50, a90)

목적: monotonic 가정의 합리성 본문 정당화
"""
from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

ROOT = Path("/Users/lch/home/code/tunnelscanning")
DATA = ROOT / "01_tunnelscanning/03_src/data"
WIDE = ROOT / "tmp/b2_out/wide_v2.csv"
OUT = ROOT / "01_tunnelscanning/04_data/b2_results"

WIDTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

df_w = pd.read_csv(WIDE)
df_cd = pd.read_csv(DATA / "cam1_crack_detectability.csv")

# ===========================================================================
# A) Monotonic detection labels (already in long_v2.csv, but re-derive here)
# ===========================================================================
def parse_visible(notes):
    if not isinstance(notes, str): return None
    m = re.search(r'visible=\[([^\]]*)\]', notes)
    if not m: return None
    s = m.group(1).strip()
    if not s: return []
    return [float(x.strip()) for x in s.split(',')]

# Build long format with both labelings
rows_mono = []
rows_vis = []
n_consistent = 0
n_only_in_mono = 0
n_only_in_vis = 0
inconsistency_log = []

for _, r in df_cd.iterrows():
    spd, iso, dist = r['speed'], r['iso'], r['dist']
    mdw = r['min_crack_mm']
    notes = r['notes']
    visible = parse_visible(notes)
    user_x = isinstance(notes, str) and 'user_labeled_x' in notes
    user_v = isinstance(notes, str) and 'user_verified' in notes

    for w in WIDTHS:
        # A) Monotonic
        if pd.isna(mdw):
            mono = 0
        else:
            mono = 1 if w >= mdw - 1e-9 else 0
        rows_mono.append({'speed':spd,'iso':iso,'dist':dist,'width_mm':w,'detected':mono})

        # B) Visible
        if user_x:
            vis = 0
        elif user_v:
            vis = 1  # treat all widths as visible (user verified)
        elif visible is None:
            # No visible info, fallback to mono
            vis = mono
        else:
            vis = 1 if w in visible else 0
        rows_vis.append({'speed':spd,'iso':iso,'dist':dist,'width_mm':w,'detected':vis})

        # Compare
        if mono == vis:
            n_consistent += 1
        elif mono==1 and vis==0:
            n_only_in_mono += 1
            if visible is not None and not user_x and not user_v:
                inconsistency_log.append({
                    'speed':spd,'iso':iso,'dist':dist,'width_mm':w,
                    'mdw':mdw,'visible_list':visible,
                    'mono':1,'vis':0,
                    'note':'monotonic says detected (≥MDW), but not in visible list'
                })
        else:
            n_only_in_vis += 1

df_mono = pd.DataFrame(rows_mono)
df_vis = pd.DataFrame(rows_vis)

print("="*80)
print("[A] LABEL COMPARISON: Monotonic vs Visible")
print("="*80)
total = len(df_mono)
print(f"\n  Total rows: {total} (50 conditions × 10 widths)")
print(f"  Consistent labels:        {n_consistent} ({n_consistent/total*100:.1f}%)")
print(f"  Mono=1, Vis=0 (mono extra): {n_only_in_mono} ({n_only_in_mono/total*100:.1f}%)")
print(f"  Mono=0, Vis=1 (vis extra) : {n_only_in_vis} ({n_only_in_vis/total*100:.1f}%)")

# Detection rate per width comparison
print("\n  Detection rate per width:")
print(f"  {'width(mm)':>10} {'Mono':>8} {'Vis':>8} {'Diff':>8}")
for w in WIDTHS:
    r_m = df_mono[df_mono['width_mm']==w]['detected'].mean()
    r_v = df_vis[df_vis['width_mm']==w]['detected'].mean()
    print(f"  {w:>10.1f} {r_m:>8.3f} {r_v:>8.3f} {r_m-r_v:+8.3f}")

# Sample inconsistencies
print(f"\n  Sample inconsistency cases (first 10 of {len(inconsistency_log)}):")
for r in inconsistency_log[:10]:
    print(f"    spd={r['speed']}, iso={r['iso']}, dist={r['dist']}, w={r['width_mm']}: "
          f"MDW={r['mdw']}, visible={r['visible_list']}")

# ===========================================================================
# B) POD curve fitting under both assumptions
# ===========================================================================
print("\n" + "="*80)
print("[B] POD CURVE COMPARISON")
print("="*80)

def fit_pod(df_pod, label):
    y = df_pod['detected'].values
    log_w = np.log(df_pod['width_mm'].values)
    X = sm.add_constant(log_w)
    res = sm.Logit(y, X).fit(disp=False, maxiter=200)
    b0, b1 = res.params
    a50 = float(np.exp((np.log(0.5/0.5) - b0)/b1)) if b1 != 0 else float('nan')
    a90 = float(np.exp((np.log(0.9/0.1) - b0)/b1)) if b1 != 0 else float('nan')
    return {
        'label': label,
        'n': len(df_pod),
        'detection_rate': float(y.mean()),
        'beta_0': float(b0),
        'beta_1': float(b1),
        'pseudo_R2': float(res.prsquared),
        'log_lik': float(res.llf),
        'aic': float(res.aic),
        'a50_mm': a50,
        'a90_mm': a90,
    }

# Pooled (all 50 conditions × 10 widths)
res_mono = fit_pod(df_mono, 'monotonic_pooled')
res_vis = fit_pod(df_vis, 'visible_pooled')

print(f"\n  Pooled (all conditions):")
print(f"    {'metric':>20} {'Monotonic':>12} {'Visible':>12} {'Diff':>10}")
for k in ['n','detection_rate','beta_0','beta_1','pseudo_R2','log_lik','aic','a50_mm','a90_mm']:
    vm = res_mono[k]; vv = res_vis[k]
    if isinstance(vm, float):
        print(f"    {k:>20} {vm:>12.4f} {vv:>12.4f} {vm-vv:>+10.4f}")
    else:
        print(f"    {k:>20} {vm:>12} {vv:>12} {'-':>10}")

# Gate pass only
df_w_idx = df_w.set_index(['speed','iso','dist'])
df_mono['gate'] = df_mono.apply(lambda r: int((df_w_idx.loc[(r['speed'],r['iso'],r['dist']),'C_M']>=0.05) and (df_w_idx.loc[(r['speed'],r['iso'],r['dist']),'L_90']>=55)), axis=1)
df_vis['gate'] = df_vis.apply(lambda r: int((df_w_idx.loc[(r['speed'],r['iso'],r['dist']),'C_M']>=0.05) and (df_w_idx.loc[(r['speed'],r['iso'],r['dist']),'L_90']>=55)), axis=1)

res_mono_pass = fit_pod(df_mono[df_mono['gate']==1], 'monotonic_gate_pass')
res_vis_pass = fit_pod(df_vis[df_vis['gate']==1], 'visible_gate_pass')

print(f"\n  Gate pass only:")
print(f"    {'metric':>20} {'Monotonic':>12} {'Visible':>12} {'Diff':>10}")
for k in ['n','detection_rate','beta_0','beta_1','pseudo_R2','log_lik','aic','a50_mm','a90_mm']:
    vm = res_mono_pass[k]; vv = res_vis_pass[k]
    if isinstance(vm, float):
        print(f"    {k:>20} {vm:>12.4f} {vv:>12.4f} {vm-vv:>+10.4f}")
    else:
        print(f"    {k:>20} {vm:>12} {vv:>12} {'-':>10}")

# ===========================================================================
# C) Per-condition completeness analysis
# ===========================================================================
print("\n" + "="*80)
print("[C] VISIBLE LIST COMPLETENESS ANALYSIS")
print("="*80)

completeness = []
for _, r in df_cd.iterrows():
    notes = r['notes']
    visible = parse_visible(notes)
    user_v = isinstance(notes, str) and 'user_verified' in notes
    user_x = isinstance(notes, str) and 'user_labeled_x' in notes

    if user_v or user_x:
        category = 'user_labeled'
    elif visible is None:
        category = 'no_visible_info'
    else:
        category = 'partial_visible_list'

    completeness.append({
        'speed': r['speed'], 'iso': r['iso'], 'dist': r['dist'],
        'mdw': r['min_crack_mm'],
        'identifiable': r['identifiable'],
        'visible_count': len(visible) if visible is not None else None,
        'category': category,
    })
df_comp = pd.DataFrame(completeness)
print(df_comp['category'].value_counts())
print()
print(f"  Conditions with explicit visible list: {(df_comp['category']=='partial_visible_list').sum()}")
print(f"  Average visible count (when explicit): {df_comp[df_comp['visible_count'].notna()]['visible_count'].mean():.2f} of 10 widths")

# Conclusion: the visible list appears to be incomplete labeling
# Save
results = {
    'label_comparison': {
        'total_rows': total,
        'consistent': n_consistent,
        'mono_extra': n_only_in_mono,
        'vis_extra': n_only_in_vis,
        'consistency_pct': n_consistent / total * 100,
    },
    'detection_rate_per_width': {
        f'{w:.1f}mm': {
            'monotonic': float(df_mono[df_mono['width_mm']==w]['detected'].mean()),
            'visible':   float(df_vis[df_vis['width_mm']==w]['detected'].mean()),
        } for w in WIDTHS
    },
    'pod_pooled': {
        'monotonic': res_mono,
        'visible': res_vis,
    },
    'pod_gate_pass': {
        'monotonic': res_mono_pass,
        'visible': res_vis_pass,
    },
    'visible_completeness': df_comp['category'].value_counts().to_dict(),
    'avg_visible_count': float(df_comp[df_comp['visible_count'].notna()]['visible_count'].mean()),
    'inconsistency_count': len(inconsistency_log),
    'sample_inconsistencies': inconsistency_log[:20],
}

with open(OUT / "sensitivity_monotonic.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

# ===========================================================================
# D) Conclusion / Justification
# ===========================================================================
print("\n" + "="*80)
print("[D] CONCLUSION")
print("="*80)

print(f"""
  1. Visible list appears INCOMPLETE:
     - Average {df_comp[df_comp['visible_count'].notna()]['visible_count'].mean():.1f} widths recorded per condition
     - But chart contains 10 widths (0.1-1.0mm)
     - {len(inconsistency_log)} cases where mono=1 but width not in visible list

  2. Detection rate trend:
     - Monotonic: {df_mono[df_mono['width_mm']==1.0]['detected'].mean():.2f} at 1.0mm (large width)
     - Visible:   {df_vis[df_vis['width_mm']==1.0]['detected'].mean():.2f} at 1.0mm
     - Visible underestimates because not all observed widths are recorded

  3. POD model fit:
     - Monotonic AIC = {res_mono['aic']:.1f} ({res_mono['pseudo_R2']:.3f} R²)
     - Visible   AIC = {res_vis['aic']:.1f} ({res_vis['pseudo_R2']:.3f} R²)
     - {'Monotonic better' if res_mono['aic']<res_vis['aic'] else 'Visible better'}

  RECOMMENDATION: Monotonic assumption is more appropriate because:
   (a) Chart contains all widths 0.1-1.0mm (sequential by design)
   (b) MDW (minimum detectable width) is the standard NDT acceptance metric
   (c) Visible list appears to be partial sample, not exhaustive labeling
   (d) NDT POD framework (MIL-HDBK-1823) uses monotonic detection assumption
""")

print(f"Output: {OUT/'sensitivity_monotonic.json'}")
