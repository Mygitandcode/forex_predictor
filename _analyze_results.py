"""
快速分析 Optuna study 结果：AUC 分布、Top-N、SHAP 特征贡献
用法: python _analyze_results.py --study v2_80t
"""
import sys
import argparse
import numpy as np
import optuna
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--study", default="v2_80t")
args = parser.parse_args()

study_name = f"EURUSD_15m_xgboost__{args.study}"
storage = f"sqlite:///optuna_{study_name}.db"

study = optuna.load_study(study_name=study_name, storage=storage)

completed = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
pruned = study.get_trials(states=[optuna.trial.TrialState.PRUNED])
failed = study.get_trials(states=[optuna.trial.TrialState.FAIL])

print(f"{'='*60}")
print(f"Study: {study_name}")
print(f"Completed: {len(completed)} | Pruned: {len(pruned)} | Failed: {len(failed)}")
print(f"{'='*60}")

# ---- 1. AUC 分布 ----
values = [t.value for t in completed if t.value is not None]
if not values:
    print("No completed trials with values found.")
    sys.exit(1)

values = np.array(values)
print(f"\n--- AUC Distribution ({len(values)} trials) ---")
print(f"  Mean:   {values.mean():.4f}")
print(f"  Median: {np.median(values):.4f}")
print(f"  Min:    {values.min():.4f}")
print(f"  Max:    {values.max():.4f}")
print(f"  Std:    {values.std():.4f}")

# Percentile bins
pcts = [10, 25, 50, 75, 90, 95]
print(f"  Percentiles: " + ", ".join(f"P{p}={np.percentile(values, p):.4f}" for p in pcts))

# Histogram bins
print(f"\n  Bins:")
bins = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 1.0]
for i in range(len(bins)-1):
    cnt = ((values >= bins[i]) & (values < bins[i+1])).sum()
    bar = "█" * max(1, cnt)
    print(f"  [{bins[i]:.2f},{bins[i+1]:.2f}): {cnt:3d} {bar}")

# ---- 2. Top-10 Trials ----
print(f"\n{'='*60}")
print(f"Top-10 Trials by AUC")
print(f"{'='*60}")
sorted_trials = sorted(completed, key=lambda t: t.value or 0, reverse=True)
print(f"{'Rank':<5} {'Trial':<6} {'AUC':<8} {'TrainA':<8} {'ValA':<8} {'Sig%':<8} {'Trees':<8} {'Gamma':<8} {'Time':<8}")
print("-" * 85)

for rank, t in enumerate(sorted_trials[:10], 1):
    p = t.params
    # Extract train AUC from MLflow metrics... we don't have that easily
    # Just show the key params
    val_auc = t.value or 0
    # Try to get user attrs or intermediate values
    gamma = p.get('gamma', 0)
    n_est = p.get('n_estimators', 0)
    duration = t.duration.seconds if t.duration else 0
    dur_m = f"{duration//60}m{duration%60}s" if duration else "N/A"

    # n_estimators_used may be stored in user_attrs
    n_used = t.user_attrs.get('n_estimators_used', n_est)

    print(f"{rank:<5} {t.number:<6} {val_auc:<8.4f} {'N/A':<8} {'N/A':<8} {'N/A':<8} "
          f"{n_used:<8} {gamma:<8.4f} {dur_m:<8}")

# ---- 3. SHAP Top 特征聚合 ----
print(f"\n{'='*60}")
print(f"Sample Trial Details (last completed)")
print(f"{'='*60}")

# Show the best trial details
best = study.best_trial
print(f"\nBest Trial (#{best.number}): AUC = {best.value:.4f}")
print(f"  Params: {best.params}")
print(f"  User Attrs: {dict(best.user_attrs)}")

# ---- 4. SHAP Top-10 from best trial (from MLflow tag) ----
# The SHAP top10 is stored as a tag in MLflow, not directly in Optuna.
# But we can try to get it from trial user_attrs if stored.
print(f"\n{'='*60}")
print(f"Params analysis across Top-10 trials")
print(f"{'='*60}")

# Aggregate param stats for top trials
top10 = sorted_trials[:10]
for param_name in ['gamma', 'n_estimators', 'max_depth', 'learning_rate', 'min_child_weight']:
    vals = [t.params[param_name] for t in top10 if param_name in t.params]
    if vals:
        print(f"  {param_name:>20s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

print(f"\nDone. Study file: optuna_{study_name}.db")
