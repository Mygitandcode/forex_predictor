"""Full analysis of completed Stage2 Optuna study"""
import optuna
import numpy as np

study = optuna.load_study(
    study_name='EURUSD_15m_xgboost',
    storage='sqlite:///optuna_EURUSD_15m_xgboost.db'
)

print("=" * 70)
print("  STAGE2 OPTUNA STUDY — FULL REPORT")
print("=" * 70)

# Basic stats
all_trials = list(study.trials)
completed = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned = [t for t in all_trials if t.state == optuna.trial.TrialState.PRUNED]
failed = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]
running = [t for t in all_trials if t.state == optuna.trial.TrialState.RUNNING]

print(f"\nTotal trials: {len(all_trials)}")
print(f"  COMPLETE: {len(completed)}")
print(f"  PRUNED:   {len(pruned)}")
print(f"  FAIL:     {len(failed)}")
print(f"  RUNNING:  {len(running)}")

# Best
print(f"\nBest trial: #{study.best_trial.number}")
print(f"Best value (Val AUC): {study.best_value:.6f}")
print(f"\nBest params:")
for k, v in sorted(study.best_params.items()):
    print(f"  {k}: {v}")

# All completed trials sorted by AUC
print(f"\n{'='*70}")
print(f"  ALL COMPLETED TRIALS (sorted by Val AUC)")
print(f"{'='*70}")
print(f"{'#':>4} {'AUC':>8} {'lr':>8} {'dp':>3} {'trees':>6} {'TrA':>7} {'Sig%':>7} {'time_s':>7}")
print(f"{'-'*4} {'-'*8} {'-'*8} {'-'*3} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")

completed_sorted = sorted(completed, key=lambda t: t.value, reverse=True)
for t in completed_sorted[:30]:
    p = t.params
    # Extract metrics from user_attrs if available, otherwise use trial metadata
    n_est = p.get('n_estimators', '?')
    print(f"{t.number:>4} {t.value:>8.4f} {p['learning_rate']:>8.4f} {p['max_depth']:>3} "
          f"{n_est:>6} {'...':>7} {'...':>7} {t.duration.seconds if t.duration else 0:>7}")

# Distribution of best params
print(f"\n{'='*70}")
print(f"  PARAMETER DISTRIBUTION (top 10 trials)")
print(f"{'='*70}")

top10 = completed_sorted[:10]
for param in ['learning_rate', 'max_depth', 'subsample', 'colsample_bytree', 
              'min_child_weight', 'reg_alpha', 'reg_lambda', 'gamma']:
    vals = [t.params[param] for t in top10]
    print(f"  {param:>20}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
          f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")

# AUC progression over trials
print(f"\n{'='*70}")
print(f"  AUC PROGRESSION (first trial -> best trial)")
print(f"{'='*70}")

completed_by_number = sorted(completed, key=lambda t: t.number)
aucs = [t.value for t in completed_by_number]
print(f"  Start AUC: {aucs[0]:.6f} (trial #{completed_by_number[0].number})")
print(f"  Best AUC:  {aucs[-1]:.6f} (trial #{completed_by_number[-1].number})")
print(f"  Improvement: +{aucs[-1] - aucs[0]:.6f} ({(aucs[-1]/aucs[0] - 1)*100:.2f}%)")
print(f"  Mean AUC:   {np.mean(aucs):.6f}")
print(f"  AUC Std:    {np.std(aucs):.6f}")

# Trial number distribution (to see if later trials are better)
first_half = [t for t in completed if t.number < 50]
second_half = [t for t in completed if t.number >= 50]
if first_half and second_half:
    print(f"\n  First 50 trials mean AUC:  {np.mean([t.value for t in first_half]):.6f} (n={len(first_half)})")
    print(f"  Second 50+ trials mean AUC: {np.mean([t.value for t in second_half]):.6f} (n={len(second_half)})")
    print(f"  Later trials improvement:   +{np.mean([t.value for t in second_half]) - np.mean([t.value for t in first_half]):.6f}")

# Number of trees (best_iteration proxy from trial)
print(f"\n{'='*70}")
print(f"  TRAINING EFFICIENCY (completed trials)")
print(f"{'='*70}")
durations = [t.duration.seconds for t in completed if t.duration]
if durations:
    print(f"  Mean duration: {np.mean(durations):.0f}s")
    print(f"  Min duration:  {np.min(durations):.0f}s")
    print(f"  Max duration:  {np.max(durations):.0f}s")
    print(f"  Std duration:  {np.std(durations):.0f}s")
    print(f"  Total time:    {sum(durations)/3600:.1f}h")

print(f"\n{'='*70}")
print(f"  PRUNED TRIALS ANALYSIS")
print(f"{'='*70}")
if pruned:
    print(f"  Count: {len(pruned)}")
    # Show what params get pruned
    pruned_params = [t.params for t in pruned[:10]]
    for param in ['learning_rate', 'max_depth', 'subsample']:
        vals = [p[param] for p in pruned_params]
        print(f"  {param:>20}: mean={np.mean(vals):.4f}, range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")
    print(f"\n  Pruned trial numbers: {[t.number for t in sorted(pruned, key=lambda t: t.number)]}")
