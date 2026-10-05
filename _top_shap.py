"""Pull real SHAP top-10 from MLflow for best trials"""
import mlflow
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")
exp = mlflow.get_experiment_by_name("forex_stage2")

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="tags.mlflow.runName LIKE 'trial_%'",
)

# Collect trials with real SHAP data
trials = []
for _, run in runs.iterrows():
    shap = str(run.get("tags.shap_top10", ""))
    auc = run.get("metrics.val_auc", None)
    tn = run.get("metrics.trial_number", None)
    
    if tn is None or auc is None:
        continue
    if "SHAP_FAILED" in shap or shap == "nan" or not shap.strip():
        continue
    
    # Parse SHAP: "f208(0.0255), f209(0.0274), ..."
    items = shap.split(", ")
    feats = []
    for item in items:
        try:
            name, val = item.split("(")
            val = float(val.rstrip(")"))
            feats.append((name, val))
        except:
            pass
    
    trials.append({
        "number": int(tn),
        "auc": float(auc),
        "shap_top10": feats,
    })

# Sort by AUC descending
trials.sort(key=lambda t: t["auc"], reverse=True)

print("=" * 70)
print("  TOP 10 TRIALS — REAL SHAP FEATURE CONTRIBUTIONS")
print("=" * 70)

for i, t in enumerate(trials[:10]):
    print(f"\n--- Trial #{t['number']:>3} | Val AUC = {t['auc']:.6f} ---")
    print(f"{'Rank':<5} {'Feature':<8} {'SHAP':>10}")
    print("-" * 30)
    for rank, (name, val) in enumerate(t['shap_top10']):
        print(f"  {rank+1:<3}  {name:<6} {val:>10.4f}")

# Aggregate: which features appear most often in top-10 across best trials
print(f"\n{'=' * 70}")
print("  FEATURE FREQUENCY IN TOP-10 (across best trials)")
print("  = how many times each feature appeared in top 10")
print("=" * 70)

from collections import Counter
freq = Counter()
for t in trials[:10]:
    for name, _ in t['shap_top10']:
        freq[name] += 1

for name, count in freq.most_common(20):
    bar = "#" * count
    print(f"  {name:<8} {count:>2}/10 {bar}")
