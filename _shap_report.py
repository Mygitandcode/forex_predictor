"""SHAP feature importance for final best model"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import joblib
import shap
from features.feature_store import FeatureStore

# Load data
print("Loading data...")
store = FeatureStore()  # default: E:/forex_npy
data = store.load_all("EURUSD")
X = data['features']
y = data.get('15m')
valid = y >= 0
X_valid = X[valid]
y_valid = y[valid]

# 20% holdout (same as training split)
split = int(len(X_valid) * 0.8)
X_test = X_valid[split:]   # the unseen 20%
y_test = y_valid[split:]

print(f"Test set: {X_test.shape}")

# Load model
print("Loading model...")
model = joblib.load("./models/EURUSD_15m.pkl")
print(f"Model: n_estimators={model.n_estimators}")

# Compute SHAP
print("Computing SHAP (PermutationExplainer, 200 samples)...")
import time
t0 = time.time()

explainer = shap.Explainer(
    lambda x: model.predict_proba(x)[:, 1],
    X_test[:min(200, len(X_test))],
)
shap_result = explainer(X_test[:min(200, len(X_test))])
shap_importance = np.abs(shap_result.values).mean(axis=0)

print(f"  Done in {time.time()-t0:.0f}s")

# Rank features
ranked = np.argsort(shap_importance)[::-1]

print("\n" + "=" * 65)
print("  SHAP FEATURE IMPORTANCE — TOP 30")
print("=" * 65)
print(f"{'Rank':<5} {'Feature':<8} {'Importance':>10}")
print("-" * 30)

for rank, idx in enumerate(ranked[:30]):
    print(f"  {rank+1:<3}  f{idx:<6} {shap_importance[idx]:>10.6f}")

# Cumulative
cumsum = np.cumsum(shap_importance[ranked])
total = shap_importance.sum()
print(f"\n{'=' * 65}")
print(f"  CUMULATIVE COVERAGE")
print(f"{'=' * 65}")
for pct in [25, 50, 75, 90, 100]:
    mask = cumsum / total <= pct / 100.0
    n_feats = mask.sum() + 1 if mask.sum() < len(ranked) else len(ranked)
    print(f"  Top {n_feats:>3} features cover {pct:>3}% of total importance")

print(f"\n  Total features: 212")
print(f"  Total SHAP mass: {total:.6f}")
print(f"  Top 1 feature: {shap_importance[ranked[0]]:.6f} = {shap_importance[ranked[0]]/total*100:.1f}%")
print(f"  Top 5 features: {shap_importance[ranked[:5]].sum():.6f} = {shap_importance[ranked[:5]].sum()/total*100:.1f}%")
print(f"  Top 10 features: {shap_importance[ranked[:10]].sum():.6f} = {shap_importance[ranked[:10]].sum()/total*100:.1f}%")
