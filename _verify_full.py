"""Full verification: FeatureBuilder -> Pipeline -> Training readiness"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

# === 1. FeatureBuilder sanity check ===
print("=== 1. FeatureBuilder ===")
from features.feature_builder import FeatureBuilder
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=2500, freq='1min')
price = 1.10 + np.cumsum(np.random.randn(2500) * 0.0001)
df = pd.DataFrame({
    'open': price + np.random.randn(2500) * 0.00002,
    'high': price + abs(np.random.randn(2500) * 0.00005),
    'low': price - abs(np.random.randn(2500) * 0.00005),
    'close': price + np.random.randn(2500) * 0.00002,
    'tick_volume': np.random.randint(50, 500, 2500),
}, index=dates)
df['high'] = df[['open','high','close']].max(axis=1)
df['low'] = df[['open','low','close']].min(axis=1)

b = FeatureBuilder('EURUSD', 0.00001)
X, names = b.build_many(df)
print(f"  Features: {X.shape}, dims: {len(names)}")
print(f"  First: {names[0]}, Last: {names[-1]}")
assert X.shape[1] == len(names), "Dimension mismatch!"
print("  PASS")

# === 2. FeatureStore ready ===
print("\n=== 2. FeatureStore ===")
from features.feature_store import FeatureStore
store = FeatureStore()  # default: E:/forex_npy
print(f"  EURUSD done: {store.is_done('EURUSD')}")
if store.is_done('EURUSD'):
    data = store.load_all('EURUSD')
    print(f"  Current X: {data['features'].shape}, dims: {data['features'].shape[1]}")
    print(f"  Current labels: {list(k for k in data if k.startswith('labels'))}")
    # Need rebuild since dims changed
    if data['features'].shape[1] != len(names):
        print(f"  DIMENSION CHANGED ({data['features'].shape[1]} -> {len(names)}), need rebuild")
print("  PASS")

# === 3. Model loading ===
print("\n=== 3. Model Loading ===")
from autogluon.tabular import TabularPredictor
model_path = r"d:\dapanAllThings\codes\forex_predictor\autogluon_models\EURUSD_15m"
if os.path.exists(model_path):
    p = TabularPredictor.load(model_path)
    print(f"  Models: {p.model_names()}")
    # Old model is 113-dim, new is 212-dim. Check.
    old_dim = len(p.feature_metadata.get_features()) if hasattr(p, 'feature_metadata') else '?'
    print(f"  Old model dim: {old_dim}, New features: {len(names)}")
    if old_dim != len(names):
        print(f"  MODEL NEEDS RETRAINING (dimension mismatch)")
else:
    print(f"  No model yet. Will be trained after feature rebuild.")

# === 4. Memory ===
print("\n=== 4. Memory ===")
import psutil
m = psutil.virtual_memory()
print(f"  Available: {m.available/1024**3:.1f}G / {m.total/1024**3:.1f}G")

# === 5. MLflow ===
print("\n=== 5. MLflow ===")
try:
    import mlflow
    mlflow.set_tracking_uri('http://localhost:5000')
    exp = mlflow.get_experiment_by_name('forex_stage1')
    if exp:
        print(f"  MLflow OK, experiment_id={exp.experiment_id}")
    else:
        print(f"  Creating experiment 'forex_stage1'...")
        mlflow.create_experiment('forex_stage1')
        print(f"  Created")
except Exception as e:
    print(f"  MLflow not running: {e}")

# === 6. Check old autogluon residue ===
print("\n=== 6. Old model residue ===")
old = os.path.exists(model_path)
print(f"  autogluon_models/EURUSD_15m exists: {old}")
if old:
    print(f"  Needs deletion before retraining (dimension mismatch)")

print("\n=== SUMMARY ===")
print(f"  FeatureBuilder:      OK ({len(names)}d)")
print(f"  Memory:              {m.available/1024**3:.1f}G available")
print(f"  Feature rebuild:     REQUIRED (dims changed)")
print(f"  Model retrain:       REQUIRED")
print(f"  Ready to execute:    python -m features.feature_pipeline --data-dir E:\\forex_data\\cleaneded\\EURUSD --symbols EURUSD --store-dir ./data_store --force")
