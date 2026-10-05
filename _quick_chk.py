import psutil, os, sys
sys.path.insert(0, os.path.dirname(__file__))

# Memory
m = psutil.virtual_memory()
print(f"Memory: {m.available/1024**3:.1f}G / {m.total/1024**3:.1f}G")

# Existing features
from features.feature_store import FeatureStore
s = FeatureStore()  # default: E:/forex_npy
print(f"EURUSD done: {s.is_done('EURUSD')}")
if s.is_done('EURUSD'):
    d = s.load_all('EURUSD')
    print(f"Current dims: {d['features'].shape[1]}")

# Model
model_path = r"autogluon_models/EURUSD_15m"
if os.path.exists(model_path):
    from autogluon.tabular import TabularPredictor
    p = TabularPredictor.load(model_path)
    old_d = len(p.feature_metadata.get_features())
    print(f"Old model dims: {old_d}")
    print("-> NEED RETRAIN (dims changed)")

# MLflow
try:
    import mlflow
    mlflow.set_tracking_uri('http://localhost:5000')
    exp = mlflow.get_experiment_by_name('forex_stage1')
    print(f"MLflow: OK (exp={exp.experiment_id})")
except Exception as e:
    print(f"MLflow: NOT RUNNING ({e})")
