"""Direct XGBoost training — bypass AutoGluon deadlock"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.metrics import roc_auc_score

# Load features and labels
store = np.load
d = "E:/forex_npy/EURUSD"
X = store(f"{d}/features.npy")
y = store(f"{d}/labels_15m.npy")
ts = store(f"{d}/timestamps.npy")

# Filter valid samples
valid = y >= 0
X, y = X[valid], y[valid]
print(f"Data: X={X.shape}, y={len(y)}, pos={(y==1).sum()/len(y)*100:.1f}%")

# Time-based split (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("forex_stage1")

# XGBoost params
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 2000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': 4,  # 4 threads works on Windows
    'random_state': 42,
    'verbosity': 1,
}

with mlflow.start_run(run_name="XGBoost_direct_212d"):
    mlflow.log_params(params)
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_train", len(X_train))
    mlflow.log_param("n_test", len(X_test))

    print("Training...")
    t0 = time.time()
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s")

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    acc = model.score(X_test, y_test)
    print(f"Test AUC: {auc:.4f}, Accuracy: {acc:.4f}")

    mlflow.log_metric("test_auc", auc)
    mlflow.log_metric("test_accuracy", acc)

    # Save
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_EURUSD_15m.pkl")
    mlflow.log_artifact("models/xgb_EURUSD_15m.pkl")
    print("Model saved to models/xgb_EURUSD_15m.pkl")
