"""Test SHAP compatibility with XGBoost 3.1.3"""
import numpy as np
from xgboost import XGBClassifier
import shap

# Small dataset
X = np.random.randn(200, 20)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = XGBClassifier(n_estimators=10, max_depth=3, verbosity=0, n_jobs=1, random_state=42)
model.fit(X, y)

X_test = np.random.randn(50, 20)

# Test 1: TreeExplainer(model) - current approach
print("Test 1: shap.TreeExplainer(model)...")
try:
    e1 = shap.TreeExplainer(model)
    sv1 = e1.shap_values(X_test[:10])
    print(f"  OK: shape={np.array(sv1).shape}")
except Exception as ex:
    print(f"  FAIL: {ex}")

# Test 2: TreeExplainer(model.get_booster())  
print("Test 2: shap.TreeExplainer(model.get_booster())...")
try:
    e2 = shap.TreeExplainer(model.get_booster())
    sv2 = e2.shap_values(X_test[:10])
    print(f"  OK: shape={np.array(sv2).shape}")
except Exception as ex:
    print(f"  FAIL: {ex}")

# Test 3: shap.Explainer(model)
print("Test 3: shap.Explainer(model)...")
try:
    e3 = shap.Explainer(model)
    sv3 = e3(X_test[:10])
    print(f"  OK: shape={sv3.values.shape}")
except Exception as ex:
    print(f"  FAIL: {ex}")

# Test 4: shap.Explainer(model.predict, X_test[:50])
print("Test 4: shap.Explainer(model.predict, X_test[:50])...")
try:
    e4 = shap.Explainer(model.predict, X_test[:50])
    sv4 = e4(X_test[:10])
    print(f"  OK: shape={sv4.values.shape}")
except Exception as ex:
    print(f"  FAIL: {ex}")

# Test 5: model.predict_proba as Explainer function
print("Test 5: shap.Explainer(model.predict_proba, X_test[:50])...")
try:
    e5 = shap.Explainer(lambda x: model.predict_proba(x)[:, 1], X_test[:50])
    sv5 = e5(X_test[:10])
    print(f"  OK: shape={sv5.values.shape}")
except Exception as ex:
    print(f"  FAIL: {ex}")

# Test 6: Use model's internal feature importance as fallback
print("Test 6: XGBoost built-in feature_importances_...")
try:
    fi = model.feature_importances_
    print(f"  OK: shape={fi.shape}, sum={fi.sum():.4f}")
except Exception as ex:
    print(f"  FAIL: {ex}")
