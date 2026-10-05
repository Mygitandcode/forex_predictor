"""Extract XGBoost params from trained AutoGluon model"""
import json
from autogluon.tabular import TabularPredictor

p = TabularPredictor.load(r'd:\dapanAllThings\codes\forex_predictor\autogluon_models\EURUSD_15m')
lb = p._learner
t = lb.load_trainer()
m = t.load_model('XGBoost')

print(f"Model class: {type(m).__name__}")

# Try to get XGBoost params
if hasattr(m, 'params'):
    print(f"\nXGBoost hyperparams:")
    for k, v in m.params.items():
        print(f"  {k}: {v}")
else:
    print("No .params attribute")

# Try inner model
if hasattr(m, 'model'):
    inner = m.model
    print(f"\nInner model: {type(inner).__name__}")
    try:
        config = json.loads(inner.save_config())
        learner = config.get('learner', {})
        gb = learner.get('gradient_booster', {})
        print(f"\nLearner config keys: {list(learner.keys())}")
        train_param = learner.get('learner_train_param', {})
        print(f"\nTrain params:")
        for k, v in train_param.items():
            print(f"  {k}: {v}")
        obj_param = learner.get('objective', {})
        print(f"\nObjective: {obj_param}")
        metric = learner.get('metric', {})
        print(f"Metric: {metric}")
        print(f"\nBooster type: {gb.get('name', 'unknown')}")
        up = gb.get('updater', {})
        print(f"Updater params: {up}")
    except Exception as e:
        print(f"Config error: {e}")

# Feature preprocessing info
print(f"\n=== Preprocessing ===")
print(f"Features in: {p.feature_metadata.get_features() if hasattr(p, 'feature_metadata') else 'unknown'}")
print(f"Label: {p.label}")
print(f"Problem type: {p.problem_type}")
