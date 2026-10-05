"""Check Optuna study state"""
import optuna
s = optuna.load_study(study_name='EURUSD_15m_xgboost', storage='sqlite:///optuna_EURUSD_15m_xgboost.db')
print(f"Trials: {len(s.trials)}")
for t in s.trials:
    print(f"  Trial {t.number}: state={t.state}, value={t.value}")
print(f"\nBest value: {s.best_value}")
print(f"Best params: {s.best_params}")
