import os, psutil, time
m = psutil.virtual_memory()
print(f"Mem: {m.available/1024**3:.1f}G free")

# Check if training model dir exists and has content
path = r"d:\dapanAllThings\codes\forex_predictor\autogluon_models\EURUSD_15m"
if os.path.exists(path):
    files = list(os.walk(path))[:20]
    print(f"Model dir exists: {len(files)} entries")
else:
    print("Model dir: NOT FOUND")

# Check for mlflow log
mlog = r"d:\dapanAllThings\codes\forex_predictor\mlruns"
runs = list(os.walk(mlog)) if os.path.exists(mlog) else []
print(f"MLflow runs dirs: {sum(1 for d in runs if d[0].count(os.sep)==len(mlog.split(os.sep))+2)}")
