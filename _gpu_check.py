"""Quick GPU / memory check before training."""
import psutil

print("=== MEMORY ===")
m = psutil.virtual_memory()
print(f"Total:     {m.total/1024**3:.1f}G")
print(f"Available: {m.available/1024**3:.1f}G")
print(f"Used:      {m.used/1024**3:.1f}G ({m.percent}%)")

print("\n=== TOP MEMORY PROCESSES (>200MB) ===")
procs = []
for p in psutil.process_iter(['name','memory_info']):
    try:
        rss = p.memory_info().rss // 1048576
        if rss > 200:
            procs.append((p.pid, p.name(), rss))
    except:
        pass
procs.sort(key=lambda x: -x[2])
for pid, name, rss in procs:
    print(f"  PID={pid}: {name} RSS={rss}MB")

print("\n=== XGBoost GPU ===")
import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")
try:
    dtrain = xgb.DMatrix([[1.0]], label=[1.0])
    params = {'tree_method': 'gpu_hist', 'gpu_id': 0, 'max_depth': 1}
    xgb.train(params, dtrain, num_boost_round=1)
    print("GPU training: OK  ✅")
    gpu_ok = True
except Exception as e:
    print(f"GPU training: FAILED - {type(e).__name__}: {e}")
    gpu_ok = False

print("\n=== PyTorch CUDA ===")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_mem // 1024**3
        print(f"  VRAM: {vram}G")
    else:
        print("  No CUDA device found")
except Exception as e:
    print(f"PyTorch: {e}")

print("\n=== DECISION ===")
avail = m.available / 1024**3
train_need = 15  # estimated GB needed
if gpu_ok:
    print(f"Recommendation: GPU training (XGBoost GPU works)")
    print(f"  GPU is fastest for tree-based models")
elif avail > train_need:
    print(f"Recommendation: CPU training (memory: {avail:.1f}G available > {train_need}G needed)")
else:
    print(f"WARNING: Memory {avail:.1f}G < {train_need}G needed. Consider downsampling.")
