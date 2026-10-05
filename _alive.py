import time, psutil, os
print("Sampling 3s + 3s...")
time.sleep(3)
psutil.cpu_percent(interval=3)  # warm up
for p in psutil.process_iter(['pid','name','cpu_percent','memory_info']):
    n = (p.info['name'] or '').lower()
    if 'python' in n:
        cpu = p.info['cpu_percent'] or 0
        mem = p.info['memory_info'].rss // 1048576
        if cpu > 0 or mem > 100:
            print(f"PID={p.pid} cpu={cpu:.1f}% mem={mem}MB")

mdir = r"d:\dapanAllThings\codes\forex_predictor\autogluon_models\EURUSD_15m"
total = 0
for root, dirs, files in os.walk(mdir):
    for f in files:
        total += os.path.getsize(os.path.join(root, f))
print(f"Dir size: {total/1024**3:.2f}GB")
print(f"Models exist: {os.path.exists(os.path.join(mdir, 'models'))}")
