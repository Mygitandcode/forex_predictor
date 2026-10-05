import subprocess, os, time, json
from pathlib import Path

# Progress check script - shows training status
r = subprocess.run('tasklist /fi "imagename eq python.exe" /nh', capture_output=True, text=True, shell=True)
pcount = len([l for l in r.stdout.splitlines() if l.strip()])

mdir = Path(r"autogluon_models/EURUSD_15m/models")
model_names = sorted([s.name for s in mdir.iterdir()]) if mdir.exists() else []

all_mdir = Path(r"autogluon_models/EURUSD_15m")
total_mb = 0
recent = []
now = time.time()
if all_mdir.exists():
    for f in all_mdir.rglob("*"):
        if f.is_file():
            size_kb = f.stat().st_size // 1024
            total_mb += size_kb / 1024
            if now - f.stat().st_mtime < 120:
                recent.append(f"{f.name}({size_kb}KB)")

if model_names:
    status = f"DONE: {model_names}"
elif recent:
    status = f"ACTIVE (recent files: {recent})"
elif pcount > 3:
    status = f"FITTING ({pcount} procs)"
else:
    status = "STOPPED"

result = {
    "time": time.strftime("%H:%M:%S"),
    "status": status,
    "procs": pcount,
    "models": model_names,
    "total_mb": round(total_mb, 1),
}
print(json.dumps(result, indent=2))
