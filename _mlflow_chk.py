import os, json
from pathlib import Path

base = Path(r'd:\dapanAllThings\codes\forex_predictor\mlruns\551959362222917720')
for run_dir in sorted(base.iterdir()):
    if not run_dir.is_dir() or len(run_dir.name) != 32:
        continue
    meta = run_dir / 'meta.yaml'
    name_tag = run_dir / 'tags' / 'mlflow.runName'
    params_dir = run_dir / 'params'
    
    run_name = 'unknown'
    if name_tag.exists():
        run_name = name_tag.read_text().strip()
    
    params = {}
    if params_dir.exists():
        for pf in params_dir.iterdir():
            if pf.is_file():
                params[pf.name] = pf.read_text().strip()
    
    status = '?'
    if meta.exists():
        for line in meta.read_text().splitlines():
            if 'status:' in line:
                status = line.split(':')[1].strip()
    
    print(f"\nRun: {run_dir.name}")
    print(f"  Name: {run_name}")
    print(f"  Status: {status}")
    print(f"  Params: {params}")
