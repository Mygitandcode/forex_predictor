"""Clean up stale MLflow runs from emoji crash sessions.
- 475a2c9f (sample_ratio=0.5) = successful training → mark FINISHED  
- rest (OOM / no models) → mark FAILED
"""
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('http://localhost:5000')
client = MlflowClient()

# Locate the forex_stage1 experiment
exp = client.get_experiment_by_name('forex_stage1')
if not exp:
    print("Experiment not found")
    exit()

runs = client.search_runs([exp.experiment_id])
print(f"Found {len(runs)} runs in experiment '{exp.name}'")

successful_run = '475a2c9fa1de418d95a2b44a2daeb2bb'  # sample_ratio=0.5, training succeeded

for r in runs:
    rid = r.info.run_id
    params = r.data.params
    status = r.info.status
    
    has_sample = 'sample_ratio' in params
    
    if rid == successful_run or has_sample:
        print(f"  {rid[:8]}... [FINISHED] (sample={params.get('sample_ratio','?')}, train={params.get('n_train','?')})")
        client.set_terminated(rid, 'FINISHED')
    elif status == 'RUNNING':
        print(f"  {rid[:8]}... [FAILED]   (train={params.get('n_train','?')})")
        client.set_terminated(rid, 'FAILED')
    else:
        print(f"  {rid[:8]}... [{status}]  (train={params.get('n_train','?')})")

print("\nDone - all runs cleaned.")
