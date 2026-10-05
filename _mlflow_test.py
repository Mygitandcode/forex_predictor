import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('forex_stage1')
with mlflow.start_run(run_name='test_emoji_fix') as run:
    mlflow.log_param('test', 1)
    print(f'OK - run_id: {run.info.run_id}')
    print(f'Status: {run.info.status}')
print('end_run OK - no crash')
