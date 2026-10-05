"""_fi_monitor.py - 特征迭代 训练状态检查

无状态 AI 在 MONITOR 时调用, 一次性查: 进程/日志末尾/E盘/Optuna trial数/Redis状态。
输出 JSON Lines。

用法: python _fi_monitor.py EURUSD_15m_E1
"""

import glob
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.jsonlog import jlog

LOG_ROOT = "E:/forex_logs"
CODE_ROOT = "D:/dapanAllThings/codes/forex_predictor"


def main():
    trail = sys.argv[1] if len(sys.argv) > 1 else ""
    if not trail:
        jlog("error", msg="用法: python _fi_monitor.py {trailName}")
        sys.exit(1)

    result = {"trail": trail}

    # 1. 进程存活
    try:
        r = subprocess.run(
            'wmic process where "name=\'python.exe\'" get commandline /format:csv',
            capture_output=True, shell=True, timeout=10)
        text = r.stdout.decode("gbk", errors="replace")
        result["orchestrator_alive"] = "orchestrator" in text
        result["train_process_alive"] = trail in text or "stage2_optuna" in text
    except Exception as e:
        result["process_check_error"] = str(e)

    # 2. python.log 末尾
    log_path = os.path.join(LOG_ROOT, f"{trail}.log")
    result["log_path"] = log_path
    if os.path.exists(log_path):
        lines = open(log_path, encoding="utf-8", errors="replace").readlines()
        result["log_lines"] = len(lines)
        tail = lines[-5:] if len(lines) >= 5 else lines
        result["log_tail"] = [l.strip() for l in tail]
        result["log_last"] = (tail[-1].strip()[:200] if tail else "")
    else:
        result["log_exists"] = False

    # 3. E盘
    try:
        u = shutil.disk_usage("E:/")
        result["e_free_gb"] = round(u.free / 1e9, 1)
    except Exception:
        pass

    # 4. Optuna trial 数
    try:
        import optuna
        dbs = glob.glob(os.path.join(CODE_ROOT, f"optuna_*{trail}*.db"))
        if dbs:
            db = dbs[0]
            study_name = os.path.basename(db).replace("optuna_", "").replace(".db", "")
            st = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db}")
            complete = len([t for t in st.trials
                            if t.state == optuna.trial.TrialState.COMPLETE])
            result["optuna_db"] = os.path.basename(db)
            result["trials_complete"] = complete
            result["best_auc"] = round(float(st.best_value), 4) if complete > 0 else None
        else:
            result["optuna_db"] = None
            result["trials_complete"] = 0
    except Exception as e:
        result["optuna_error"] = str(e)
        result["trials_complete"] = 0

    # 5. Redis 状态
    try:
        from pipeline.mq import get_redis
        r = get_redis()
        state = r.hgetall(f"pipeline:run:{trail}")
        result["redis_status"] = state.get("status", "?")
        result["redis_stages"] = {f"stage-{i}": state.get(f"stage-{i}", "?")
                                  for i in range(1, 7)}
    except Exception as e:
        result["redis_error"] = str(e)

    jlog("monitor", **result)


if __name__ == "__main__":
    main()
