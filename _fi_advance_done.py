"""_fi_advance_done.py - 完工后移 check(第二态) -> done(第三态)

无状态 AI 在 ADVANCE 时(_fi_check_complete.py all_ok 后)调用。
读 check.jsonl 活动格 + 查 auc/trials + 追加 done.jsonl + 清 check.jsonl。
strategy.md(研判)由 AI 单独写, 本脚本只做状态迁移。

用法: python _fi_advance_done.py {trailName}
"""

import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.jsonlog import jlog

FE = "D:/dapanAllThings/dapanPlan/outputs/feature-iteration"
CODE_ROOT = "D:/dapanAllThings/codes/forex_predictor"


def main():
    trail = sys.argv[1] if len(sys.argv) > 1 else ""
    if not trail:
        jlog("error", msg="用法: python _fi_advance_done.py {trailName}")
        sys.exit(1)

    # 1. 读 check.jsonl 活动格
    check_path = os.path.join(FE, "check.jsonl")
    lines = [l for l in open(check_path, encoding="utf-8") if l.strip()]
    if not lines:
        jlog("error", msg="check.jsonl 空, 无活动格可移")
        sys.exit(1)
    check = json.loads(lines[0])

    # 2. 查 auc/trials(从 optuna db)
    auc = None
    trials = 0
    try:
        import optuna
        dbs = glob.glob(os.path.join(CODE_ROOT, f"optuna_*{trail}*.db"))
        if dbs:
            db = dbs[0]
            study_name = os.path.basename(db).replace("optuna_", "").replace(".db", "")
            st = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db}")
            trials = len([t for t in st.trials
                          if t.state == optuna.trial.TrialState.COMPLETE])
            auc = round(float(st.best_value), 4) if trials > 0 else None
    except Exception as e:
        jlog("warning", msg=f"查 auc/trials 失败: {e}")

    # 3. 追加 done.jsonl(字段与 check 近似 + auc/trials/strategy/done_at)
    done_entry = {
        "item": check["item"],
        "trail": trail,
        "features_ref": check.get("features_ref", "当前可用特征"),
        "auc": auc,
        "trials": trials,
        "python_log": check.get("python_log", f"E:/forex_logs/{trail}.log"),
        "ai_log": check.get("ai_log", f"logs/{check['item']}/ai.log"),
        "strategy": f"logs/{check['item']}/strategy.md",
        "done_at": datetime.now().isoformat(timespec="seconds"),
    }
    done_path = os.path.join(FE, "done.jsonl")
    with open(done_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(done_entry, ensure_ascii=False) + "\n")

    # 4. 清 check.jsonl(活动格移走)
    open(check_path, "w", encoding="utf-8").close()

    jlog("advance_done", item=check["item"], trail=trail, auc=auc,
         trials=trials, done_path=done_path)


if __name__ == "__main__":
    main()
