"""_fi_check_complete.py - 特征迭代 完工检查

无状态 AI 在 ADVANCE 时调用, 检查训练是否完工。
完工门槛: n_trials>=40 + AUC>0.50 + 产物齐全(模型/预测/summary/feat_imp/.done)。
输出 JSON Lines, 退出码 0=完工 1=未完工。

用法: python _fi_check_complete.py EURUSD_15m_E1
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.jsonlog import jlog

N_TRIALS_REQUIRED = 40
AUC_MIN = 0.50
CODE_ROOT = "D:/dapanAllThings/codes/forex_predictor"


def main():
    trail = sys.argv[1] if len(sys.argv) > 1 else ""
    if not trail:
        jlog("error", msg="用法: python _fi_check_complete.py {trailName}")
        sys.exit(1)

    result = {"trail": trail}
    checks = {}

    # 1. Optuna trial 数 + AUC
    try:
        import optuna
        dbs = glob.glob(os.path.join(CODE_ROOT, f"optuna_*{trail}*.db"))
        if dbs:
            db = dbs[0]
            study_name = os.path.basename(db).replace("optuna_", "").replace(".db", "")
            st = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db}")
            complete = len([t for t in st.trials
                            if t.state == optuna.trial.TrialState.COMPLETE])
            checks["trials_complete"] = complete
            checks["trials_ok"] = complete >= N_TRIALS_REQUIRED
            if complete > 0:
                auc = float(st.best_value)
                checks["best_auc"] = round(auc, 4)
                checks["auc_ok"] = auc > AUC_MIN
            else:
                checks["auc_ok"] = False
        else:
            checks["trials_ok"] = False
            checks["auc_ok"] = False
            checks["error"] = "optuna db 未找到"
    except Exception as e:
        checks["error"] = str(e)
        checks["trials_ok"] = False
        checks["auc_ok"] = False

    # 2. 产物文件
    # 模型(精确 + 模糊)
    model = os.path.join(CODE_ROOT, "models", f"{trail}.pkl")
    if not os.path.exists(model):
        matches = glob.glob(os.path.join(CODE_ROOT, "models", f"*{trail}*.pkl"))
        if matches:
            model = matches[0]
    checks["model_exists"] = os.path.exists(model)

    # outputs/YYYY_MM/{trail}/ 下的产物
    output_dirs = glob.glob(os.path.join(CODE_ROOT, "outputs", "*", trail))
    if output_dirs:
        odir = output_dirs[0]
        checks["predictions_exists"] = os.path.exists(
            os.path.join(odir, "03_live_predictions.xlsx"))
        checks["summary_exists"] = os.path.exists(
            os.path.join(odir, "06_summary.md"))
        checks["feat_imp_exists"] = os.path.exists(
            os.path.join(odir, "02_feature_importance.md"))
    else:
        checks["predictions_exists"] = False
        checks["summary_exists"] = False
        checks["feat_imp_exists"] = False

    # npy .done
    npy_done = glob.glob(f"E:/forex_npy/{trail}/*/.done")
    checks["npy_done_exists"] = len(npy_done) > 0

    # 总判定
    result["checks"] = checks
    result["all_ok"] = all([
        checks.get("trials_ok", False),
        checks.get("auc_ok", False),
        checks.get("model_exists", False),
        checks.get("predictions_exists", False),
        checks.get("summary_exists", False),
        checks.get("feat_imp_exists", False),
        checks.get("npy_done_exists", False),
    ])

    jlog("check_complete", **result)
    sys.exit(0 if result["all_ok"] else 1)


if __name__ == "__main__":
    main()
