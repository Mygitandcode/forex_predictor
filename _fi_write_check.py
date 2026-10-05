"""_fi_write_check.py - 写 check.jsonl 活动格(第二态)

无状态 AI 在 START 时调用, 写入活动格。字段与 done.jsonl(第三态)近似(都有 python_log + ai_log),
下一个 AI 读 check.jsonl 即知日志位置。

用法: python _fi_write_check.py EURUSD_15m --trail EURUSD_15m_E1
      (trail 从 _fi_push_config.py 的输出读)
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.jsonlog import jlog

FE = "D:/dapanAllThings/dapanPlan/outputs/feature-iteration"


def main():
    parser = argparse.ArgumentParser(description="写 check.jsonl 活动格(第二态)")
    parser.add_argument("item", help="item 名, 如 EURUSD_15m")
    parser.add_argument("--trail", required=True, help="trailName, 如 EURUSD_15m_E1(从 push_config 输出读)")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--features-count", type=int, default=60, help="特征数(默认60, 第一版构建)")
    args = parser.parse_args()

    # 解析 item -> symbol + target_tf
    parts = args.item.split("_", 1)
    if len(parts) != 2:
        jlog("error", msg=f"item 格式错误: {args.item}")
        sys.exit(1)
    symbol, target_tf = parts[0], parts[1]

    # 活动格(字段与 done.jsonl 近似: 都有 python_log + ai_log)
    entry = {
        "item": args.item,
        "symbol": symbol,
        "target_tf": target_tf,
        "trailName": args.trail,
        "features_ref": "当前可用特征",
        "features_count": args.features_count,
        "n_trials": args.n_trials,
        "holdout": False,
        "mode": "train",
        "status": "running",
        "current_stage": 0,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python_log": f"E:/forex_logs/{args.trail}.log",  # 和 done.jsonl 字段一致
        "ai_log": f"logs/{args.item}/ai.log",
    }

    path = os.path.join(FE, "check.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 建 ai.log 目录
    os.makedirs(os.path.join(FE, "logs", args.item), exist_ok=True)

    jlog("write_check", item=args.item, trail=args.trail,
         python_log=entry["python_log"], ai_log=entry["ai_log"])


if __name__ == "__main__":
    main()
