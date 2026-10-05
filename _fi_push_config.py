"""_fi_push_config.py - 特征迭代 推送训练 config 到 Redis

无状态 AI 在 START 时调用, 从 features-base.json 取特征, 构建 config, 推到 Redis。
trailName 命名: {item}_E{N}, N 自动递增(从 done.jsonl 找最大 E{N} +1)。
输出 JSON Lines 到 stdout。

用法:
    python _fi_push_config.py EURUSD_15m
    python _fi_push_config.py EURUSD_15m --n-trials 40 --trial-seq E1
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.jsonlog import jlog
from pipeline.mq import RedisStreamClient

# 特征注册表
REG_PATH = "D:/dapanAllThings/dapanPlan/outputs/feature-iteration/registry/feature-registry.json"
BASE_PATH = "D:/dapanAllThings/dapanPlan/outputs/feature-iteration/registry/features-base.json"
DONE_PATH = "D:/dapanAllThings/dapanPlan/outputs/feature-iteration/done.jsonl"


def _next_seq():
    """从 done.jsonl 找最大 E{N} 序号, 返回下一个。空则 E1。"""
    max_n = 0
    if os.path.exists(DONE_PATH):
        for line in open(DONE_PATH, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            # trail 字段或 item 字段里找 E{N}
            for key in ("trail", "trailName", "item"):
                val = d.get(key, "")
                m = re.search(r"E(\d+)$", str(val))
                if m:
                    max_n = max(max_n, int(m.group(1)))
    return f"E{max_n + 1}"


def main():
    parser = argparse.ArgumentParser(description="推送训练 config 到 Redis")
    parser.add_argument("item", help="item 名, 如 EURUSD_15m")
    parser.add_argument("--n-trials", type=int, default=40, help="Optuna 试验次数")
    parser.add_argument("--trial-seq", default=None,
                        help="命名序号, 如 E1(不传则自动从 done.jsonl 递增)")
    parser.add_argument("--features-ref", default="第一版构建特征",
                        help="特征注册表 key(默认'第一版构建特征', 60个)")
    args = parser.parse_args()

    # 解析 item -> symbol + target_tf
    parts = args.item.split("_", 1)
    if len(parts) != 2:
        jlog("error", msg=f"item 格式错误: {args.item}, 应为 SYMBOL_tf 如 EURUSD_15m")
        sys.exit(1)
    symbol, target_tf = parts[0], parts[1]

    # 算 trailName 序号
    seq = args.trial_seq or _next_seq()
    trail_name = f"{args.item}_{seq}"

    # 取特征: 读 registry 的 key, 有 selected 用 selected(子集), 否则 base 全量
    reg = json.load(open(REG_PATH, encoding="utf-8"))
    if args.features_ref in reg and reg[args.features_ref].get("selected"):
        features = reg[args.features_ref]["selected"]
    else:
        base = json.load(open(BASE_PATH, encoding="utf-8"))
        features = base["features"]

    config = {
        "trailName": trail_name,
        "symbol": symbol,
        "target_tf": target_tf,
        "features": features,
        "n_trials": args.n_trials,
        "holdout": False,
        "mode": "train",
        "created_by": "feature-iteration autonomous",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    # 推到 Redis
    n = RedisStreamClient().push_request(config)

    jlog("push_config", trail=trail_name, item=args.item, symbol=symbol,
         target_tf=target_tf, features=len(features), n_trials=args.n_trials,
         seq=seq, features_ref=args.features_ref, queue_len=n)


if __name__ == "__main__":
    main()
