"""
pipeline/nlu.py — 自然语言 → JSON 配置

将用户的自然语言训练描述转为标准 JSON Schema。
调用 feature_selector 提取有效特征。

使用:
    from pipeline.nlu import parse
    config = parse("Top-50特征 训练100轮 代号测试版")
    # → {"trailName": "测试版", "features": [...], "n_trials": 100, ...}
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .feature_selector import get_effective_features, get_top_features

logger = logging.getLogger(__name__)


# ================================================================
# 解析规则
# ================================================================

# 特征数量模式
FEATURE_PATTERNS = [
    (r"(?:top[-\s]?)?(\d+)\s*(?:个|维)?\s*(?:特征|feature)", "top_n"),
    (r"(?:全部|所有|all|317)\s*(?:个|)?\s*(?:有效)?\s*(?:特征|feature)", "all"),
    (r"(?:前|top)\s*(\d+)", "top_n"),
]

# trial 数量模式
TRIAL_PATTERNS = [
    (r"(\d+)\s*(?:轮|trial|试验|次)", "n_trials"),
]

# 代号模式
TRAIL_PATTERNS = [
    (r"(?:代号|名称|叫)\s*[：:\s]*(\S+)", "trailName"),
]

# holdout 模式
HOLDOUT_PATTERNS = [
    (r"(?:投产|自用|最终|vir|生产|上线)", "holdout_true"),
    (r"(?:训练|实验|测试|开发)", "holdout_false"),
]

# 周期模式
TF_PATTERNS = [
    (r"(?:7m|7分钟|7分)", "7m"),
    (r"(?:10m|10分钟|10分)", "10m"),
    (r"(?:15m|15分钟|15分)", "15m"),
    (r"(?:30m|30分钟|30分)", "30m"),
    (r"(?:1h|1小时|60分钟|60分)", "1h"),
]

# 货币对模式
SYMBOL_PATTERNS = [
    (r"\b(EURUSD|GBPUSD|USDJPY|AUDUSD|NZDUSD|USDCAD|USDCHF|EURGBP|EURJPY|GBPJPY|AUDJPY|NZDJPY)\b", "symbol"),
]


def parse(text: str) -> dict:
    """将自然语言描述转为 JSON 配置。

    Args:
        text: 用户自然语言输入，如:
            "Top-50特征 训练100轮 代号202605首版"
            "317有效特征 代号最终投产版 投产"

    Returns:
        标准化训练配置 dict (schema.REQUEST_SCHEMA 格式)
    """
    config = {
        "trailName": "",
        "features": [],
        "n_trials": 100,
        "holdout": False,
        "mode": "train",
        "symbol": "EURUSD",
        "target_tf": "15m",
        "created_by": text,
        "created_at": datetime.now().isoformat(),
    }

    # ---- 解析 trailName ----
    trail_match = _match_first(text, TRAIL_PATTERNS)
    if trail_match:
        config["trailName"] = trail_match
    else:
        # 默认: 当前日期
        config["trailName"] = datetime.now().strftime("%Y%m%d_%H%M")

    # ---- 解析特征 ----
    feature_config = _parse_features(text)
    config["features"] = feature_config["features"]

    # ---- 解析 trial 数 ----
    trial_match = _match_first(text, TRIAL_PATTERNS)
    if trial_match:
        config["n_trials"] = int(trial_match)

    # ---- 解析 holdout ----
    holdout_match = _match_first(text, HOLDOUT_PATTERNS)
    if holdout_match:
        config["holdout"] = holdout_match == "holdout_true"

    # ---- 解析周期 ----
    tf_match = re.search("|".join(p[0] for p in TF_PATTERNS), text, re.IGNORECASE)
    if tf_match:
        config["target_tf"] = tf_match.group(0)

    # ---- 解析货币对 ----
    symbol_match = _match_first(text, SYMBOL_PATTERNS)
    if symbol_match:
        config["symbol"] = symbol_match.upper()

    return config


def _parse_features(text: str) -> dict:
    """从文本解析特征选择"""
    features = []

    # 尝试匹配 Top-N 或全部
    for pattern, flag in FEATURE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            if flag == "top_n":
                n = int(m.group(1))
                features = get_top_features(n)
                logger.info(f"选中 Top-{n} 特征: {len(features)} 个")
            elif flag == "all":
                features = get_effective_features()
                logger.info(f"选中全部有效特征: {len(features)} 个")
            break

    if not features:
        # 默认: Top-50
        features = get_top_features(50)
        logger.info(f"未指定特征, 默认 Top-50: {len(features)} 个")

    return {"features": features}


def _match_first(text: str, patterns: List[Tuple[str, str]]) -> Optional[str]:
    """匹配第一个命中模式的值"""
    for pattern, flag in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            if flag in ("holdout_true", "holdout_false"):
                return flag
            return m.group(1)
    return None


def to_json_str(config: dict) -> str:
    """配置 → JSON 字符串"""
    return json.dumps(config, ensure_ascii=False, indent=2)


# ================================================================
# 自测
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tests = [
        "Top-50特征 训练100轮 代号202605首版",
        "317有效特征 代号最终投产版 投产",
        "前20特征 50轮 EURJPY 1h",
        "默认特征 代号自动测试",
    ]

    for t in tests:
        config = parse(t)
        print(f"\n输入: {t}")
        print(f"输出: trailName={config['trailName']}, features={len(config['features'])}, "
              f"trials={config['n_trials']}, holdout={config['holdout']}, "
              f"symbol={config['symbol']}, tf={config['target_tf']}")
