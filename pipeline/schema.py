"""
pipeline/schema.py — JSON Schema 定义 + 验证 + holdout 逻辑

MQ 驱动全链路训练系统的契约层。所有消费者以此模块为唯一校验入口。
设计蓝图: dapanPlan/guides/mq-pipeline-blueprint.md
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 项目根 — 用于构建绝对路径输出目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


# ================================================================
# 常量
# ================================================================

# 支持的货币对
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD",
    "USDCHF", "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
]

# 支持的目标周期
TARGET_TFS = ["7m", "10m", "15m", "30m", "1h"]

# 默认值
DEFAULT_SYMBOL = "EURUSD"
DEFAULT_TARGET_TF = "15m"
DEFAULT_N_TRIALS = 100
DEFAULT_HOLDOUT = False
DEFAULT_MODE = "train"

# Redis Stream 命名
STREAM_REQUESTS = "training:requests"
STREAM_STATUS = "training:status"
HASH_PREFIX = "pipeline:run"

# Stage 流命名
STAGE_STREAMS = {
    1: "training:stage-1-sync",
    2: "training:stage-2-features",
    3: "training:stage-3-export",
    4: "training:stage-4-model",
    5: "training:stage-5-validate",
    6: "training:stage-6-report",
}

# Stage 名称
STAGE_NAMES = {
    1: "拉齐原始数据",
    2: "特征构建",
    3: "特征导出",
    4: "Optuna训练",
    5: "验证预测",
    6: "报告聚合",
}


# ================================================================
# 请求 JSON Schema
# ================================================================

REQUEST_SCHEMA = {
    "type": "object",
    "required": ["trailName", "features", "n_trials", "holdout", "mode"],
    "properties": {
        "trailName": {
            "type": "string",
            "minLength": 1,
            "maxLength": 128,
            "description": "贯穿全链路的 run 标识，决定输出目录名和模型文件名",
        },
        "symbol": {
            "type": "string",
            "enum": SYMBOLS + [""],
            "default": DEFAULT_SYMBOL,
            "description": "货币对",
        },
        "target_tf": {
            "type": "string",
            "enum": TARGET_TFS + [""],
            "default": DEFAULT_TARGET_TF,
            "description": "目标周期",
        },
        "features": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "训练使用的特征名列表。来源: docs/feature_importance_report.md",
        },
        "n_trials": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10000,
            "default": DEFAULT_N_TRIALS,
            "description": "Optuna 试验次数",
        },
        "holdout": {
            "type": "boolean",
            "default": DEFAULT_HOLDOUT,
            "description": (
                "双层切分真相(2026-09-09 双模型检核定稿): "
                "消费链分两层——① 本字段经 stage3_export 由 schema.needs_holdout(L229-231 "
                "`return data.get('holdout', DEFAULT_HOLDOUT) is False`,缺省即 false)消费: "
                "holdout=false → 导出时按 calc_holdout_cutoff(L208-226, max_ts 所在月−3 的月初)"
                "切后 4 个自然月为 holdout 验证段,训练段≈2015→holdout 起点前一日; "
                "holdout=true → train=val=全量(投产/最终模型)。"
                "② training/stage2_optuna 对训练段内部恒定 80/20 时间序切分(L481),"
                "无滑动窗口/CV。E2 取 false=实验模式(末 4 月为真 holdout)。"
                "既有文案错位不扩散警示: stage3_export.py L179 日志文案把 true 分支写为"
                "'holdout=false (投产)'——代码侧既有、非本注释放置范围"
            ),
        },
        "mode": {
            "type": "string",
            "enum": ["train", "postprocess"],
            "default": DEFAULT_MODE,
            "description": "train=主链路 postprocess=后处理",
        },
        "created_by": {
            "type": "string",
            "description": "用户原始自然语言输入备份",
        },
        "created_at": {
            "type": "string",
            "description": "创建时间 ISO 格式",
        },
    },
}


# ================================================================
# 验证函数
# ================================================================

def validate_request(data: dict) -> Tuple[bool, str, dict]:
    """验证 MQ 请求消息是否符合 Schema。

    Args:
        data: 原始请求 dict

    Returns:
        (ok, error_message, normalized_data)
        - ok=True  → error_message=""  normalized 是补齐默认值后的完整 dict
        - ok=False → error_message 描述错误原因
    """
    # ---- 必填字段检查 ----
    for field in REQUEST_SCHEMA["required"]:
        if field not in data:
            return False, f"缺少必填字段: '{field}'", {}

    # ---- 字段类型检查 ----
    trail_name = data.get("trailName", "")
    if not isinstance(trail_name, str) or len(trail_name.strip()) == 0:
        return False, f"trailName 必须是非空字符串", {}

    features = data.get("features", [])
    if not isinstance(features, list) or len(features) == 0:
        return False, "features 必须是非空数组", {}

    n_trials = data.get("n_trials", DEFAULT_N_TRIALS)
    if not isinstance(n_trials, int) or n_trials < 1:
        return False, f"n_trials 必须是 >=1 的整数, 当前: {n_trials}", {}

    holdout = data.get("holdout", DEFAULT_HOLDOUT)
    if not isinstance(holdout, bool):
        return False, f"holdout 必须是布尔值", {}

    mode = data.get("mode", DEFAULT_MODE)
    if mode not in ("train", "postprocess"):
        return False, f"mode 必须是 'train' 或 'postprocess', 当前: {mode}", {}

    # ---- 业务规则校验 ----
    if mode == "postprocess" and holdout is True:
        return False, (
            "业务规则冲突: holdout=true(投产模型) 不可执行 postprocess。"
            "投产模型已全量训练，没有预留验证集用于后处理。"
        ), {}

    # ---- 补齐默认值 ----
    normalized = {
        "trailName": trail_name.strip(),
        "symbol": data.get("symbol") or DEFAULT_SYMBOL,
        "target_tf": data.get("target_tf") or DEFAULT_TARGET_TF,
        "features": features,
        "n_trials": n_trials,
        "holdout": holdout,
        "mode": mode,
        "created_by": data.get("created_by", ""),
        "created_at": data.get("created_at") or datetime.now().isoformat(),
    }

    # 可选字段类型修正
    if normalized["symbol"] not in SYMBOLS:
        logger.warning(f"symbol '{normalized['symbol']}' 不在已知列表，已接受但请确认")

    if normalized["target_tf"] not in TARGET_TFS:
        logger.warning(f"target_tf '{normalized['target_tf']}' 不在已知列表，已接受但请确认")

    return True, "", normalized


# ================================================================
# Holdout 计算
# ================================================================

def calc_holdout_cutoff(max_timestamp: datetime) -> datetime:
    """计算 holdout 切分点：max_ts 所在月 - 3 个月 → 月初。

    验证集 = [cutoff, max_ts]，即后 4 个自然月。
    例: max_ts=2026-05-30 → cutoff=2026-02-01
        验证集 = 2026-02-01 00:00 ~ 2026-05-30 全部

    Args:
        max_timestamp: MySQL candles_1m 中的最新 timestamp

    Returns:
        cutoff 时间点 (月初 00:00:00)
    """
    from dateutil.relativedelta import relativedelta

    cutoff = max_timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    cutoff = cutoff - relativedelta(months=3)

    return cutoff


def needs_holdout(data: dict) -> bool:
    """是否需要在导出时做训练/验证切分"""
    return data.get("holdout", DEFAULT_HOLDOUT) is False


# ================================================================
# 辅助函数
# ================================================================

def make_state_key(trail_name: str) -> str:
    """构建 Redis Hash 状态 key"""
    return f"{HASH_PREFIX}:{trail_name}"


def make_status_payload(
    trail_name: str,
    status: str,
    stage: Optional[int] = None,
    error: str = "",
    **extra,
) -> dict:
    """构建状态更新 payload"""
    payload = {
        "trailName": trail_name,
        "status": status,
        "updated_at": datetime.now().isoformat(),
    }
    if stage is not None:
        payload["stage"] = stage
        payload["stage_name"] = STAGE_NAMES.get(stage, f"Stage{stage}")
    if error:
        payload["error"] = error
    payload.update(extra)
    return payload


def make_output_dir(trail_name: str, created_at: Optional[str] = None) -> str:
    """构建输出目录路径: outputs/YYYY_MM/{trailName}/"""
    if created_at:
        dt = datetime.fromisoformat(created_at)
    else:
        dt = datetime.now()
    month_dir = dt.strftime("%Y_%m")
    return os.path.join(_PROJECT_ROOT, "outputs", month_dir, trail_name)


# 日志根目录（E 盘，用户不常清理 → 持久保存，便于事后追溯）
LOG_ROOT = "E:/forex_logs"


def make_log_path(trail_name: str) -> str:
    """构建每格训练的日志文件路径: E:/forex_logs/{trailName}.log

    每次完整训练一个独立日志文件（trailName 唯一 → 文件唯一，下次训练写新文件）。
    done.md 记录此绝对路径，每个完工格可追溯其完整日志。
    """
    os.makedirs(LOG_ROOT, exist_ok=True)
    return os.path.join(LOG_ROOT, f"{trail_name}.log")


def to_json(data: dict) -> str:
    """dict → JSON string (用于 XADD)"""
    return json.dumps(data, ensure_ascii=False, default=str)


def from_json(raw: str) -> dict:
    """JSON string → dict (用于 XREAD)"""
    return json.loads(raw)


# ================================================================
# 自测
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 正常请求
    ok, err, norm = validate_request({
        "trailName": "测试版_t50",
        "features": ["1m_close_ma60_ratio", "15min_close_1h_position"],
        "n_trials": 100,
        "holdout": False,
        "mode": "train",
    })
    print(f"[1] 正常: ok={ok} err='{err}'")
    print(f"    normalized: {norm['trailName']}, features={len(norm['features'])}")

    # 缺少必填字段
    ok, err, _ = validate_request({"trailName": "x"})
    print(f"[2] 缺字段: ok={ok} err='{err}'")

    # 空 features
    ok, err, _ = validate_request({
        "trailName": "x", "features": [], "n_trials": 1, "holdout": False, "mode": "train",
    })
    print(f"[3] 空features: ok={ok} err='{err}'")

    # holdout=true 时 postprocess
    ok, err, _ = validate_request({
        "trailName": "x", "features": ["a"], "n_trials": 1, "holdout": True, "mode": "postprocess",
    })
    print(f"[4] 投产模型后处理: ok={ok} err='{err}'")

    # holdout 计算
    cutoff = calc_holdout_cutoff(datetime(2026, 5, 30, 14, 30))
    print(f"[5] cutoff(2026-05-30) = {cutoff}")

    # 输出目录
    print(f"[6] output_dir = {make_output_dir('测试版_t50', '2026-05-30T10:00:00')}")
