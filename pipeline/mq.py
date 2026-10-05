"""
pipeline/mq.py — Redis Streams 生产者/消费者/状态追踪基类

提供:
  - RedisStreamClient : Redis Streams 底层封装 (XADD/XREAD/XGROUP)
  - StageRunner       : 各 Stage 的基类模板
  - StageResult       : Stage 执行结果
  - PipelineState     : 全链路状态追踪 (Redis Hash)

设计蓝图: dapanPlan/guides/mq-pipeline-blueprint.md
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import sys, os
# 确保项目根在 sys.path（脚本直接运行时需要）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import redis
from redis.exceptions import RedisError

# 兼容模块导入和脚本直接执行
try:
    from .schema import (
        HASH_PREFIX, STAGE_NAMES, STAGE_STREAMS, STREAM_REQUESTS,
        to_json, from_json,
    )
except ImportError:
    from pipeline.schema import (
        HASH_PREFIX, STAGE_NAMES, STAGE_STREAMS, STREAM_REQUESTS,
        to_json, from_json,
    )

logger = logging.getLogger(__name__)


# ================================================================
# Redis 连接配置
# ================================================================

REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "decode_responses": True,
    "socket_connect_timeout": 5,
    "socket_timeout": 10,
    "max_connections": 10,
    "protocol": 2,  # RESP2, 兼容 Redis 5.x+
}


def get_redis() -> redis.Redis:
    """获取 Redis 连接（连接池自动管理）"""
    return redis.Redis(**REDIS_CONFIG)


# ================================================================
# Stage 结果
# ================================================================

@dataclass
class StageResult:
    """Stage 执行结果

    Attributes:
        ok: 是否成功
        stage: Stage 编号 (1-6)
        message: 结果描述
        data: 传递给下游的数据 dict
        error: 错误信息 (仅 ok=False 时)
        duration_s: 执行耗时 (秒)
    """
    ok: bool
    stage: int
    message: str = ""
    data: dict = field(default_factory=dict)
    error: str = ""
    duration_s: float = 0.0

    @property
    def stage_name(self) -> str:
        return STAGE_NAMES.get(self.stage, f"Stage{self.stage}")

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stage": self.stage,
            "stage_name": self.stage_name,
            "message": self.message,
            "data": self.data,
            "error": self.error,
            "duration_s": round(self.duration_s, 1),
        }


# ================================================================
# Pipeline 状态追踪 (Redis Hash)
# ================================================================

class PipelineState:
    """全链路状态读写 — 底层 Redis Hash 操作。

    每个 trailName 对应一个 Hash: pipeline:run:{trailName}
    字段: status, stage-1~stage-6, last_error, best_auc, ...

    使用方式:
        state = PipelineState("202605首版_t50")
        state.set_stage(1, "running")
        state.set_stage(1, "done", auc=0.528, trials=100)
        status = state.get_status()
    """

    def __init__(self, trail_name: str, r: Optional[redis.Redis] = None):
        self.trail_name = trail_name
        self.key = f"{HASH_PREFIX}:{trail_name}"
        self._r = r or get_redis()

    def init(self, config: dict) -> None:
        """初始化 Hash"""
        for k, v in {
            "status": "init",
            "created_at": config.get("created_at", datetime.now().isoformat()),
            "holdout": json.dumps(config.get("holdout", False)),
            "n_trials": str(config.get("n_trials", 0)),
            "features_count": str(len(config.get("features", []))),
        }.items():
            self._r.hset(self.key, k, v)
        for s in range(1, 7):
            self._r.hset(self.key, f"stage-{s}", "pending")

    def set_status(self, status: str) -> None:
        self._r.hset(self.key, "status", status)
        self._r.hset(self.key, "updated_at", datetime.now().isoformat())

    def set_stage(self, stage: int, st: str, **extra) -> None:
        """设置某 Stage 状态 + 可选额外字段"""
        self._r.hset(self.key, f"stage-{stage}", st)
        self._r.hset(self.key, "updated_at", datetime.now().isoformat())
        for k, v in extra.items():
            self._r.hset(self.key, k, v if isinstance(v, str) else json.dumps(v))
        new_status = st if st != "done" else (self._r.hget(self.key, "status") or "running")
        self.set_status(new_status)

    def set_error(self, stage: int, error: str) -> None:
        self.set_stage(stage, "failed")
        self._r.hset(self.key, "last_error", error)
        self.set_status("failed")

    def set_done(self) -> None:
        self.set_status("done")

    def get_status(self) -> dict:
        raw = self._r.hgetall(self.key)
        if not raw:
            return {}
        return {k: v for k, v in raw.items()}

    def is_stage_done(self, stage: int) -> bool:
        return self._r.hget(self.key, f"stage-{stage}") == "done"

    def is_stage_failed(self, stage: int) -> bool:
        return self._r.hget(self.key, f"stage-{stage}") == "failed"

    def get_last_error(self) -> str:
        return self._r.hget(self.key, "last_error") or ""

    def exists(self) -> bool:
        return self._r.exists(self.key) > 0

    def __repr__(self) -> str:
        return f"PipelineState({self.trail_name})"


# ================================================================
# Redis List 操作 (LPUSH/BRPOP, 兼容 Redis 3.x+)
# ================================================================

class RedisStreamClient:
    """Redis List 生产者/消费者封装。

    使用 Redis List 而非 Streams，兼容 Redis 3.x+。
    生产者 LPUSH，消费者 BRPOP(FIFO)。
    """

    def __init__(self, r: Optional[redis.Redis] = None):
        self._r = r or get_redis()

    # ---- 请求入口 ----
    def push_request(self, data: dict) -> int:
        """推送训练/后处理请求到 training:requests (List)"""
        return self._r.lpush(STREAM_REQUESTS, to_json(data))

    def read_requests(self, count: int = 1, block_ms: int = 5000) -> List[dict]:
        """阻塞读取训练请求 (BRPOP, FIFO)

        阻塞等待最多 block_ms 毫秒。如果超时返回空列表。
        """
        # BRPOP: 返回 (key, value) 或 None (超时)
        result = self._r.brpop(STREAM_REQUESTS, timeout=block_ms // 1000)
        if not result:
            return []

        _, raw = result
        try:
            data = from_json(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Invalid JSON in queue message")
            return []

        return [{"msg_id": "", "data": data}]

    # ---- Stage 间消息传递 ----
    def push_stage_result(self, stage: int, result: StageResult) -> int:
        """推送 Stage 完成结果到下游 List"""
        key = STAGE_STREAMS.get(stage)
        if not key:
            raise ValueError(f"Unknown stage: {stage}")
        return self._r.lpush(key, to_json(result.to_dict()))

    def read_stage_result(self, stage: int, block_ms: int = 5000) -> Optional[StageResult]:
        """阻塞等待上游 Stage 的结果 (BRPOP)"""
        key = STAGE_STREAMS.get(stage - 1) if stage > 1 else STREAM_REQUESTS
        if not key:
            return None

        result = self._r.brpop(key, timeout=block_ms // 1000)
        if not result:
            return None

        _, raw = result
        try:
            data = from_json(raw)
            return StageResult(
                ok=data.get("ok", False),
                stage=data.get("stage", stage - 1),
                message=data.get("message", ""),
                data=data.get("data", {}),
                error=data.get("error", ""),
                duration_s=data.get("duration_s", 0.0),
            )
        except (json.JSONDecodeError, TypeError):
            return None

    # ---- 队列管理 ----
    def queue_length(self) -> int:
        """当前队列中待处理消息数"""
        return self._r.llen(STREAM_REQUESTS)

    # ---- 健康检查 ----
    def ping(self) -> bool:
        try:
            return self._r.ping()
        except RedisError:
            return False


# ================================================================
# Stage Runner 基类 — 所有 Stage 实现继承此模板
# ================================================================

class StageRunner(ABC):
    """Stage 基类模板。

    子类需要实现:
        - name: str         Stage 名称
        - stage_num: int    Stage 编号 (1-6)
        - run(config, prev_data) -> StageResult  核心逻辑

    使用方式:
        runner = MyStage()
        result = runner.execute(config, prev_data)
        if not result.ok:
            raise ...
    """

    name: str = ""
    stage_num: int = 0

    def __init__(self, r: Optional[redis.Redis] = None):
        self._r = r or get_redis()
        self.state: Optional[PipelineState] = None

    def execute(self, config: dict, prev_data: Optional[dict] = None) -> StageResult:
        """执行 Stage 并返回结果。

        Args:
            config: 标准化的训练请求 dict
            prev_data: 上游 Stage 传递的数据

        Returns:
            StageResult
        """
        trail_name = config.get("trailName", "unknown")
        self.state = PipelineState(trail_name, self._r)

        # 检查上游是否失败
        if self.stage_num > 1:
            prev_stage = self.stage_num - 1
            if self.state.is_stage_failed(prev_stage):
                err = self.state.get_last_error()
                logger.warning(f"[{self.name}] 上游 Stage {prev_stage} 已失败，跳过: {err}")
                return StageResult(ok=False, stage=self.stage_num,
                                   error=f"上游 Stage {prev_stage} 失败: {err}")

        # 更新状态为 running
        self.state.set_stage(self.stage_num, "running")
        logger.info(f"[{self.name}] 开始执行 | trailName={trail_name}")

        t0 = time.time()
        try:
            result = self.run(config, prev_data or {})
            result.stage = self.stage_num
            result.duration_s = time.time() - t0

            if result.ok:
                self.state.set_stage(self.stage_num, "done", **result.data)
                logger.info(f"[{self.name}] 完成 ✅ ({result.duration_s:.0f}s) | {result.message}")
            else:
                self.state.set_error(self.stage_num, result.error)
                logger.error(f"[{self.name}] 失败 ❌ | {result.error}")
        except Exception as e:
            elapsed = time.time() - t0
            err_msg = f"{type(e).__name__}: {e}"
            self.state.set_error(self.stage_num, err_msg)
            logger.error(f"[{self.name}] 异常 ❌ ({elapsed:.0f}s) | {err_msg}", exc_info=True)
            result = StageResult(ok=False, stage=self.stage_num,
                                  error=err_msg, duration_s=elapsed)

        return result

    @abstractmethod
    def run(self, config: dict, prev_data: dict) -> StageResult:
        """子类实现核心逻辑"""
        ...

    def skip(self, reason: str = "") -> StageResult:
        """返回一个 SKIP 结果（用于后处理时跳过不需要的 Stage）"""
        logger.info(f"[{self.name}] SKIP | {reason}")
        return StageResult(ok=True, stage=self.stage_num,
                           message=f"SKIP: {reason}", data={})


# ================================================================
# 便利函数
# ================================================================

def check_redis() -> bool:
    """检查 Redis 是否可用"""
    try:
        r = get_redis()
        return r.ping()
    except RedisError as e:
        logger.error(f"Redis 连接失败: {e}")
        return False


def wait_for_redis(max_wait: int = 30) -> bool:
    """等待 Redis 可用 (最多 max_wait 秒)"""
    for i in range(max_wait):
        if check_redis():
            logger.info("Redis 连接成功")
            return True
        time.sleep(1)
    logger.error(f"Redis 在 {max_wait}s 内未就绪")
    return False


# ================================================================
# 自测
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"Redis 连接: {check_redis()}")

    # 测试 PipelineState
    r = get_redis()
    if check_redis():
        state = PipelineState("_mq_test", r)
        r.delete(state.key)  # 清理
        state.init({"trailName": "_mq_test", "holdout": False, "n_trials": 1, "features": ["a"]})
        state.set_stage(1, "done", rows=4187)
        state.set_stage(2, "running")
        print(f"State: {state.get_status()}")
        r.delete(state.key)
        print("MQ 测试通过 [OK]")
    else:
        print("[WARN] Redis 不可用，MQ 功能需要启动 Redis 服务")
