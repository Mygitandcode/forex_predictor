"""
pipeline/orchestrator.py — 主调度器

单进程串行: 消费 training:requests → 依次执行 Stage1~6 → 写入状态

使用:
    python -m pipeline.orchestrator          # 启动守护进程
    python -m pipeline.orchestrator --once   # 处理一条消息后退出
"""

from __future__ import annotations

import logging
import signal
import sys
import os
import time
from typing import Optional

# 确保项目根在 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.schema import (
    validate_request,
    make_output_dir,
    make_log_path,
    make_status_payload,
    STAGE_NAMES,
)
from pipeline.mq import (
    RedisStreamClient,
    PipelineState,
    StageResult,
    check_redis,
    wait_for_redis,
    get_redis,
)
from pipeline.stage1_sync import Stage1DataSync
from pipeline.stage2_features import Stage2FeatureBuild
from pipeline.stage3_export import Stage3Export
from pipeline.stage4_train import Stage4Train
from pipeline.stage5_validate import Stage5Validate
from pipeline.stage6_report import Stage6Report

from pipeline.jsonlog import setup_json_logging, JsonFormatter  # JSON Lines 日志

logger = logging.getLogger(__name__)

# 注册所有 Stage（添加新 Stage 时只需在此注册）
STAGE_REGISTRY = {
    1: Stage1DataSync(),
    2: Stage2FeatureBuild(),
    3: Stage3Export(),
    4: Stage4Train(),
    5: Stage5Validate(),
    6: Stage6Report(),
}


class Orchestrator:
    """主调度器 — 消费 MQ 消息 → 串行执行 Stage"""

    def __init__(self):
        self.mq = RedisStreamClient()
        self.running = True
        self._current_trail: Optional[str] = None

    def start(self, once: bool = False):
        """启动消费循环"""
        logger.info("=== Orchestrator 启动 ===")

        # 等待 Redis
        if not wait_for_redis(max_wait=10):
            logger.error("Redis 不可用，退出")
            return

        # 信号处理: 优雅退出
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        while self.running:
            try:
                messages = self.mq.read_requests(count=1, block_ms=5000)
                if not messages:
                    continue

                for msg in messages:
                    self._process_message(msg)
                    if once:
                        return

            except Exception as e:
                logger.error(f"消费循环异常: {e}", exc_info=True)
                time.sleep(5)

        logger.info("Orchestrator 已退出")

    def _process_message(self, msg: dict):
        """处理单条 MQ 消息"""
        data = msg["data"]
        trail_name = data.get("trailName", "unknown")
        mode = data.get("mode", "train")

        logger.info(f"\n{'='*60}")
        logger.info(f"收到请求: {trail_name} | mode={mode}")
        logger.info(f"{'='*60}")

        # ---- 1. 验证 ----
        ok, err, config = validate_request(data)
        if not ok:
            logger.error(f"请求验证失败: {err}")
            return

        self._current_trail = config["trailName"]

        # ---- 2. 初始化状态 ----
        state = PipelineState(config["trailName"])
        state.init(config)

        # ---- 3. 创建输出目录 ----
        output_dir = make_output_dir(config["trailName"], config.get("created_at"))
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"输出目录: {output_dir}")

        # 保存输入配置
        import json as _json
        config_path = os.path.join(output_dir, "01_input_config.md")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(f"# 训练配置: {config['trailName']}\n\n")
            f.write(f"```json\n{_json.dumps(config, ensure_ascii=False, indent=2)}\n```\n")
        logger.info(f"配置已保存: {config_path}")

        # ---- 3b. 设置每格日志文件（E 盘持久化，一次训练一个文件）----
        log_path = make_log_path(config["trailName"])
        config["log_path"] = log_path  # 传给各 Stage 用于 subprocess 重定向
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        logging.getLogger().addHandler(file_handler)

        try:
            logger.info(f"\n{'#'*60}")
            logger.info(f"# 训练运行开始: {config['trailName']} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"# 日志文件: {log_path}")
            logger.info(f"{'#'*60}")

            # ---- 4. 串行执行 Stage ----
            prev_data: dict = {"output_dir": output_dir, "log_path": log_path}
            trail_complete = True
            test_stopped = False

            for stage_num in sorted(STAGE_REGISTRY.keys()):
                runner = STAGE_REGISTRY[stage_num]
                logger.info(f"\n--- Stage {stage_num}: {runner.name} ---")

                result = runner.execute(config, prev_data)

                if result.ok:
                    prev_data.update(result.data)
                    logger.info(f"Stage {stage_num} 完成: {result.message}")
                    # 测试停止点: FI_STOP_AFTER_STAGE=N 跑完 Stage N 后停(逐步累加测试用)
                    _stop_after = int(os.environ.get("FI_STOP_AFTER_STAGE", "0"))
                    if _stop_after and stage_num >= _stop_after:
                        logger.info(f"FI_STOP_AFTER_STAGE={_stop_after} 命中, 测试停止(非完成, 重启可续)")
                        trail_complete = False
                        test_stopped = True
                        break
                else:
                    trail_complete = False
                    logger.error(f"Stage {stage_num} 失败: {result.error}")
                    break

            # ---- 5. 最终状态 ----
            if trail_complete:
                state.set_done()
                logger.info(f"\n{'='*60}")
                logger.info(f"全链路完成: {config['trailName']}")
                logger.info(f"输出目录: {output_dir}")
                logger.info(f"日志文件: {log_path}")
                logger.info(f"{'='*60}")
            elif test_stopped:
                logger.info(f"\n测试停(FI_STOP_AFTER_STAGE): {config['trailName']} - 非完成非失败, 重启 orchestrator 可续跑")
            else:
                logger.error(f"\n链路中止: {config['trailName']}")
                logger.error(f"已完成至 Stage 失败之前")
        finally:
            # ---- 6. 关闭本次运行日志 handler（防泄漏，Java try-finally 同款）----
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()

        self._current_trail = None

    def _handle_signal(self, signum, frame):
        logger.info(f"收到信号 {signum}，优雅退出...")
        self.running = False


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse

    setup_json_logging()

    parser = argparse.ArgumentParser(description="MQ 全链路训练调度器")
    parser.add_argument("--once", action="store_true", help="处理一条消息后退出")
    args = parser.parse_args()

    orch = Orchestrator()
    orch.start(once=args.once)
