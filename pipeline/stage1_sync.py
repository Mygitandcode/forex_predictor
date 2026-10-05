"""
pipeline/stage1_sync.py — Stage 1: 拉齐原始数据

封装 _data_sync.py: CSV → MySQL candles_1m~1h
预检: 如果 candles_1m 已有足够数据 → SKIP
"""

from __future__ import annotations

import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional

import pymysql

from .mq import StageRunner, StageResult

logger = logging.getLogger(__name__)

MYSQL_CONFIG = dict(
    host="localhost", port=3305, user="root", password="",
    database="forex_predictor", charset="utf8mb4",
    connect_timeout=5,
)


class Stage1DataSync(StageRunner):
    name = "拉齐原始数据"
    stage_num = 1

    def run(self, config: dict, prev_data: dict) -> StageResult:
        trail_name = config["trailName"]
        symbol = config.get("symbol", "EURUSD")
        log_path = config.get("log_path")  # 本次运行日志文件（E 盘）

        # ---- 预检: 检查 candles_1m 是否已有数据 ----
        expected_min_rows = config.get("min_candle_rows", 1000)
        current_count = self._count_candles(symbol)

        if current_count >= expected_min_rows:
            logger.info(f"candles_1m 已有 {current_count:,} 行 (>= {expected_min_rows}), 跳过同步")
            return StageResult(ok=True, stage=1,
                               message=f"数据已对齐, candles_1m={current_count:,}行",
                               data={"candles_1m_rows": current_count})

        # ---- 执行数据同步 ----
        script = Path(__file__).parent.parent / "_data_sync.py"
        logger.info(f"执行数据同步: {script}")

        log_fh = open(log_path, "ab") if log_path else None
        try:
            child_env = {k: v for k, v in os.environ.items()
                         if not any(x in k.upper() for x in ("DEBUGPY", "PYDEVD", "PYCHARM", "PYTHONPATH", "PYCHARM_HOSTED"))}
            stripped = [k for k in os.environ if any(x in k.upper() for x in ("DEBUGPY", "PYDEVD", "PYCHARM", "PYTHONPATH", "PYCHARM_HOSTED"))]
            logger.info(f"Subprocess: 剥离调试变量={stripped if stripped else '无'} → 子进程原生速度")
            result = subprocess.run(
                [sys.executable, "-u", str(script), "--symbol", symbol],
                stdout=log_fh if log_fh else subprocess.DEVNULL,
                stderr=log_fh if log_fh else subprocess.DEVNULL,  # 重定向到日志文件：防死锁+留日志
                env=child_env,
                timeout=3600,  # 1h timeout
                cwd=str(Path(__file__).parent.parent),
            )

            if result.returncode != 0:
                return StageResult(ok=False, stage=1,
                                   error=f"_data_sync.py 返回 {result.returncode}, 详见日志: {log_path}")

        except subprocess.TimeoutExpired:
            return StageResult(ok=False, stage=1, error="_data_sync.py 超时 (>1h)")
        except FileNotFoundError:
            return StageResult(ok=False, stage=1,
                               error=f"脚本不存在: {script}")
        finally:
            if log_fh:
                log_fh.close()

        # ---- 验证 ----
        new_count = self._count_candles(symbol)
        if new_count < expected_min_rows:
            return StageResult(ok=False, stage=1,
                               error=f"candles_1m 行数不足: {new_count} < {expected_min_rows}")

        return StageResult(ok=True, stage=1,
                           message=f"数据同步完成, candles_1m={new_count:,}行",
                           data={"candles_1m_rows": new_count})

    def _count_candles(self, symbol: str) -> int:
        try:
            conn = pymysql.connect(**MYSQL_CONFIG)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM candles_1m WHERE symbol=%s", (symbol,))
            n = cur.fetchone()[0]
            cur.close(); conn.close()
            return n
        except Exception as e:
            logger.warning(f"无法读取 candles_1m 行数: {e}")
            return 0
