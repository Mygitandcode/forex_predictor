"""
pipeline/stage2_features.py — Stage 2: 特征构建

封装 _build_features_v2.py --mode full
预检: features_a 行数对齐 → SKIP
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pymysql

from .mq import StageRunner, StageResult

logger = logging.getLogger(__name__)

MYSQL_CONFIG = dict(
    host="localhost", port=3305, user="root", password="",
    database="forex_predictor", charset="utf8mb4",
    connect_timeout=5,
)

WARMUP = 200  # FeatureBuilder 需要 200 根 bar 预热


class Stage2FeatureBuild(StageRunner):
    name = "特征构建"
    stage_num = 2

    def run(self, config: dict, prev_data: dict) -> StageResult:
        trail_name = config["trailName"]
        symbol = config.get("symbol", "EURUSD")
        log_path = config.get("log_path")  # 本次运行日志文件（E 盘）

        # ---- 预检: 特征是否已构建 ----
        n_candles = self._count("candles_1m", symbol)
        n_features = self._count("features_a", symbol)

        expected = max(0, n_candles - WARMUP)
        if n_features == expected and n_features > 0:
            logger.info(f"features_a 已对齐: {n_features:,} (candles={n_candles:,}, WARMUP={WARMUP})")
            return StageResult(ok=True, stage=2,
                               message=f"特征已对齐, features_a={n_features:,}行",
                               data={"features_rows": n_features,
                                     "candles_rows": n_candles})

        # ---- 执行特征构建 ----
        script = Path(__file__).parent.parent / "_build_features_v2.py"
        logger.info(f"执行特征构建 (full mode): {script}")

        log_fh = open(log_path, "ab") if log_path else None
        try:
            child_env = {k: v for k, v in os.environ.items()
                         if not any(x in k.upper() for x in ("DEBUGPY", "PYDEVD", "PYCHARM", "PYTHONPATH", "PYCHARM_HOSTED"))}
            stripped = [k for k in os.environ if any(x in k.upper() for x in ("DEBUGPY", "PYDEVD", "PYCHARM", "PYTHONPATH", "PYCHARM_HOSTED"))]
            logger.info(f"Subprocess: 剥离调试变量={stripped if stripped else '无'} → 子进程原生速度")
            result = subprocess.run(
                [sys.executable, "-u", str(script), "--mode", "full", "--symbol", symbol],
                stdout=log_fh if log_fh else subprocess.DEVNULL,
                stderr=log_fh if log_fh else subprocess.DEVNULL,  # 重定向到日志文件：防死锁+留日志
                env=child_env,
                timeout=7200,  # 2h timeout
                cwd=str(Path(__file__).parent.parent),
            )

            if result.returncode != 0:
                return StageResult(ok=False, stage=2,
                                   error=f"_build_features_v2.py rc={result.returncode}, 详见日志: {log_path}")

        except subprocess.TimeoutExpired:
            return StageResult(ok=False, stage=2, error="_build_features_v2.py 超时 (>2h)")
        except FileNotFoundError:
            return StageResult(ok=False, stage=2, error=f"脚本不存在: {script}")
        finally:
            if log_fh:
                log_fh.close()

        # ---- 验证 ----
        n_features_new = self._count("features_a", symbol)
        if n_features_new < expected * 0.99:  # 允许 1% 行数偏差
            return StageResult(ok=False, stage=2,
                               error=f"features_a 行数不足: {n_features_new} vs 期望 {expected}")

        return StageResult(ok=True, stage=2,
                           message=f"特征构建完成, features={n_features_new:,}行",
                           data={"features_rows": n_features_new,
                                 "candles_rows": n_candles,
                                 "warmup": WARMUP})

    def _count(self, table: str, symbol: str) -> int:
        try:
            conn = pymysql.connect(**MYSQL_CONFIG)
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE symbol=%s", (symbol,))
            n = cur.fetchone()[0]
            cur.close(); conn.close()
            return n
        except Exception as e:
            logger.warning(f"无法读取 {table} 行数: {e}")
            return 0
