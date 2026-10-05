"""
pipeline/stage3_export.py — Stage 3: 特征导出 + 裁剪 + holdout 切分

封装 _export_v2.py:
  1. 写入裁剪后的特征列表 JSON 文件
  2. 调用 _export_v2.py --features xxx.json
  3. holdout=false → 切分 train/val (后4自然月)
  4. holdout=true  → train=val=全量

后处理模式: SKIP (val.npy 已存在, 不重建)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pymysql

from .mq import StageRunner, StageResult
from .schema import needs_holdout, calc_holdout_cutoff

logger = logging.getLogger(__name__)

MYSQL_CONFIG = dict(
    host="localhost", port=3305, user="root", password="",
    database="forex_predictor", charset="utf8mb4",
    connect_timeout=5,
)


class Stage3Export(StageRunner):
    name = "特征导出"
    stage_num = 3

    def run(self, config: dict, prev_data: dict) -> StageResult:
        trail_name = config["trailName"]
        symbol = config.get("symbol", "EURUSD")
        target_tf = config.get("target_tf", "15m")
        features = config["features"]
        holdout = needs_holdout(config)
        mode = config.get("mode", "train")
        log_path = config.get("log_path")  # 本次运行日志文件（E 盘）
        # FeatureStore 布局: data_root/{symbol}/features.npy
        # export 数据输出到 E:/forex_npy/{trail_name}/{symbol}/
        dst_dir = f"E:/forex_npy/{trail_name}/{symbol}"

        # ---- 后处理模式: 直接 SKIP ----
        if mode == "postprocess":
            val_npy = os.path.join(dst_dir, "features_val.npy")
            if os.path.exists(val_npy):
                logger.info(f"后处理模式: val.npy 已存在, SKIP")
                return StageResult(ok=True, stage=3,
                                   message="SKIP: val.npy 已存在",
                                   data={"dst_dir": dst_dir,
                                         "features_count": len(features)})
            else:
                return StageResult(ok=False, stage=3,
                                   error=f"后处理模式但 val.npy 不存在: {val_npy}。请先运行主链路 train。")

        # ---- 训练模式 ----
        # Step 1: 写入裁剪特征列表 JSON
        feature_json_path = f"{dst_dir}_features.json"
        os.makedirs(os.path.dirname(feature_json_path) or dst_dir, exist_ok=True)
        with open(feature_json_path, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        logger.info(f"特征列表已写入: {feature_json_path} ({len(features)} features)")

        # Step 2: 执行 _export_v2.py
        script = Path(__file__).parent.parent / "_export_v2.py"

        # 先导出全量（不裁剪），再切分
        logger.info(f"执行特征导出: {script} --features {feature_json_path} --dst {dst_dir} --symbol {symbol} --target {target_tf}")

        log_fh = open(log_path, "ab") if log_path else None
        try:
            # 注意: 不用 capture_output=True!
            # _export_v2.py 每批 flush=True 输出大量进度, 管道缓冲区满会导致父子进程死锁
            child_env = {k: v for k, v in os.environ.items()
                         if not any(x in k.upper() for x in ("DEBUGPY", "PYDEVD", "PYCHARM", "PYTHONPATH", "PYCHARM_HOSTED"))}
            stripped = [k for k in os.environ if any(x in k.upper() for x in ("DEBUGPY", "PYDEVD", "PYCHARM", "PYTHONPATH", "PYCHARM_HOSTED"))]
            logger.info(f"Subprocess: 剥离调试变量={stripped if stripped else '无'} → 子进程原生速度")
            result = subprocess.run(
                [sys.executable, "-u", str(script),
                 "--features", feature_json_path,
                 "--dst", dst_dir,
                 "--symbol", symbol,
                 "--target", target_tf],
                stdout=log_fh if log_fh else subprocess.DEVNULL,
                stderr=log_fh if log_fh else subprocess.DEVNULL,  # 重定向到日志文件：防死锁+留日志
                env=child_env,
                timeout=7200,
                cwd=str(Path(__file__).parent.parent),
            )
            if result.returncode != 0:
                return StageResult(ok=False, stage=3,
                                   error=f"_export_v2.py rc={result.returncode}, 详见日志: {log_path}")
        except subprocess.TimeoutExpired:
            return StageResult(ok=False, stage=3, error="_export_v2.py 超时 (>2h)")
        finally:
            if log_fh:
                log_fh.close()

        # Step 3: 验证总导出
        features_npy = os.path.join(dst_dir, "features.npy")
        labels_npy = os.path.join(dst_dir, f"labels_{target_tf}.npy")
        timestamps_npy = os.path.join(dst_dir, "timestamps.npy")
        done_file = os.path.join(dst_dir, ".done")

        if not os.path.exists(features_npy):
            return StageResult(ok=False, stage=3, error=f"导出失败: features.npy 不存在于 {dst_dir}")
        if not os.path.exists(done_file):
            return StageResult(ok=False, stage=3, error=f"导出失败: .done 不存在于 {dst_dir}")

        X = np.load(features_npy, mmap_mode='r')
        y = np.load(labels_npy, mmap_mode='r')
        timestamps = np.load(timestamps_npy, allow_pickle=True)

        expected_dims = len(features)
        if X.shape[1] != expected_dims:
            return StageResult(ok=False, stage=3,
                               error=f"维度不匹配: features.npy={X.shape[1]} vs 期望={expected_dims}")

        n_total = X.shape[0]
        logger.info(f"导出完成: X={X.shape}, y={y.shape}, dims={expected_dims}")

        # Step 4: holdout 切分
        if holdout:
            # 获取 MySQL 最新 candle timestamp
            max_ts = self._get_max_candle_ts(symbol)
            if max_ts is None:
                logger.warning("无法获取 candles_1m 最大时间戳，跳过 holdout 切分")
                shutil.copy(features_npy, os.path.join(dst_dir, "features_train.npy"))
                shutil.copy(labels_npy, os.path.join(dst_dir, "labels_train.npy"))
                return StageResult(ok=True, stage=3,
                                   message="全量导出 (无holdout, DB无时间戳)",
                                   data={"dst_dir": dst_dir, "n_total": n_total,
                                         "features_count": expected_dims})

            cutoff = calc_holdout_cutoff(max_ts)
            cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Holdout 切分: cutoff={cutoff_str} (max_ts={max_ts})")

            # 如果 train/val 文件已存在 (上次导出留下的), 直接跳过切分
            if os.path.exists(os.path.join(dst_dir, "features_train.npy")):
                logger.info("Holdout: train/val 文件已存在, 跳过切分")
            else:
                # 找 cutoff 索引: 用 mmap 加载 timestamps, 二进制搜索 datetime64
                ts_mmap = np.load(timestamps_npy, mmap_mode='r', allow_pickle=True)
                # pandas 的 to_datetime 能高效处理混合类型的时间戳数组
                import pandas as pd
                ts_series = pd.to_datetime(pd.Series(ts_mmap))
                cutoff_idx = ts_series.searchsorted(pd.Timestamp(cutoff))
                del ts_mmap  # 释放 mmap

                if cutoff_idx <= 0 or cutoff_idx >= n_total:
                    logger.warning(f"cutoff 索引无效: {cutoff_idx}/{n_total}, 使用 96% 近似切")
                    cutoff_idx = int(n_total * 0.96)

                # 切分
                X_train = X[:cutoff_idx]
                y_train = y[:cutoff_idx]
                X_val = X[cutoff_idx:]
                y_val = y[cutoff_idx:]

                np.save(os.path.join(dst_dir, "features_train.npy"), X_train)
                np.save(os.path.join(dst_dir, "labels_train.npy"), y_train)
                np.save(os.path.join(dst_dir, "features_val.npy"), X_val)
                np.save(os.path.join(dst_dir, "labels_val.npy"), y_val)

                logger.info(f"Holdout 完成: train={X_train.shape}, val={X_val.shape}")
        else:
            # holdout=false: train = val = 全量
            logger.info("holdout=true (投产), train=val=全量")
            shutil.copy(features_npy, os.path.join(dst_dir, "features_train.npy"))
            shutil.copy(labels_npy, os.path.join(dst_dir, "labels_train.npy"))
            shutil.copy(features_npy, os.path.join(dst_dir, "features_val.npy"))
            shutil.copy(labels_npy, os.path.join(dst_dir, "labels_val.npy"))

        # Step 5: 写入 holdout 元信息
        meta_path = os.path.join(dst_dir, "export_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["holdout"] = holdout
            if holdout:
                meta["cutoff"] = cutoff_str if 'cutoff' in dir() else "N/A"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        return StageResult(ok=True, stage=3,
                           message=f"特征导出完成, dims={expected_dims}, total={n_total}",
                           data={"dst_dir": dst_dir,
                                 "n_total": n_total,
                                 "features_count": expected_dims,
                                 "holdout": holdout})

    def _get_max_candle_ts(self, symbol: str) -> Optional[datetime]:
        try:
            conn = pymysql.connect(**MYSQL_CONFIG)
            cur = conn.cursor()
            cur.execute("SELECT MAX(timestamp) FROM candles_1m WHERE symbol=%s", (symbol,))
            row = cur.fetchone()
            cur.close(); conn.close()
            if row and row[0]:
                return row[0]
        except Exception as e:
            logger.warning(f"无法读取 candles_1m 最大时间戳: {e}")
        return None
