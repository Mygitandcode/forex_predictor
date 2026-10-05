"""
pipeline/stage4_train.py — Stage 4: Optuna 训练 + 增量模式

封装 training/stage2_optuna.py:
  主链路: 全量训练 → model.pkl
  后处理: trial<50 → 增量训练; trial>=50 → SKIP
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .mq import StageRunner, StageResult

logger = logging.getLogger(__name__)


class Stage4Train(StageRunner):
    name = "Optuna训练"
    stage_num = 4

    def run(self, config: dict, prev_data: dict) -> StageResult:
        trail_name = config["trailName"]
        symbol = config.get("symbol", "EURUSD")
        target_tf = config.get("target_tf", "15m")
        n_trials = config["n_trials"]
        mode = config.get("mode", "train")
        # FeatureStore 的 data_dir: E:/forex_npy/{trail_name}
        # 实际文件在: E:/forex_npy/{trail_name}/{symbol}/
        dst_dir = prev_data.get("dst_dir", f"E:/forex_npy/{trail_name}/{symbol}")
        feature_data_dir = f"E:/forex_npy/{trail_name}"

        script = Path(__file__).parent.parent / "training" / "stage2_optuna.py"

        if mode == "postprocess":
            return self._run_postprocess(config, prev_data, script, dst_dir, feature_data_dir)
        else:
            return self._run_train(config, prev_data, script, dst_dir, feature_data_dir,
                                   symbol, target_tf, n_trials, trail_name)

    def _run_train(self, config, prev_data, script, dst_dir, feature_data_dir, symbol, target_tf, n_trials, trail_name):
        """主链路训练"""
        # 检查训练数据
        train_npy = os.path.join(dst_dir, "features_train.npy")
        val_npy = os.path.join(dst_dir, "features_val.npy")

        if not os.path.exists(train_npy):
            return StageResult(ok=False, stage=4,
                               error=f"训练数据不存在: {train_npy}")

        # stage2_optuna 的 FeatureStore 期望 features.npy 和 labels_{target_tf}.npy
        # 把 train 文件链接为默认名
        for src, dst_name in [
            ("features_train.npy", "features.npy"),
            ("labels_train.npy", f"labels_{target_tf}.npy"),
            ("timestamps.npy", "timestamps.npy"),
            ("weights.npy", "weights.npy"),
        ]:
            src_path = os.path.join(dst_dir, src)
            dst_path = os.path.join(dst_dir, dst_name)
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

        logger.info(f"开始训练: {trail_name}, n_trials={n_trials}, data={feature_data_dir}")

        log_path = config.get("log_path")  # 本次运行日志文件（E 盘）
        log_fh = open(log_path, "ab") if log_path else None
        try:
            # 清除 PyCharm debugpy 环境变量, 防止子进程被调试器拖慢 7x
            # PyCharm 通过 PYTHONPATH + PYDEVD_* 双重注入 debugpy, 必须全部清除
            child_env = {k: v for k, v in os.environ.items()
                         if not any(x in k.upper() for x in ("DEBUGPY", "PYDEVD", "PYCHARM", "PYTHONPATH", "PYCHARM_HOSTED", "PYTHONUNBUFFERED"))}
            stripped = [k for k in os.environ if any(x in k.upper() for x in ("DEBUGPY", "PYDEVD", "PYCHARM", "PYTHONPATH", "PYCHARM_HOSTED"))]
            logger.info(f"Subprocess: 剥离调试变量={stripped if stripped else '无'} → 子进程原生速度")
            result = subprocess.run(
                [sys.executable, "-u", "-m", "training.stage2_optuna",
                 "--symbol", symbol,
                 "--target", target_tf,
                 "--n-trials", str(n_trials),
                 "--data-dir", feature_data_dir,
                 "--run-name", trail_name,
                 "--mlflow-uri", "http://localhost:5001"],  # 5000 被企业微信 WXWork 占用，MLflow 迁至 5001
                stdout=log_fh if log_fh else subprocess.DEVNULL,
                stderr=log_fh if log_fh else subprocess.DEVNULL,  # 重定向到日志文件：防死锁+留日志
                env=child_env,
                timeout=86400,  # 24h timeout for 100 trials
                cwd=str(Path(__file__).parent.parent),
            )

            if result.returncode != 0:
                return StageResult(ok=False, stage=4,
                                   error=f"stage2_optuna rc={result.returncode}, 详见日志: {log_path}")

        except subprocess.TimeoutExpired:
            return StageResult(ok=False, stage=4,
                               error=f"训练超时 (>24h, n_trials={n_trials})")
        finally:
            if log_fh:
                log_fh.close()

        # 查找训练产出的模型文件
        model_path = self._find_model(trail_name)
        if not model_path:
            return StageResult(ok=False, stage=4,
                               error=f"训练完成但未找到模型文件: {trail_name}*.pkl (在 D:/dapanAllThings/codes/forex_predictor/models/ 下)")

        return StageResult(ok=True, stage=4,
                           message=f"训练完成, model={os.path.basename(model_path)}",
                           data={"model_path": os.path.abspath(model_path),
                                 "n_trials": n_trials})

    def _run_postprocess(self, config, prev_data, script, dst_dir, feature_data_dir):
        """后处理模式"""
        trail_name = config["trailName"]
        n_trials = config["n_trials"]

        model_path = self._find_model(trail_name)
        if not model_path:
            return StageResult(ok=False, stage=4,
                               error=f"后处理: 未找到模型文件 {trail_name}*.pkl (在 D:/dapanAllThings/codes/forex_predictor/models/ 下)")

        if n_trials < 50:
            # 增量训练
            logger.info(f"后处理增量训练: trial<50 ({n_trials}), 追加 trial")
            # 复用 train 模式逻辑（--n-trials 语义=新增trial数）
            return self._run_train(config, prev_data, script, dst_dir, feature_data_dir,
                                   config.get("symbol", "EURUSD"),
                                   config.get("target_tf", "15m"),
                                   n_trials, trail_name)
        else:
            # trial>=50: SKIP
            logger.info(f"后处理跳过训练: trial>=50 ({n_trials})")
            return StageResult(ok=True, stage=4,
                               message=f"SKIP: trial>=50, 模型不变",
                               data={"model_path": os.path.abspath(model_path)})

    def _find_model(self, trail_name: str) -> Optional[str]:
        """查找 trailName 对应的模型文件（可能带前缀）"""
        models_dir = Path(__file__).parent.parent / "models"
        # 精确匹配
        exact = models_dir / f"{trail_name}.pkl"
        if exact.exists():
            return str(exact)
        # 模糊匹配: *{trailName}*.pkl
        matches = list(models_dir.glob(f"*{trail_name}*.pkl"))
        if matches:
            return str(sorted(matches, key=os.path.getmtime, reverse=True)[0])
        return None
