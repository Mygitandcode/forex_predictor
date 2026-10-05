"""
stage1_autogluon.py — Stage 1: AutoGluon + MLflow 初始探索

目标: 在13年历史数据上，快速找到最优模型类型和参数范围。
AutoGluon 自动尝试多种模型(RF/GBDT/NN等)，MLflow 自动追踪。

流程:
    1. mlflow.autogluon.autolog() 开启自动追踪
    2. AutoGluon TabularPredictor 自动尝试多种模型
    3. MLflow 自动记录: 参数、指标、模型文件、特征重要性
    4. 输出: 最佳模型类型 + 参数范围 + MLflow experiment_id

使用方式:
    python -m training.stage1_autogluon \
        --symbol EURUSD \
        --target 15m \
        --data-dir ./data_store \
        --time-budget 3600
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_store import FeatureStore
from training.model_eval import ModelEvaluator

logger = logging.getLogger(__name__)


class Stage1Trainer:
    """AutoGluon 初始探索训练器"""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "forex_stage1",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def run(
        self,
        symbol: str,
        target_window: str = "15m",
        data_dir: str = "./data_store",
        time_limit: int = 3600,
        presets: str = "best_quality",
        eval_metric: str = "roc_auc",
        test_size: float = 0.2,
        sample_ratio: Optional[float] = None,
    ) -> Dict:
        """
        执行 Stage 1 AutoGluon 训练
        
        Args:
            symbol: 货币对名称
            target_window: 目标时间窗口 (7m/10m/15m/30m/60m)
            data_dir: 特征存储目录
            time_limit: 训练时间预算（秒）
            presets: AutoGluon 预设 (best_quality/high_quality/medium_quality)
            eval_metric: 评估指标
            test_size: 测试集比例
            sample_ratio: 训练集降采样比例 (e.g., 0.5 = 50% of train data), None = 全量
            
        Returns:
            训练结果字典
        """
        # ---- 1. 加载数据 ----
        logger.info(f"=== Stage1: {symbol} @ {target_window} ===")
        
        store = FeatureStore(data_dir)
        data = store.load_all(symbol, mmap_mode='r')  # mmap 避免全量加载
        
        X = data['features']
        # load_labels returns keys like '15m', '7m', etc.
        y = data.get(target_window, data.get('labels'))
        if y is None:
            raise KeyError(f"Label '{target_window}' not found. Available: {[k for k in data if not k.startswith('features') and not k.startswith('timestamp') and not k.startswith('weight')]}")

        # 标签有效性处理：极少数无效行(<0.1%) → 钳位到 0，避免 X[valid] 引发 17GB 拷贝
        n_invalid = int((y < 0).sum())
        if n_invalid > 0:
            if n_invalid < len(y) * 0.001:
                y = np.clip(y, 0, 1)
                logger.info(f"标签钳位: {n_invalid}/{len(y)} 无效行 -> 0")
            else:
                valid_idx = np.where(y >= 0)[0]
                X = X[valid_idx]; y = y[valid_idx]
                if 'timestamps' in data: data['timestamps'] = data['timestamps'][valid_idx]
                logger.info(f"标签过滤: 移除 {n_invalid} 行")
        
        if 'timestamps' in data:
            timestamps = data['timestamps']
        else:
            timestamps = np.arange(len(y))
        
        logger.info(f"数据: X={X.shape}, y={y.shape}, "
                    f"正样本={y.sum()/len(y):.1%}")

        # ---- 2. 时序划分 ----
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # ---- 2.5 降采样训练集（缓解内存压力）----
        if sample_ratio is not None and sample_ratio < 1.0:
            stride = max(1, int(1.0 / sample_ratio))
            X_train = X_train[::stride]
            y_train = y_train[::stride]
            logger.info(f"降采样 (ratio={sample_ratio}, stride={stride}): "
                        f"训练集缩减为 {X_train.shape}")

        # ---- 3. 构建 DataFrame ----
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]
        train_df = pd.DataFrame(X_train, columns=feature_names)
        train_df['target'] = y_train.astype(int)
        
        test_df = pd.DataFrame(X_test, columns=feature_names)
        test_df['target'] = y_test.astype(int)

        # ---- 4. Setup MLflow ----
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        # ---- 5. AutoGluon training ----
        from autogluon.tabular import TabularPredictor

        logger.info(f"开始 AutoGluon 训练 (时间预算: {time_limit}s, 预设: {presets})")
        
        with mlflow.start_run(run_name=f"{symbol}_{target_window}") as run:
            mlflow.log_params({
                'symbol': symbol,
                'target_window': target_window,
                'n_features': X_train.shape[1],
                'n_train': len(X_train),
                'n_test': len(X_test),
                'time_limit': time_limit,
                'presets': presets,
                'sample_ratio': sample_ratio if sample_ratio else 1.0,
            })

            predictor = TabularPredictor(
                label='target',
                problem_type='binary',
                eval_metric=eval_metric,
                path=f"./autogluon_models/{symbol}_{target_window}",
            ).fit(
                train_data=train_df,
                time_limit=time_limit,
                presets=presets,
                hyperparameters={'XGB': {'ag_args_fit': {'n_jobs': 1}}},  # avoid multi-thread deadlock on Windows
                num_bag_folds=0,
                num_stack_levels=0,
            )

            # ---- 6. 评估 ----
            leaderboard = predictor.leaderboard(test_df, silent=True)
            logger.info(f"\n模型排行榜:\n{leaderboard.to_string()}")

            # 测试集评估
            y_pred_proba = predictor.predict_proba(test_df)
            if y_pred_proba.shape[1] > 1:
                y_prob = y_pred_proba.iloc[:, 1].values  # 涨概率
            else:
                y_prob = y_pred_proba.values.flatten()

            evaluator = ModelEvaluator(threshold_up=0.7, threshold_down=0.3)
            metrics = evaluator.evaluate(y_test, y_prob)
            evaluator.print_report(metrics)

            # 记录 MLflow 指标
            mlflow.log_metrics({
                'test_accuracy': metrics.get('accuracy', 0),
                'test_auc': metrics.get('auc', 0),
                'test_signal_ratio': metrics.get('signal_ratio', 0),
            })

            if 'up_accuracy' in metrics:
                mlflow.log_metric('test_up_accuracy', metrics['up_accuracy'])
            if 'down_accuracy' in metrics:
                mlflow.log_metric('test_down_accuracy', metrics['down_accuracy'])

            # ---- 7. 提取最佳模型信息 ----
            best_model_name = leaderboard.iloc[0]['model']
            best_score = leaderboard.iloc[0]['score_test']

            result = {
                'experiment_id': mlflow.active_run().info.experiment_id,
                'run_id': run.info.run_id,
                'best_model': best_model_name,
                'best_score': best_score,
                'leaderboard': leaderboard,
                'metrics': metrics,
                'feature_importance': predictor.feature_importance(test_df) if hasattr(predictor, 'feature_importance') else None,
            }

            logger.info(f"✅ Stage1 完成! 最佳模型: {best_model_name}, 分数: {best_score:.4f}")

        return result


# ---- CLI 入口 ----
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Stage1: AutoGluon 初始探索")
    parser.add_argument("--symbol", required=True, help="货币对名称 (如 EURUSD)")
    parser.add_argument("--target", default="15m", help="目标时间窗口 (7m/10m/15m/30m/60m)")
    parser.add_argument("--data-dir", default="E:/forex_npy", help="特征存储目录")
    parser.add_argument("--time-budget", type=int, default=3600, help="训练时间预算(秒)")
    parser.add_argument("--presets", default="best_quality", help="AutoGluon 预设")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000", help="MLflow 服务地址")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--sample-ratio", type=float, default=None,
                        help="训练集降采样比例 (如 0.5 = 50%%，None = 全量)")
    args = parser.parse_args()

    trainer = Stage1Trainer(
        tracking_uri=args.mlflow_uri,
    )

    result = trainer.run(
        symbol=args.symbol,
        target_window=args.target,
        data_dir=args.data_dir,
        time_limit=args.time_budget,
        presets=args.presets,
        test_size=args.test_size,
        sample_ratio=args.sample_ratio,
    )

    print(f"\n最佳模型: {result['best_model']}")
    print(f"MLflow Run ID: {result['run_id']}")
    print(f"实验已完成，可启动 mlflow ui 查看: mlflow ui --backend-store-uri http://localhost:5000")
