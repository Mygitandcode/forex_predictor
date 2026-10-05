"""
feature_pipeline.py — 批处理特征构建管线

负责从 CSV 导入的历史数据批量构建特征矩阵和标签，
输出为本地 numpy 文件，供训练层使用。

流程:
    1. CSVImporter 加载多个货币对数据
    2. FeatureBuilder.build_many() 批量构建特征
    3. 计算各时间窗口标签
    4. 计算时间衰减权重
    5. FeatureStore 保存所有数据

使用方式:
    python -m features.feature_pipeline
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.memory_monitor import MemoryMonitor

from data.csv_importer import CSVImporter
from features.feature_builder import FeatureBuilder
from features.feature_store import FeatureStore
from training.data_weights import compute_time_weights

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """批量特征构建管线"""

    def __init__(
        self,
        data_dir: str,
        store_dir: str = "./data_store",
        symbols_config: Optional[str] = None,
    ):
        """
        Args:
            data_dir: MT4 导出 CSV 数据目录
            store_dir: 特征存储目录
            symbols_config: symbols.yaml 路径
        """
        self.data_dir = Path(data_dir)
        self.store = FeatureStore(store_dir)
        self.importer = CSVImporter(data_dir)

        # 加载货币对配置
        if symbols_config:
            import yaml
            with open(symbols_config, 'r') as f:
                config = yaml.safe_load(f)
            self.symbols = [s['name'] for s in config.get('symbols', []) if s.get('enabled')]
            self.point_sizes = {s['name']: s['point_size'] for s in config.get('symbols', [])}
            self.prediction_windows = config.get('prediction_windows', [7, 10, 15, 30, 60])
            self.decay_lambda = config.get('history', {}).get('time_decay_lambda', 1.0)
        else:
            self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
                            "USDCAD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY",
                            "AUDJPY", "NZDJPY"]
            self.point_sizes = {
                "EURUSD": 0.00001, "GBPUSD": 0.00001, "USDJPY": 0.001,
                "AUDUSD": 0.00001, "NZDUSD": 0.00001, "USDCAD": 0.00001,
                "USDCHF": 0.00001, "EURGBP": 0.00001, "EURJPY": 0.001,
                "GBPJPY": 0.001,   "AUDJPY": 0.001,   "NZDJPY": 0.001,
            }
            self.prediction_windows = [7, 10, 15, 30, 60]
            self.decay_lambda = 1.0

    def run(
        self,
        symbols: Optional[List[str]] = None,
        skip_existing: bool = True,
        save_labels: bool = True,
        save_weights: bool = True,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
    ):
        """
        执行完整管线：导入 → 构建特征 → 计算标签 → 保存
        
        Args:
            symbols: 要处理的货币对列表，None = 全部
            skip_existing: 跳过已存在的特征文件
            save_labels: 是否计算并保存标签
            save_weights: 是否计算并保存时间权重
            date_start: 起始日期 (YYYY-MM-DD)，之前的数据排除
            date_end: 截止日期 (YYYY-MM-DD)，之后的数据排除
        """
        if symbols is None:
            symbols = self.symbols

        # Start memory monitor (30s interval, logs to E:\forex_data\logs)
        monitor = MemoryMonitor(
            log_dir=r"E:\forex_data\logs",
            interval=30,
            label="feature_pipeline",
        )
        monitor.start()

        results = {}
        for symbol in tqdm(symbols, desc="Processing symbols"):
            logger.info(f"=== Processing {symbol} ===")
            
            # Check existing (use .done flag for atomic integrity)
            if skip_existing and self.store.is_done(symbol):
                logger.info(f"[{symbol}] already done, skipping")
                continue

            try:
                # 1. Load data — search data_dir/{symbol}/, fallback to parent/{symbol}/
                symbol_dir = Path(self.data_dir) / symbol
                if not symbol_dir.exists():
                    symbol_dir = Path(self.data_dir).parent / symbol
                if not symbol_dir.exists():
                    symbol_dir = Path(self.data_dir)  # last resort
                importer = CSVImporter(str(symbol_dir))
                df = importer.load_symbol(symbol)
                if date_start:
                    df = df[df.index >= pd.Timestamp(date_start)]
                if date_end:
                    df = df[df.index <= pd.Timestamp(date_end)]
                logger.info(f"[{symbol}] loaded {len(df)} rows")
                
                if len(df) < 500:
                    logger.warning(f"[{symbol}] too few rows (<500), skipping")
                    continue

                monitor.log_once()  # snapshot after load

                # 2. Build features
                point_size = self.point_sizes.get(symbol, 0.00001)
                builder = FeatureBuilder(symbol=symbol, point_size=point_size)
                
                X, feature_names = builder.build_many(df)
                timestamps = df.index[200:len(X)+200].values  # align
                
                logger.info(f"[{symbol}] features: {X.shape}, dims: {len(feature_names)}")

                monitor.log_once()  # snapshot after feature build

                # 3. 保存特征
                self.store.save_features(symbol, X, timestamps)

                # 4. 计算标签
                if save_labels:
                    labels = self._compute_labels(df, len(X))
                    self.store.save_labels(symbol, labels)

                # 5. 计算时间衰减权重
                if save_weights:
                    weights = compute_time_weights(
                        timestamps, lambda_decay=self.decay_lambda
                    )
                    self.store.save_weights(symbol, weights)

                results[symbol] = {
                    'n_samples': len(X),
                    'n_features': X.shape[1],
                    'date_range': f"{timestamps[0]} ~ {timestamps[-1]}",
                }
                
                # Mark complete (after all files saved)
                self.store.mark_done(symbol)
                logger.info(f"[{symbol}] done. Features: {X.shape}")

                # Release memory before next pair
                del df, X, builder
                import gc; gc.collect()

            except FileNotFoundError as e:
                logger.warning(f"[{symbol}] data file not found: {e}")
            except Exception as e:
                logger.error(f"[{symbol}] failed: {e}", exc_info=True)

        # Snapshot & stop monitor
        monitor.log_once()
        monitor.stop()

        self._print_summary(results)
        return results

    def _compute_labels(
        self, df: pd.DataFrame, n_samples: int
    ) -> Dict[str, np.ndarray]:
        """
        计算各时间窗口的涨跌标签
        
        标签定义: 未来 close > 当前 close → 涨 (1)
                 未来 close ≤ 当前 close → 跌 (0)
        """
        closes = df['close'].values
        labels = {}
        
        for window in self.prediction_windows:
            label_arr = np.zeros(n_samples, dtype=np.int8)
            for i in range(n_samples):
                future_idx = 200 + i + window  # 200 是前面跳过的行
                if future_idx < len(closes):
                    current_c = closes[200 + i]
                    future_c = closes[future_idx]
                    label_arr[i] = 1 if future_c > current_c else 0
                else:
                    label_arr[i] = -1  # 无标签 (样本不完整)
            
            # 过滤掉无标签的样本
            labels[f"{window}m"] = label_arr

        return labels

    def _print_summary(self, results: dict):
        """打印处理摘要"""
        print("\n" + "=" * 60)
        print("特征构建管线 — 处理摘要")
        print("=" * 60)
        for symbol, info in results.items():
            print(f"  {symbol}:")
            print(f"    样本数: {info['n_samples']:,}")
            print(f"    特征维数: {info['n_features']}")
            print(f"    时间范围: {info['date_range']}")
        print("=" * 60)


# ---- CLI 入口 ----
if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="批量构建特征管线")
    parser.add_argument(
        "--data-dir", required=True,
        help="MT4 导出 CSV 数据目录"
    )
    parser.add_argument(
        "--store-dir", default="./data_store",
        help="特征存储目录"
    )
    parser.add_argument(
        "--config", default=None,
        help="symbols.yaml 配置路径"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="指定处理的货币对 (如 EURUSD GBPUSD)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重建已存在的特征"
    )
    parser.add_argument(
        "--date-start", default=None,
        help="训练数据起始日期 (YYYY-MM-DD)，之前数据排除"
    )
    parser.add_argument(
        "--date-end", default=None,
        help="训练数据截止日期 (YYYY-MM-DD)，之后数据排除"
    )
    
    args = parser.parse_args()
    
    pipeline = FeaturePipeline(
        data_dir=args.data_dir,
        store_dir=args.store_dir,
        symbols_config=args.config,
    )
    
    pipeline.run(
        symbols=args.symbols,
        skip_existing=not args.force,
        date_start=args.date_start,
        date_end=args.date_end,
    )
