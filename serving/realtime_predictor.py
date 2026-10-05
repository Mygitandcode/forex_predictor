"""
realtime_predictor.py — 实时预测主循环

while 循环从 ZeroMQ 消费 K 线数据 → 构建特征 → 模型推理 → 保存结果

这是整个系统的"心跳"——持续运行的实时预测引擎。

使用方式:
    python -m serving.realtime_predictor --symbol EURUSD

或集成模式:
    python -m serving.realtime_predictor --all
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mt4_connector import MT4Connector
from features.feature_builder import FeatureBuilder
from features.feature_store import FeatureStore

# 多时间框架合成：将 1m K 线转为未闭合高 TF bar 的累积状态
# 与 _build_features_v2.py 训练侧完全一致，确保推理输入 = 训练输入
from _build_features_v2 import make_synthetic_tf

logger = logging.getLogger(__name__)

# ---- 1060 维模型所用的五层时间框架 ----
# 每层独立调用 FeatureBuilder，产出 212 维，拼接 = 1060 维
# 含义见 docs/feature_importance_report.md §二
TF_LAYERS_1060 = [
    (1,  '1m'),     # 原始 1m 蜡烛 → 212 维 (约 105 个有效特征)
    (5,  '5min'),   # 5m 合成蜡烛 → 212 维 (约 75 个有效特征)
    (15, '15min'),  # 15m 合成蜡烛 → 212 维 (约 65 个有效特征)
    (30, '30min'),  # 30m 合成蜡烛 → 212 维 (约 45 个有效特征)
    (60, '1h'),     # 1h 合成蜡烛 → 212 维 (约 27 个有效特征)
]
# 有效特征共 317 个 (29.9%)，见 feature_importance_report.md §一


class RealtimePredictor:
    """实时预测引擎"""

    def __init__(
        self,
        symbol: str,
        model_path: Optional[str] = None,
        zmq_host: str = "127.0.0.1",
        zmq_port: int = 5555,
        output_file: Optional[str] = None,
        point_size: float = 0.00001,
    ):
        self.symbol = symbol
        self.model_path = model_path
        self.output_file = output_file or f"live_predictions_{symbol}.jsonl"
        
        # 初始化组件
        self.connector = MT4Connector(
            host=zmq_host,
            port=zmq_port,
            symbols=[symbol],
        )
        
        # ---- 特征构建器 ----
        # 旧版兼容：单层 212 维构建器（用于 212 维旧模型）
        self.feature_builder = FeatureBuilder(
            symbol=symbol,
            point_size=point_size,
        )

        # 新版 1060 维：5 个独立构建器，各自用不同 TF 输入数据
        # FeatureBuilder 的 build() 会覆盖 self.feature_names，多 builder 避免竞态
        self.tf_builders = {}
        for tf_min, tf_label in TF_LAYERS_1060:
            self.tf_builders[tf_label] = FeatureBuilder(
                symbol=symbol,
                point_size=point_size,
            )

        self.store = FeatureStore()
        
        # 加载模型
        self.model = self._load_model()

        # 模型维度检测：n_features_in_ 决定用 212 还是 1060 维构造
        self.model_dim = None  # 延迟到首个预测时检测，避免模型未加载就崩溃

        # 维度检查哨兵：首次推理前强制核对
        self._dims_checked = False
        
        # 历史缓冲区 (DataFrame)
        self.buffer: List[Dict] = []
        self.max_buffer_size = 2000  # 保留最近2000根K线（覆盖最长 MA=100 的需要）
        
        # 统计
        self.n_received = 0
        self.n_predicted = 0
        self.start_time = None

    def _load_model(self):
        """加载训练好的模型。

        加载优先级:
            1. 用户指定的 --model 路径
            2. 1060 维新模型 (EURUSD_15m_xgboost__*.pkl 中最新)
            3. 212 维旧模型 (EURUSD_15m.pkl，向后兼容)

        加载后日志打印模型维度，供操作者人工核对。
        """
        import joblib, glob as _glob

        if self.model_path and os.path.exists(self.model_path):
            model = joblib.load(self.model_path)
            dim = getattr(model, 'n_features_in_', '?')
            logger.info(f"已加载模型: {self.model_path} | 维度={dim}")
            return model

        # 默认：优先 1060 维新模型
        candidates = sorted(
            _glob.glob(f"./models/{self.symbol}_15m_xgboost__*.pkl"),
            reverse=True,  # 最新排前
        )
        candidates.append(f"./models/{self.symbol}_15m.pkl")  # 旧模型兜底

        for path in candidates:
            if os.path.exists(path):
                model = joblib.load(path)
                dim = getattr(model, 'n_features_in_', '?')
                logger.info(f"已加载模型: {path} | 维度={dim}")
                return model

        logger.warning("未找到模型文件，将跳过推理")
        return None

    def start(self):
        """启动实时预测循环"""
        self.start_time = datetime.now()
        logger.info(f"=== 实时预测引擎启动: {self.symbol} ===")
        logger.info(f"ZeroMQ: {self.connector.address}")
        logger.info(f"输出文件: {self.output_file}")
        
        try:
            with self.connector:
                for k_data in self.connector.receive_stream():
                    self._process_kline(k_data)
                    
        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            self._print_stats()

    # ================================================================
    # 维度核对 (每次启动/换模型时执行一次)
    # ================================================================
    def _check_model_dims(self, vec_length: int) -> bool:
        """核对模型期望维度与实际特征向量维度是否一致。

        这是防呆检查 —— 212 维模型不能吃 1060 维向量，反之亦然。
        模型冻结后 n_features_in_ 不变，所以只需在首次预测时检查一次。

        Returns:
            True  = 维度匹配，可以推理
            False = 维度不匹配，推理中止
        """
        if self.model is None:
            logger.warning("_check_model_dims: model is None, skip check")
            return False

        expected = self.model.n_features_in_
        if vec_length == expected:
            self._dims_checked = True
            logger.info(
                f"_check_model_dims: PASS | 特征向量={vec_length}维"
                f" == 模型期望={expected}维"
            )
            return True
        else:
            logger.error(
                f"_check_model_dims: FAIL | 特征向量={vec_length}维"
                f" != 模型期望={expected}维 | "
                f"请检查: (1) 模型文件是否正确"
                f" (2) 特征构建逻辑是否与训练时一致"
                f" (3) 是否误用了 212/1060 维的交叉组合"
            )
            return False

    # ================================================================
    # 1060 维特征构建 (5 层 TF 合成 → 拼接头)
    # 与 _build_features_v2.py 训练侧完全等价
    # ================================================================
    def _build_features_1060(self, buffer_df: pd.DataFrame) -> np.ndarray:
        """从原始 1m 缓冲区构建 1060 维特征向量。

        复现 _build_features_v2.py 的逻辑：
            1. 取 buffer_df 最后一行为"当前时刻"
            2. 对每个 TF 层:
               a. 1m 层：直接用原始 df → FeatureBuilder.build()
               b. 5m/15m/30m/1h 层：先调 make_synthetic_tf() 构造
                  未闭合 bar 的累积 OHLC，再 FeatureBuilder.build()
            3. 5 层 × 212 维 = 1060 维拼接

        每层 FeatureBuilder 的 history 窗口 = buffer 前 2000 行
        (该 TF 尺度的 bar)，与训练时 FeatureBuilder.build_many()
        中的 df.iloc[max(0,i-2000):i] 一致。

        Args:
            buffer_df: 含 open/high/low/close/tick_volume 的 1m DataFrame，
                       DatetimeIndex，最近 2000 行

        Returns:
            (1, 1060) float32 特征向量
        """
        n_rows = len(buffer_df)
        if n_rows < 200:
            raise ValueError(
                f"buffer 不足 200 行 (当前 {n_rows})，无法构建特征"
            )

        all_feats = []  # 5 段 212 维 → 最终拼 1060 维

        for tf_min, tf_label in TF_LAYERS_1060:
            if tf_min == 1:
                # 1m 层：直接用原始 1m 蜡烛 —
                # 因为当前 K 线已追加到 buffer 末尾，
                # buffer_df.iloc[-1] 就是当前时刻，buffer_df.iloc[:-1] 是历史
                df_tf = buffer_df.copy()
            else:
                # 高 TF 层：合成未闭合 bar 的累积状态
                # make_synthetic_tf 对每个 {tf_min}min 桶:
                #   open  = 桶内第 1 根 bar 的 open
                #   high  = 桶内 cummax(high)
                #   low   = 桶内 cummin(low)
                #   close = 最新 1m close
                # 与 MT4 实盘的未闭合 bar 语义完全一致
                df_tf = make_synthetic_tf(buffer_df, tf_min, None)

            # 当前快照 (该 TF 视角)
            cur_row = df_tf.iloc[-1]
            cur = {
                'open':      cur_row['open'],
                'high':      cur_row['high'],
                'low':       cur_row['low'],
                'close':     cur_row['close'],
                'timestamp': df_tf.index[-1],
            }
            # 历史窗口 (该 TF 视角的过去 2000 行)
            history = df_tf.iloc[-2001:-1] if n_rows >= 2001 else df_tf.iloc[:-1]

            builder = self.tf_builders[tf_label]
            vec = builder.build(cur, history)  # (1, 212)
            all_feats.append(vec.flatten())

        return np.array(all_feats, dtype=np.float32).reshape(1, -1)  # (1, 1060)

    # ================================================================
    # 推理主流程
    # ================================================================
    def _process_kline(self, k_data: Dict):
        """处理单条 K 线：缓冲 → 特征构建 → 模型推理 → 双阈值分类

        首次推理前自动执行维度核对 (_check_model_dims)。
        自适应选择 1060 维或 212 维特征构造路径：
            - 模型 n_features_in_ == 1060 → 5 层 TF 合成 (训练等价)
            - 模型 n_features_in_ == 212  → 单层 FeatureBuilder (旧版兼容)
        """
        self.n_received += 1
        
        symbol = k_data.get('symbol', self.symbol)
        
        # 缓冲区维护：保持最近 2000 根 K 线
        self.buffer.append(k_data)
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size:]
        
        # 预热期：至少 200 根才能计算 MA(100) 等技术指标
        if len(self.buffer) < 200:
            if self.n_received % 50 == 0:
                logger.info(f"收集历史数据中... ({len(self.buffer)}/200)")
            return

        # ---- 特征构建 ----
        try:
            buffer_df = self._buffer_to_dataframe()

            # 首次推理：检测模型维度，选择正确的构造路径
            if not self._dims_checked:
                if self.model is None:
                    # 无模型：用 212 维占位但跳过推理
                    self.model_dim = 212
                    vec = self.feature_builder.build(
                        k_data, buffer_df.iloc[:-1]
                    )
                else:
                    self.model_dim = self.model.n_features_in_
                    if self.model_dim == 1060:
                        # 核心路径：5 层 TF 合成 → 1060 维
                        # 与 _build_features_v2.py 训练侧完全一致
                        vec = self._build_features_1060(buffer_df)
                    elif self.model_dim == 212:
                        # 旧版兼容：单层 FeatureBuilder → 212 维
                        logger.info("检测到 212 维旧模型，使用单层特征构造")
                        vec = self.feature_builder.build(
                            k_data, buffer_df.iloc[:-1]
                        )
                    else:
                        raise ValueError(
                            f"不支持的模型维度: {self.model_dim}，"
                            f"仅支持 212 或 1060"
                        )

                if not self._check_model_dims(vec.shape[1]):
                    logger.error("维度核对失败，推理终止。请修正模型/特征配置后重启。")
                    return
            else:
                # 后续推理：直接走已选择的路径
                if self.model_dim == 1060:
                    vec = self._build_features_1060(buffer_df)
                else:
                    vec = self.feature_builder.build(
                        k_data, buffer_df.iloc[:-1]
                    )

        except Exception as e:
            logger.error(f"特征构建失败: {e}", exc_info=True)
            return

        # ---- 模型推理 ----
        predictions = {}
        signals = {}
        
        if self.model is not None:
            try:
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(vec)
                    if hasattr(proba, 'ndim') and proba.ndim == 2:
                        proba = proba[:, 1]
                else:
                    proba = self.model.predict(vec)
                
                proba = np.array(proba).flatten()

                # 输出目标：15m 方向概率
                # 当前模型仅训练了 15m 方向，7m/10m/30m/60m 复用同一概率
                # (多周期泛化是后续迭代目标，当前标注仅为日志可读性)
                p = float(proba[0])
                predictions["7m"] = round(p, 4)
                predictions["10m"] = round(p, 4)
                predictions["15m"] = round(p, 4)
                predictions["30m"] = round(p, 4)
                predictions["60m"] = round(p, 4)
                for w in predictions:
                    signals[w] = self._classify(p)
            except Exception as e:
                logger.error(f"推理失败: {e}", exc_info=True)
                predictions = {"15m": 0.5}
                signals = {"15m": "无信号"}
        else:
            predictions = {"15m": 0.5}
            signals = {"15m": "无模型"}

        # ---- 保存结果 ----
        record = {
            "symbol": symbol,
            "timestamp": k_data.get('timestamp', datetime.now().isoformat()),
            "open": k_data.get('open'),
            "high": k_data.get('high'),
            "low": k_data.get('low'),
            "close": k_data.get('close'),
            "predictions": predictions,
            "signals": signals,
            "predicted_at": datetime.now().isoformat(),
        }

        self._save_record(record)
        self.n_predicted += 1

        # 每 10 次预测打印一条统计
        if self.n_predicted % 10 == 0:
            sig_str = " | ".join(f"{w}:{s}" for w, s in signals.items())
            logger.info(
                f"[{symbol}] {k_data.get('timestamp', '?')} "
                f"C={k_data.get('close', 0):.5f} | {sig_str} "
                f"(已处理: {self.n_received}, 已预测: {self.n_predicted})"
            )

    def _buffer_to_dataframe(self) -> pd.DataFrame:
        """将缓冲区 dict 列表转为 DatetimeIndex 的 OHLCV DataFrame。

        make_synthetic_tf 需要 tick_volume 列做 cumsum，
        MT4 推送数据可能不含此字段 → 补默认值 1。
        """
        df = pd.DataFrame(self.buffer)
        
        # 时间索引
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        # make_synthetic_tf 需要 tick_volume 列做累积量
        if 'tick_volume' not in df.columns:
            df['tick_volume'] = 1  # 缺省：每根 K 线默认成交量为 1
        
        return df

    def _classify(self, prob: float) -> str:
        """双阈值分类"""
        if prob >= 0.7:
            return "涨"
        elif prob <= 0.3:
            return "跌"
        return "震荡"

    def _save_record(self, record: Dict):
        """追加保存到 JSONL 文件"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # 同时追加到 FeatureStore
        # 简化版: 只保存特征向量（TODO）

    def _print_stats(self):
        """打印运行统计"""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.n_received / elapsed if elapsed > 0 else 0
            print(f"\n=== 运行统计 ===")
            print(f"运行时长: {elapsed:.0f}s")
            print(f"接收K线: {self.n_received}")
            print(f"完成预测: {self.n_predicted}")
            print(f"接收速率: {rate:.1f} K线/秒")


# ---- CLI ----
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="实时预测引擎")
    parser.add_argument("--symbol", default="EURUSD", help="货币对名称")
    parser.add_argument("--model", default=None, help="模型文件路径")
    parser.add_argument("--zmq-host", default="127.0.0.1", help="ZeroMQ 主机")
    parser.add_argument("--zmq-port", type=int, default=5555, help="ZeroMQ 端口")
    parser.add_argument("--output", default=None, help="预测输出文件")
    parser.add_argument("--all", action="store_true", help="处理所有货币对")
    args = parser.parse_args()

    if args.all:
        # 启动多个预测器（后期可用多线程/多进程）
        print("多货币对模式尚未实现，请使用 --symbol 指定单个货币对")
    else:
        predictor = RealtimePredictor(
            symbol=args.symbol,
            model_path=args.model,
            zmq_host=args.zmq_host,
            zmq_port=args.zmq_port,
            output_file=args.output,
        )
        predictor.start()
