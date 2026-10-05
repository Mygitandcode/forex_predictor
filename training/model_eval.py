"""
model_eval.py — 回测模拟 + 评估指标

针对金融预测场景的专用评估工具：
1. 双阈值决策评估 (p≥0.7→涨, p≤0.3→跌, 中间→震荡)
2. 方向准确率 (涨跌预测的正确率，忽略震荡区间)
3. 模拟回测收益
4. 按年/月滚动窗口评估（检测过拟合随时间变化）

使用方式:
    from training.model_eval import ModelEvaluator
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred_proba, timestamps)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    金融预测模型评估器
    
    使用双阈值决策:
    - p ≥ 0.7 → 预测涨
    - p ≤ 0.3 → 预测跌
    - 其他 → 震荡/无信号
    """
    
    def __init__(
        self,
        threshold_up: float = 0.7,
        threshold_down: float = 0.3,
    ):
        """
        Args:
            threshold_up: 涨信号阈值
            threshold_down: 跌信号阈值
        """
        self.threshold_up = threshold_up
        self.threshold_down = threshold_down

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        symbol: str = "",
    ) -> Dict:
        """
        全面评估
        
        Args:
            y_true: 真实标签 (0=跌, 1=涨)
            y_pred_proba: 预测涨概率 [0, 1]
            timestamps: 时间戳（用于时序评估）
            symbol: 货币对名称
            
        Returns:
            包含各项指标的字典
        """
        # 过滤无效标签 (-1)
        valid = y_true >= 0
        y_true = y_true[valid]
        y_pred_proba = y_pred_proba[valid]
        if timestamps is not None:
            timestamps = timestamps[valid]

        # 双阈值分类
        y_pred = np.full_like(y_pred_proba, -1, dtype=int)
        y_pred[y_pred_proba >= self.threshold_up] = 1
        y_pred[y_pred_proba <= self.threshold_down] = 0

        # 只评估有信号的样本
        has_signal = y_pred >= 0
        n_total = len(y_true)
        n_signal = has_signal.sum()

        metrics = {
            'symbol': symbol,
            'n_total': n_total,
            'n_signal': n_signal,
            'signal_ratio': n_signal / n_total if n_total > 0 else 0,
        }

        if n_signal > 0:
            # 方向准确率
            correct = (y_pred[has_signal] == y_true[has_signal]).sum()
            metrics['accuracy'] = correct / n_signal
            metrics['n_correct'] = int(correct)
            
            # 分别计算涨跌准确率
            up_signals = (y_pred == 1) & has_signal
            down_signals = (y_pred == 0) & has_signal
            
            if up_signals.sum() > 0:
                metrics['up_accuracy'] = (
                    (y_pred[up_signals] == y_true[up_signals]).sum()
                    / up_signals.sum()
                )
                metrics['n_up_signals'] = int(up_signals.sum())
            
            if down_signals.sum() > 0:
                metrics['down_accuracy'] = (
                    (y_pred[down_signals] == y_true[down_signals]).sum()
                    / down_signals.sum()
                )
                metrics['n_down_signals'] = int(down_signals.sum())

        # AUC (二分类)
        try:
            from sklearn.metrics import roc_auc_score
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except Exception:
            metrics['auc'] = 0.5

        # 按时间窗口滚动评估
        if timestamps is not None and n_total > 100:
            rolling_metrics = self._rolling_evaluate(
                y_true, y_pred_proba, timestamps
            )
            metrics['rolling'] = rolling_metrics

        return metrics

    def _rolling_evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        timestamps: np.ndarray,
        window: str = '1M',
    ) -> pd.DataFrame:
        """
        按时间滚动窗口评估，检测模型随时间衰变
        
        Returns:
            DataFrame 含每窗口的 accuracy 和 signal_ratio
        """
        df = pd.DataFrame({
            'y_true': y_true,
            'y_prob': y_pred_proba,
            'timestamp': pd.DatetimeIndex(timestamps),
        }).sort_values('timestamp')

        df.set_index('timestamp', inplace=True)
        
        records = []
        for start in pd.date_range(df.index[0], df.index[-1], freq=window):
            end = start + pd.DateOffset(months=1)
            window_df = df[start:end]
            
            if len(window_df) < 50:
                continue
            
            y_t = window_df['y_true'].values
            y_p = window_df['y_prob'].values
            
            y_pred = np.full_like(y_p, -1, dtype=int)
            y_pred[y_p >= self.threshold_up] = 1
            y_pred[y_p <= self.threshold_down] = 0
            
            has_signal = y_pred >= 0
            if has_signal.sum() > 0:
                acc = (y_pred[has_signal] == y_t[has_signal]).sum() / has_signal.sum()
            else:
                acc = np.nan
            
            records.append({
                'date': end,
                'n_samples': len(window_df),
                'n_signals': int(has_signal.sum()),
                'accuracy': acc,
                'signal_ratio': has_signal.sum() / len(window_df),
            })
        
        return pd.DataFrame(records)

    def simulate_returns(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        price_changes: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        initial_capital: float = 10000.0,
        position_size: float = 0.1,
    ) -> Dict:
        """
        模拟回测收益
        
        Args:
            y_true: 真实方向 (1=涨, 0=跌)
            y_pred_proba: 预测涨概率
            price_changes: 每笔的实际价格变化率
            timestamps: 时间戳
            initial_capital: 初始资金
            position_size: 每次仓位比例
            
        Returns:
            回测指标字典
        """
        # 过滤
        valid = y_true >= 0
        y_true = y_true[valid]
        y_pred_proba = y_pred_proba[valid]
        price_changes = price_changes[valid]
        
        # 双阈值信号
        signals = np.zeros(len(y_true))
        signals[y_pred_proba >= self.threshold_up] = 1    # 做多
        signals[y_pred_proba <= self.threshold_down] = -1  # 做空
        
        # 每日收益
        daily_returns = signals * price_changes * position_size
        
        # 累积收益
        cumulative = (1 + daily_returns).cumprod() * initial_capital
        
        # 计算指标
        total_return = (cumulative[-1] / initial_capital - 1)
        
        # 夏普比率（年化，假设1分钟K线）
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(1440 * 252) if std_ret > 0 else 0
        
        # 最大回撤
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        
        # 胜率
        n_trades = (signals != 0).sum()
        if n_trades > 0:
            win_rate = (daily_returns[signals != 0] > 0).sum() / n_trades
        else:
            win_rate = 0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': float(cumulative[-1]),
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'n_trades': int(n_trades),
            'win_rate': float(win_rate),
            'mean_return_per_trade': float(daily_returns[signals != 0].mean()) if n_trades > 0 else 0,
        }

    def print_report(self, metrics: Dict):
        """打印可读的评估报告"""
        print("\n" + "=" * 60)
        print(f"  模型评估报告 — {metrics.get('symbol', '')}")
        print("=" * 60)
        print(f"  总样本数:     {metrics.get('n_total', 0):,}")
        print(f"  有效信号:     {metrics.get('n_signal', 0):,} "
              f"({metrics.get('signal_ratio', 0):.1%})")
        print(f"  方向准确率:   {metrics.get('accuracy', 0):.4f}")
        print(f"  AUC:          {metrics.get('auc', 0):.4f}")
        
        if 'up_accuracy' in metrics:
            print(f"  ─── 涨方向 ───")
            print(f"  涨信号数:     {metrics.get('n_up_signals', 0):,}")
            print(f"  涨准确率:     {metrics.get('up_accuracy', 0):.4f}")
        
        if 'down_accuracy' in metrics:
            print(f"  ─── 跌方向 ───")
            print(f"  跌信号数:     {metrics.get('n_down_signals', 0):,}")
            print(f"  跌准确率:     {metrics.get('down_accuracy', 0):.4f}")
        print("=" * 60)
        
        if 'rolling' in metrics:
            roll = metrics['rolling']
            if isinstance(roll, pd.DataFrame) and len(roll) > 0:
                print(f"\n  滚动窗口评估 ({len(roll)} 个窗口):")
                print(f"  平均准确率:   {roll['accuracy'].mean():.4f}")
                print(f"  准确率标准差: {roll['accuracy'].std():.4f}")
                print(f"  最高准确率:   {roll['accuracy'].max():.4f}")
                print(f"  最低准确率:   {roll['accuracy'].min():.4f}")


# ---- 快速测试 ----
if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    
    # 模拟预测和真实值
    y_true = np.random.randint(0, 2, n)
    y_prob = np.random.rand(n)  # 随机预测
    
    # 让预测有一定方向性
    noise = np.random.randn(n) * 0.3
    y_prob = np.clip(0.5 + 0.2 * (y_true - 0.5) + noise, 0, 1)
    
    ts = pd.date_range('2024-01-01', periods=n, freq='1min')
    
    evaluator = ModelEvaluator(threshold_up=0.7, threshold_down=0.3)
    metrics = evaluator.evaluate(y_true, y_prob, ts.values, "EURUSD")
    evaluator.print_report(metrics)
    
    # 模拟回测
    price_changes = np.random.randn(n) * 0.0001
    backtest = evaluator.simulate_returns(y_true, y_prob, price_changes)
    print("\n回测结果:")
    for k, v in backtest.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
