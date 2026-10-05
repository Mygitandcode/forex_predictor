"""
data_weights.py — 时间衰减权重计算

对13年历史数据按时间衰减加权，近期样本权重更高。

公式:
    w(t) = exp(-λ × (T_current - t) / T_total)

其中:
    λ = 衰减因子 (建议 0.5~2.0)
    T_current = 最新时间
    t = 样本时间
    T_total = 总时间跨度
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def compute_time_weights(
    timestamps: Union[np.ndarray, pd.DatetimeIndex],
    lambda_decay: float = 3.0,
    normalize: bool = True,
    reference_time: Optional[pd.Timestamp] = None,
) -> np.ndarray:
    """
    计算时间衰减权重
    
    Args:
        timestamps: 样本时间戳数组
        lambda_decay: 衰减因子 λ，越大近期权重越高
        normalize: 是否归一化到 sum=1
        reference_time: 参考时间（默认取最大值）
        
    Returns:
        weights: (N,) 权重数组
        
    Examples:
        >>> ts = pd.date_range('2020-01-01', periods=100, freq='D')
        >>> w = compute_time_weights(ts, lambda_decay=1.0)
        >>> print(f"最新权重: {w[-1]:.4f}, 最旧权重: {w[0]:.4f}")
    """
    # 转为 numpy datetime64
    if isinstance(timestamps, pd.DatetimeIndex):
        timestamps = timestamps.values
    
    timestamps = np.asarray(timestamps, dtype='datetime64[ns]')
    
    if reference_time is None:
        t_max = timestamps.max()
    elif isinstance(reference_time, pd.Timestamp):
        t_max = np.datetime64(reference_time)
    else:
        t_max = np.datetime64(reference_time)
    
    t_min = timestamps.min()
    
    # 总时间跨度 (秒)
    t_total = (t_max - t_min).astype('timedelta64[s]').astype(np.float64)
    
    if t_total == 0:
        return np.ones(len(timestamps)) / len(timestamps)
    
    # 每个样本的年龄 (秒)
    age_seconds = (t_max - timestamps).astype('timedelta64[s]').astype(np.float64)
    
    # 指数衰减
    weights = np.exp(-lambda_decay * age_seconds / t_total)
    
    if normalize:
        weights = weights / weights.sum()
    
    return weights


def compute_sample_weights(
    timestamps: Union[np.ndarray, pd.DatetimeIndex],
    lambda_decay: float = 3.0,
    min_weight: float = 0.01,
) -> np.ndarray:
    """
    计算可用于 sklearn 的 sample_weight
    
    与 compute_time_weights 相似，但加入了最小权重下限，
    确保所有样本至少有一定影响力。
    
    Args:
        timestamps: 时间戳
        lambda_decay: 衰减因子
        min_weight: 最小权重比例 (相对于最大值)
        
    Returns:
        sample_weights: (N,) 未归一化的样本权重
    """
    weights = compute_time_weights(timestamps, lambda_decay, normalize=False)
    
    # 设置下限
    max_w = weights.max()
    lower_bound = max_w * min_weight
    weights = np.maximum(weights, lower_bound)
    
    return weights


def get_suggested_lambda(
    n_years: float = 13.0,
    target_ratio: float = 0.1,
) -> float:
    """
    根据期望的权重比推荐 λ 值
    
    比如希望最新样本权重是最旧样本的 10 倍，
    则 target_ratio = 1/10 = 0.1
    
    w_new / w_old = exp(-λ × 0 / T_total) / exp(-λ × T_total / T_total) = exp(λ)
    → λ = -ln(target_ratio)
    
    Args:
        n_years: 历史数据年数 (仅用于日志)
        target_ratio: 最旧/最新权重比
        
    Returns:
        建议的 λ 值
    """
    lam = -np.log(target_ratio)
    return lam


# ---- 可视化 ----
def plot_weight_curve(
    timestamps,
    lambda_decay: float = 1.0,
    save_path: Optional[str] = None,
):
    """绘制时间衰减权重曲线"""
    try:
        import matplotlib.pyplot as plt
        from config.matplotlib_fonts import configure_chinese_font
        configure_chinese_font()
    except ImportError:
        print("matplotlib 未安装，无法绘图")
        return

    weights = compute_time_weights(timestamps, lambda_decay, normalize=False)
    weights = weights / weights.max()  # 归一化到 [0,1]
    
    if isinstance(timestamps, np.ndarray):
        x = pd.DatetimeIndex(timestamps)
    else:
        x = timestamps
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, weights, 'b-', linewidth=0.5, alpha=0.8)
    ax.fill_between(x, 0, weights, alpha=0.2)
    ax.set_xlabel('时间')
    ax.set_ylabel('权重 (归一化)')
    ax.set_title(f'时间衰减权重曲线 (λ={lambda_decay})')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    else:
        plt.show()


# ---- 快速测试 ----
if __name__ == "__main__":
    # 测试权重计算
    ts = pd.date_range('2010-01-01', periods=1000, freq='D')
    w = compute_time_weights(ts, lambda_decay=1.0)
    print(f"最新权重: {w[-1]:.6f}, 最旧权重: {w[0]:.6f}")
    print(f"最新/最旧比: {w[-1]/w[0]:.2f}")
    
    # 推荐 λ
    lam = get_suggested_lambda(target_ratio=0.1)
    print(f"推荐 λ (最新10倍于最旧): {lam:.2f}")
    
    # 绘制
    plot_weight_curve(ts, lambda_decay=1.0)
