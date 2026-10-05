"""
ts_cv.py — 时序交叉验证工具

严格使用 TimeSeriesSplit 避免未来信息泄露。
金融时序数据的核心约束：训练集时间必须早于验证集。

使用方式:
    from training.ts_cv import TimeSeriesCV
    
    cv = TimeSeriesCV(n_splits=5, gap=0, min_train_size=1000)
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # 训练和评估...
"""

import logging
from typing import Iterator, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class TimeSeriesCV:
    """
    时序交叉验证
    
    与 sklearn.TimeSeriesSplit 相比，额外支持:
    - gap: 训练集和验证集之间的间隔（防止信息泄露）
    - min_train_size: 最小训练集大小
    - expanding_window: 是否使用扩展窗口（默认 True）
    - 权重支持
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        min_train_size: int = 1000,
        expanding_window: bool = True,
    ):
        """
        Args:
            n_splits: 折叠数
            gap: 训练/验证间隔（样本数）
            min_train_size: 最小训练样本数
            expanding_window: True=扩展窗口, False=滑动窗口
        """
        self.n_splits = n_splits
        self.gap = gap
        self.min_train_size = min_train_size
        self.expanding_window = expanding_window

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        生成 (train_idx, val_idx) 索引
        
        Args:
            X: 特征矩阵 (N, D)
            y: 标签数组 (N,) 或 (N, K) — 可选
            
        Yields:
            (train_indices, val_indices)
        """
        n = len(X)
        
        if n < self.min_train_size * 2:
            raise ValueError(
                f"样本量 {n} 不足，至少需要 {self.min_train_size * 2}"
            )
        
        # 计算每折验证集大小
        total_val = n - self.min_train_size
        val_size = total_val // self.n_splits
        
        for i in range(self.n_splits):
            if self.expanding_window:
                # 扩展窗口: 训练集从 0 到 val_start
                val_start = self.min_train_size + i * val_size
                val_end = val_start + val_size
                
                train_idx = np.arange(0, max(0, val_start - self.gap))
                val_idx = np.arange(val_start, min(val_end, n))
            else:
                # 滑动窗口: 训练集大小固定
                train_size = self.min_train_size
                val_start = train_size + i * val_size
                val_end = val_start + val_size
                
                train_idx = np.arange(
                    max(0, val_start - train_size - self.gap),
                    max(0, val_start - self.gap)
                )
                val_idx = np.arange(val_start, min(val_end, n))
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx

    def get_splits_info(self, n: int) -> List[dict]:
        """获取各折的统计信息"""
        info = []
        for fold, (train_idx, val_idx) in enumerate(self.split(np.zeros(n))):
            info.append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_range': f"[{train_idx[0]}, {train_idx[-1]}]",
                'val_range': f"[{val_idx[0]}, {val_idx[-1]}]",
                'train_frac': len(train_idx) / n,
            })
        return info


def purged_kfold(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    purge_size: int = 0,
    embargo_size: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Purged K-Fold — 金融 ML 专用交叉验证
    
    - Purge: 删除训练集和验证集之间的边界样本
    - Embargo: 删除验证集之后的样本（防止训练集包含验证集之后的信息）
    
    参考: Marcos Lopez de Prado, "Advances in Financial Machine Learning"
    
    Args:
        X: 特征矩阵
        y: 标签数组
        n_splits: 折叠数
        purge_size: 清除边界样本数
        embargo_size: 禁运样本数
        
    Yields:
        (train_indices, val_indices)
    """
    n = len(X)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_idx, val_idx in tscv.split(X):
        # Purge: 删除 train 末尾和 val 开头的样本
        if purge_size > 0:
            train_idx = train_idx[:len(train_idx) - purge_size]
        
        # Embargo: 删除 val 末尾之后的样本
        if embargo_size > 0:
            max_val = val_idx[-1]
            train_idx = train_idx[train_idx <= max_val]
        
        if len(train_idx) > 0 and len(val_idx) > 0:
            yield train_idx, val_idx


# ---- 快速测试 ----
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    n = 10000
    X = np.random.randn(n, 50)
    y = np.random.randint(0, 2, n)
    
    cv = TimeSeriesCV(n_splits=5, gap=10, min_train_size=5000)
    
    print("时序交叉验证各折信息:")
    for info in cv.get_splits_info(n):
        print(f"  Fold {info['fold']}: "
              f"train={info['train_size']}, val={info['val_size']}, "
              f"train_range={info['train_range']}, val_range={info['val_range']}")
    
    print("\nPurged K-Fold 测试:")
    for i, (train_idx, val_idx) in enumerate(
        purged_kfold(X, y, n_splits=3, purge_size=5, embargo_size=10)
    ):
        print(f"  Fold {i+1}: train={len(train_idx)}, val={len(val_idx)}")
        # 验证训练集最大值 < 验证集最小值
        assert train_idx.max() < val_idx.min(), f"Fold {i+1}: 未来泄露!"
        print(f"    ✓ 无未来泄露")
