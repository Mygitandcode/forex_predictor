"""
feature_store.py — 特征存储层，基于 numpy 数组 + 本地文件

单人使用场景，无需外部中间件，直接读写 .npy 文件。
提供按货币对 + 时间窗口的读写接口。

存储结构:
    data_root/
        {symbol}/
            features.npy       # (N, D) 特征矩阵
            timestamps.npy     # (N,)  时间戳
            labels_7m.npy      # (N,)  7分钟标签
            labels_10m.npy     # (N,)  10分钟标签
            labels_15m.npy     # (N,)  15分钟标签
            labels_30m.npy     # (N,)  30分钟标签
            labels_60m.npy     # (N,)  60分钟标签
            weights.npy        # (N,)  时间衰减权重
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FeatureStore:
    """本地 numpy 文件特征存储"""

    def __init__(self, data_root: str = "E:/forex_npy"):
        """
        Args:
            data_root: 数据存储根目录
        """
        self.data_root = Path(data_root)

    def symbol_dir(self, symbol: str) -> Path:
        """获取某货币对的存储目录"""
        d = self.data_root / symbol
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- 写入 ----

    def save_features(
        self,
        symbol: str,
        features: np.ndarray,
        timestamps: np.ndarray,
    ):
        """
        保存特征矩阵和时间戳
        
        Args:
            symbol: 货币对名称
            features: (N, D) 特征矩阵
            timestamps: (N,) 时间戳数组 (Unix 时间戳或 datetime64)
        """
        d = self.symbol_dir(symbol)
        # Atomic write: save to .tmp first, then replace (avoid corrupt files on crash)
        for name, arr in [("features.npy", features), ("timestamps.npy", timestamps)]:
            tmp_path = d / f"tmp.{name}"
            np.save(tmp_path, arr)
            tmp_path.replace(d / name)  # replace overwrites existing
        logger.info(f"[{symbol}] saved features: {features.shape}, timestamps: {len(timestamps)}")

    def save_labels(
        self,
        symbol: str,
        labels: Dict[str, np.ndarray],
    ):
        """Save label arrays per time window (atomic write)."""
        d = self.symbol_dir(symbol)
        for window, arr in labels.items():
            tmp = d / f"tmp.labels_{window}.npy"
            np.save(tmp, arr)
            tmp.replace(d / f"labels_{window}.npy")
        logger.info(f"[{symbol}] saved labels: {list(labels.keys())}")

    def save_weights(self, symbol: str, weights: np.ndarray):
        """Save time-decay weights (atomic write)."""
        d = self.symbol_dir(symbol)
        tmp = d / "tmp.weights.npy"
        np.save(tmp, weights)
        tmp.replace(d / "weights.npy")
        logger.info(f"[{symbol}] saved weights: {len(weights)}")

    def mark_done(self, symbol: str):
        """Write .done flag after all files saved successfully."""
        (self.symbol_dir(symbol) / ".done").touch()

    def is_done(self, symbol: str) -> bool:
        """Check if feature pipeline completed for this symbol."""
        return (self.symbol_dir(symbol) / ".done").exists()

    def save_all(
        self,
        symbol: str,
        features: np.ndarray,
        timestamps: np.ndarray,
        labels: Dict[str, np.ndarray],
        weights: Optional[np.ndarray] = None,
    ):
        """一次性保存全部数据"""
        self.save_features(symbol, features, timestamps)
        self.save_labels(symbol, labels)
        if weights is not None:
            self.save_weights(symbol, weights)

    # ---- 读取 ----

    def load_features(
        self, symbol: str, mmap_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载特征和时间戳
        
        Args:
            symbol: 货币对名称
            mmap_mode: numpy mmap 模式 (None/'r'/'r+' 等)，训练时设为 'r' 避免全量加载
        
        Returns:
            (features (N,D), timestamps (N,))
        """
        d = self.symbol_dir(symbol)
        f_path = d / "features.npy"
        t_path = d / "timestamps.npy"
        if not f_path.exists():
            raise FileNotFoundError(f"特征文件不存在: {f_path}")
        return np.load(f_path, mmap_mode=mmap_mode), np.load(t_path)

    def load_labels(
        self, symbol: str, windows: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        加载标签
        
        Args:
            windows: 要加载的时间窗口列表，默认全部
        """
        if windows is None:
            windows = ["7m", "10m", "15m", "30m", "60m"]
        
        d = self.symbol_dir(symbol)
        labels = {}
        for w in windows:
            p = d / f"labels_{w}.npy"
            if p.exists():
                labels[w] = np.load(p)
            else:
                logger.warning(f"标签文件不存在: {p}")
        return labels

    def load_weights(self, symbol: str) -> Optional[np.ndarray]:
        """加载时间衰减权重"""
        p = self.symbol_dir(symbol) / "weights.npy"
        if p.exists():
            return np.load(p)
        return None

    def load_all(
        self, symbol: str, mmap_mode: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """加载全部数据
        
        Args:
            mmap_mode: 'r' for read-only mmap (saves ~16 GB RAM for large features)
        """
        features, timestamps = self.load_features(symbol, mmap_mode=mmap_mode)
        data = {
            "features": features,
            "timestamps": timestamps,
        }
        data.update(self.load_labels(symbol))
        weights = self.load_weights(symbol)
        if weights is not None:
            data["weights"] = weights
        return data

    # ---- 增量追加 ----

    def append_features(
        self,
        symbol: str,
        new_features: np.ndarray,
        new_timestamps: np.ndarray,
    ):
        """追加新特征行"""
        d = self.symbol_dir(symbol)
        f_path = d / "features.npy"
        t_path = d / "timestamps.npy"

        if f_path.exists():
            existing_f = np.load(f_path)
            existing_t = np.load(t_path)
            features = np.vstack([existing_f, new_features])
            timestamps = np.concatenate([existing_t, new_timestamps])
        else:
            features = new_features
            timestamps = new_timestamps

        self.save_features(symbol, features, timestamps)


# ---- 快速测试 ----
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    store = FeatureStore("./test_data_store")
    
    # 模拟数据
    n, d = 100, 50
    features = np.random.randn(n, d)
    timestamps = np.arange(n)
    labels = {
        "7m": np.random.randint(0, 2, n),
        "15m": np.random.randint(0, 2, n),
        "30m": np.random.randint(0, 2, n),
    }
    
    store.save_all("EURUSD", features, timestamps, labels)
    
    loaded = store.load_all("EURUSD")
    print(f"特征形状: {loaded['features'].shape}")
    print(f"标签: {list(loaded.keys())}")
    print("FeatureStore 测试通过")
