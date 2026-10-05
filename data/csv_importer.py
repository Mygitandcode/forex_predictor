"""
csv_importer.py — 历史 13 年 MT4 数据导入工具

从 MT4 历史数据中心导出的 CSV 文件中导入 OHLC 数据，
清洗、标准化、按货币对组织，输出为标准 DataFrame 格式。

MT4 导出 CSV 格式 (预期):
    日期,时间,开盘,最高,最低,收盘,成交量
    2024.01.15,09:01,1.09523,1.09545,1.09510,1.09530,156

使用方式:
    from data.csv_importer import CSVImporter
    importer = CSVImporter("D:/MT4_Export/")
    df = importer.load_symbol("EURUSD")
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CSVImporter:
    """MT4 历史数据 CSV 导入器"""

    # MT4 常见列名映射
    COLUMN_MAP = {
        # 标准 MT4 导出
        'date': 'date', 'time': 'time',
        'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
        'volume': 'tick_volume',
        # 中文 MT4 导出
        '日期': 'date', '时间': 'time',
        '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close',
        '成交量': 'tick_volume',
    }

    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close']

    def __init__(self, data_dir: Union[str, Path]):
        """
        Args:
            data_dir: MT4 导出 CSV 文件所在目录
                      文件命名建议: {symbol}_M1.csv (如 EURUSD_M1.csv)
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

    def list_files(self) -> List[Path]:
        """列出目录下所有 CSV 文件"""
        return sorted(self.data_dir.glob("*.csv"))

    def load_symbol(
        self,
        symbol: str,
        timeframe: str = "M1",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        加载单个货币对的历史数据
        
        Args:
            symbol: 货币对名称，如 EURUSD
            timeframe: 时间周期，默认 M1
            date_from: 起始日期 "YYYY-MM-DD" 或 "YYYY.MM.DD"
            date_to: 结束日期
            
        Returns:
            标准化的 DataFrame，包含 datetime 索引和 OHLC 列
        """
        # 查找文件: 尝试多种命名模式
        patterns = [
            f"{symbol}_{timeframe}.csv",
            f"{symbol}{timeframe}.csv",
            f"{symbol}.csv",
            f"{symbol}_1M.csv",
            f"{symbol}1.csv",
        ]

        filepath = None
        for pattern in patterns:
            candidate = self.data_dir / pattern
            if candidate.exists():
                filepath = candidate
                break

        if filepath is None:
            # 尝试模糊匹配
            all_files = self.list_files()
            for f in all_files:
                if symbol.lower() in f.stem.lower():
                    filepath = f
                    logger.info(f"模糊匹配到文件: {filepath}")
                    break

        if filepath is None:
            raise FileNotFoundError(
                f"未找到 {symbol} 的数据文件。"
                f"请确保 {self.data_dir}/ 下存在包含 '{symbol}' 的 CSV 文件"
            )

        logger.info(f"加载 {symbol} 数据: {filepath}")
        
        # 读取 CSV
        df = self._read_mt4_csv(filepath)

        # 日期过滤
        if date_from:
            df = df[df.index >= pd.Timestamp(date_from)]
        if date_to:
            df = df[df.index <= pd.Timestamp(date_to)]

        # 排序 + 去重
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # 基本清洗
        df = self._clean_data(df)

        logger.info(
            f"{symbol} 数据加载完成: "
            f"{len(df)} 行, "
            f"{df.index[0]} ~ {df.index[-1]}"
        )
        return df

    def load_all(
        self,
        symbols: List[str],
        timeframe: str = "M1",
    ) -> Dict[str, pd.DataFrame]:
        """
        批量加载多个货币对数据
        
        Returns:
            {symbol: DataFrame} 字典
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.load_symbol(symbol, timeframe)
            except FileNotFoundError as e:
                logger.warning(f"跳过 {symbol}: {e}")
        return results

    def _read_mt4_csv(self, filepath: Path) -> pd.DataFrame:
        """
        读取 MT4 导出的 CSV 并标准化
        
        MT4 导出的常见格式:
        1. 有表头: Date,Time,Open,High,Low,Close,Volume
        2. 无表头: 2010.01.04,00:00,1.43371,1.43371,1.43371,1.43371,0
        """
        # 先探测格式
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            first_line = f.readline().strip()

        # 判断是否有表头
        has_header = any(
            kw in first_line.lower()
            for kw in ['date', 'time', 'open', '日期', '时间', '开盘']
        )

        if has_header:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        else:
            df = pd.read_csv(
                filepath,
                encoding='utf-8-sig',
                header=None,
                names=['date', 'time', 'open', 'high', 'low', 'close', 'tick_volume']
            )

        # 标准化列名
        df.columns = [c.strip().lower() for c in df.columns]
        rename_map = {}
        for col in df.columns:
            for src, tgt in self.COLUMN_MAP.items():
                if src.lower() == col:
                    rename_map[col] = tgt
                    break
        df = df.rename(columns=rename_map)

        # 构建 datetime 索引
        df = self._build_datetime_index(df)

        # 确保数值列正确
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 只保留 OHLCV 列
        keep_cols = self.REQUIRED_COLUMNS + ['tick_volume']
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]

        return df

    def _build_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """从 date + time 列构建 datetime 索引"""
        if 'date' in df.columns and 'time' in df.columns:
            # 尝试多种日期格式
            for date_fmt in ['%Y.%m.%d', '%Y-%m-%d', '%d.%m.%Y']:
                try:
                    dates = pd.to_datetime(df['date'], format=date_fmt)
                    break
                except (ValueError, TypeError):
                    continue
            else:
                dates = pd.to_datetime(df['date'])

            times = df['time'].astype(str).str.strip()
            # 时间可能是 "09:01" 或 "09:01:00"
            datetimes = []
            for d, t in zip(dates, times):
                try:
                    if len(t) <= 5:
                        t += ':00'
                    dt = pd.Timestamp(f"{d.strftime('%Y-%m-%d')} {t}")
                    datetimes.append(dt)
                except (ValueError, TypeError):
                    datetimes.append(pd.NaT)

            df['datetime'] = datetimes
            df = df.dropna(subset=['datetime'])
            df = df.set_index('datetime')
            df = df.drop(columns=['date', 'time'], errors='ignore')

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本数据清洗"""
        # 删除全 NaN 行
        df = df.dropna(subset=self.REQUIRED_COLUMNS, how='all')
        
        # 填充缺失的 OHLC（前向填充）
        df[self.REQUIRED_COLUMNS] = df[self.REQUIRED_COLUMNS].ffill()
        
        # 删除 High < Low 的异常行
        df = df[df['high'] >= df['low']]
        
        # 删除价格为 0 的行
        df = df[(df[self.REQUIRED_COLUMNS] > 0).all(axis=1)]
        
        # 确保 open/close 在 high/low 之间
        df = df[df['open'].between(df['low'], df['high'])]
        df = df[df['close'].between(df['low'], df['high'])]

        return df

    def export_parquet(self, symbol: str, output_dir: Optional[Path] = None):
        """
        将清洗后的数据导出为 Parquet 格式（压缩存储，快速加载）
        
        Args:
            symbol: 货币对名称
            output_dir: 输出目录，默认与 CSV 同目录
        """
        df = self.load_symbol(symbol)
        
        if output_dir is None:
            output_dir = self.data_dir / "processed"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{symbol}_M1.parquet"
        df.to_parquet(output_path, compression='snappy')
        logger.info(f"已导出: {output_path} ({len(df)} 行)")
        
        return output_path


# ---- 快速测试 ----
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    
    # 示例: 加载 EURUSD 数据
    # importer = CSVImporter("D:/MT4_Data/")
    # df = importer.load_symbol("EURUSD")
    # print(df.head())
    # print(df.describe())
    
    print("CSVImporter 就绪。请指定 MT4 数据目录路径后运行。")
    print("用法: CSVImporter('你的MT4数据目录路径').load_symbol('EURUSD')")
