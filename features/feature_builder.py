"""
feature_builder.py — 超细向量构建（100~300维）

根据蓝图第 5 节设计，对每根 1分钟 K 线构建极度细致的特征向量。

特征类别（共7类，目标 100~300维）:
    C1: 当前K线本体特征 (~8维)
    C2: 影线/实体细分编码 (~12维)
    C3: 多周期均线关系 (~20~60维)
    C4: 多时间框架表现 (~32维)
    C5: 形态识别 (~10维)
    C6: 技术指标 ta-lib (~20维)
    C7: 波动率与时序特征 (~10维)

使用方式:
    from features.feature_builder import FeatureBuilder
    builder = FeatureBuilder(symbol="EURUSD")
    
    # 单条构建
    vec = builder.build(current_k, history_buffer)
    
    # 批量构建
    X, feature_names = builder.build_from_dataframe(df)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 尝试导入 ta-lib（可选，Phase 3 才需要）
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logger.warning("ta-lib 未安装，技术指标特征将被跳过。pip install TA-Lib")


class FeatureBuilder:
    """超细向量特征构建器"""
    
    # ---- 常数 ----
    MA_PERIODS = [5, 10, 20, 60, 100]          # 均线周期
    TIME_FRAMES = ['5min', '15min', '30min', '1h']  # 多时间框架
    
    def __init__(
        self,
        symbol: str = "EURUSD",
        point_size: float = 0.00001,
        include_talib: bool = True,
    ):
        """
        Args:
            symbol: 货币对名称（用于获取 point_size）
            point_size: 最小价格单位
            include_talib: 是否包含 ta-lib 技术指标特征
        """
        self.symbol = symbol
        self.point_size = point_size
        self.include_talib = include_talib and HAS_TALIB
        
        # 特征名列表（构建后填充）
        self.feature_names: List[str] = []
    
    def build(
        self,
        current_k: Dict,
        history_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        构建单条超细向量
        
        Args:
            current_k: 当前 K 线数据 {'open','high','low','close','timestamp'}
            history_df: 历史 K 线 DataFrame，按时间排序，至少含 100 根
            
        Returns:
            (1, D) 特征向量
        """
        o, h, l, c = current_k['open'], current_k['high'], current_k['low'], current_k['close']
        
        features = []
        names = []
        
        # ==== C1: 本体特征 (~8维) ====
        f1, n1 = self._build_body_features(o, h, l, c)
        features.extend(f1); names.extend(n1)
        
        # ==== C2: 影线/实体细分编码 (~12维) ====
        f2, n2 = self._build_shadow_body_detail(o, h, l, c)
        features.extend(f2); names.extend(n2)
        
        # ==== C1b: 10根K线累积实体动量 ====
        f1b, n1b = self._build_body_momentum(history_df)
        features.extend(f1b); names.extend(n1b)
        
        # ==== C3: 多周期均线关系 (20~60维) ====
        f3, n3 = self._build_ma_features(o, h, l, c, history_df)
        features.extend(f3); names.extend(n3)
        
        # ==== C3b: 均线态势（排列有序度 + 价格位置）====
        f3b, n3b = self._build_ma_posture(c, history_df)
        features.extend(f3b); names.extend(n3b)
        
        # ==== C4: 多时间框架表现 (~32维) ====
        f4, n4 = self._build_timeframe_features(o, h, l, c, history_df)
        features.extend(f4); names.extend(n4)
        
        # ==== C5: 形态识别 (~10维) ====
        f5, n5 = self._build_pattern_features(o, h, l, c, history_df)
        features.extend(f5); names.extend(n5)
        
        # ==== C5b: 非标准头肩顶底 (MACD交叉驱动) ====
        f5b, n5b = self._build_hs_macd(history_df)
        features.extend(f5b); names.extend(n5b)
        
        # ==== C6: 技术指标 (~20维) ====
        if self.include_talib:
            f6, n6 = self._build_talib_features(history_df)
            features.extend(f6); names.extend(n6)
        
        # ==== C6b: MACD交叉距离 ====
        f6b, n6b = self._build_macd_cross_features(o, h, l, c, history_df)
        features.extend(f6b); names.extend(n6b)
        
        # ==== C7: 波动率+时序 (~10维) ====
        f7, n7 = self._build_volatility_features(c, current_k.get('timestamp'), history_df)
        features.extend(f7); names.extend(n7)
        
        self.feature_names = names
        return np.array(features, dtype=np.float32).reshape(1, -1)
    
    def build_many(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        批量构建特征矩阵
        
        Args:
            df: 按时间排序的 K 线 DataFrame
                必须包含: open, high, low, close, timestamp
            
        Returns:
            (X (N,D), feature_names)
        """
        n = len(df)
        all_features = []
        
        for i in range(n):
            if i < 200:  # 前面数据不足，跳过
                continue
            
            current_k = {
                'open': df['open'].iloc[i],
                'high': df['high'].iloc[i],
                'low': df['low'].iloc[i],
                'close': df['close'].iloc[i],
                'timestamp': df.index[i],
            }
            history_df = df.iloc[max(0, i-2000):i].copy()
            
            vec = self.build(current_k, history_df)
            all_features.append(vec.flatten())
        
        X = np.array(all_features, dtype=np.float32)
        logger.info(f"批量构建特征完成: {X.shape}")
        return X, self.feature_names
    
    # ============================================================
    # 私有方法: 各类特征构建
    # ============================================================
    
    def _build_body_features(
        self, o: float, h: float, l: float, c: float
    ) -> Tuple[List[float], List[str]]:
        """C1: 当前K线本体特征"""
        features = [
            h, l, c,
            h - max(o, c),           # upper_shadow
            min(o, c) - l,           # lower_shadow
            abs(c - o),              # body
            (c - l) / (h - l + 1e-9), # body_position
        ]
        names = ['high', 'low', 'close',
                 'upper_shadow', 'lower_shadow', 'body', 'body_position']
        return features, names
    
    def _build_body_momentum(
        self, df: pd.DataFrame
    ) -> Tuple[List[float], List[str]]:
        """C1b: 10根K线累积实体动量
        
        当前K线 + 前9根，共10根，每根实体 = |close-open| (点数):
        - 实体极差: max(10根实体) - min(10根实体) vs 各周期阈值
        - 实体总和: 10根实体的点数之和 vs 各周期阈值
        """
        N = 10
        if len(df) < N:
            return [0.0] * 54, ['body_mom_placeholder'] * 54
        
        recent = df.iloc[-N:]
        bodies = np.abs(recent['close'].values - recent['open'].values)
        ps = self.point_size if self.point_size > 0 else 0.00001
        
        # 实体点数数组
        body_pts = bodies / ps  # 1.17438-1.17435=0.00003 → 3点
        
        # 极差: 10根中最大实体 - 最小实体
        body_range = float(np.max(body_pts) - np.min(body_pts))
        # 总和: 10根实体点数相加
        body_sum = float(np.sum(body_pts))
        
        # 极差阈值 (实体最大-最小)
        thresholds_range = {
            '1m':  [10, 20, 30, 40, 50],
            '5m':  [20, 40, 60, 80],
            '15m': [30, 50, 70, 90, 110],
            '30m': [30, 50, 70, 90, 110, 150],
            '1h':  [50, 100, 150, 200, 250],
        }
        # 总和阈值 (10根实体相加)
        thresholds_sum = {
            '1m':  [20, 30, 40, 50],
            '5m':  [40, 60, 80, 100, 120],
            '15m': [30, 60, 90, 120, 150],
            '30m': [40, 80, 120, 160, 200, 250, 300],
            '1h':  [50, 100, 150, 200, 250, 300, 350, 400],
        }
        
        features = []
        names = []
        
        for tf_name in ['1m', '5m', '15m', '30m', '1h']:
            for t in thresholds_range[tf_name]:
                features.append(1.0 if body_range > t else 0.0)
                names.append(f'body_range_{tf_name}_gt{t}')
            for t in thresholds_sum[tf_name]:
                features.append(1.0 if body_sum > t else 0.0)
                names.append(f'body_sum_{tf_name}_gt{t}')
        
        return features, names
    
    def _build_shadow_body_detail(
        self, o: float, h: float, l: float, c: float
    ) -> Tuple[List[float], List[str]]:
        """C2: 影线/实体细分编码"""
        ps = self.point_size if self.point_size > 0 else 0.00001
        
        upper_s = h - max(o, c)
        lower_s = min(o, c) - l
        body_size = abs(c - o)
        
        # 上影线涉及点数 (分桶)
        upper_points = int(upper_s / ps) if ps > 0 else 0
        lower_points = int(lower_s / ps) if ps > 0 else 0
        body_points = int(body_size / ps) if ps > 0 else 0
        
        # 上下影线比例关系
        shadow_ratio = upper_s / (lower_s + 1e-9)
        body_shadow_ratio = body_size / (upper_s + lower_s + 1e-9)
        
        # 蜡烛是阳线还是阴线
        is_bullish = 1.0 if c >= o else 0.0
        
        features = [
            float(upper_points),
            float(lower_points),
            float(body_points),
            upper_points // 3,     # 分桶编码
            lower_points // 3,
            body_points // 3,
            shadow_ratio,
            body_shadow_ratio,
            is_bullish,
            upper_s / (h - l + 1e-9),    # 上影线占比
            lower_s / (h - l + 1e-9),    # 下影线占比
            body_size / (h - l + 1e-9),  # 实体占比
        ]
        names = [
            'upper_points', 'lower_points', 'body_points',
            'upper_bucket', 'lower_bucket', 'body_bucket',
            'shadow_ratio', 'body_shadow_ratio', 'is_bullish',
            'upper_shadow_pct', 'lower_shadow_pct', 'body_pct',
        ]
        return features, names
    
    def _build_ma_features(
        self, o: float, h: float, l: float, c: float, df: pd.DataFrame
    ) -> Tuple[List[float], List[str]]:
        """C3: 多周期均线关系"""
        features = []
        names = []
        closes = df['close'].values
        
        if len(closes) < max(self.MA_PERIODS):
            return [0.0] * (4 * len(self.MA_PERIODS) * 2), ['ma_placeholder'] * (4 * len(self.MA_PERIODS) * 2)
        
        for p in self.MA_PERIODS:
            ma = np.mean(closes[-p:])
            for i, (price, price_name) in enumerate(
                [(o, 'open'), (h, 'high'), (l, 'low'), (c, 'close')]
            ):
                # 差值
                diff = (price - ma) / ma  # 归一化差值
                features.append(diff)
                names.append(f'{price_name}_ma{p}_diff')
                
                # 比值
                ratio = price / ma if ma > 0 else 1.0
                features.append(ratio)
                names.append(f'{price_name}_ma{p}_ratio')
        
        return features, names
    
    def _build_ma_posture(
        self, c: float, df: pd.DataFrame
    ) -> Tuple[List[float], List[str]]:
        """C3b: 均线态势（排列有序度 + 价格在多空中的位置）
        
        5个时间段各2维:
          - MA排列有序度: 6条MA(5/10/20/40/60/100)多头排列程度 0~1
          - 价格位置: (上方最近MA距离-下方最近MA距离)/(上方+下方距离)
        """
        MA_PERIODS = [5, 10, 20, 40, 60, 100]
        closes = df['close'].values
        
        def _compute(cl):
            n = len(cl)
            if n < max(MA_PERIODS):
                return 1.0, 0.0
            mas = [float(np.mean(cl[-p:])) for p in MA_PERIODS]
            
            # Scheme 2: MA ordering 0=perfect, 1=fully inverted
            violations = sum(1 for i in range(len(mas)-1) if mas[i] < mas[i+1])
            order = violations / (len(mas) - 1)  # 0~1
            
            # Scheme 4: price position between nearest MA above & below
            above = [ma - c for ma in mas if ma > c]
            below = [c - ma for ma in mas if ma < c]
            if above and below:
                up = min(above)
                dn = min(below)
                posture = (dn - up) / (dn + up + 1e-9)  # -1~+1
            elif above:
                posture = -1.0  # all MAs above
            elif below:
                posture = 1.0   # all MAs below
            else:
                posture = 0.0
            
            return order, posture
        
        features = []
        names = []
        
        o, p = _compute(closes)
        features.extend([o, p])
        names.extend(['ma_order_1m', 'ma_posture_1m'])
        
        for tf_name, tf_min in [('5m',5),('15m',15),('30m',30),('1h',60)]:
            n = len(closes) // tf_min
            if n < max(MA_PERIODS):
                features.extend([1.0, 0.0])
            else:
                tf_c = closes[-n*tf_min:][tf_min-1::tf_min]
                o, p = _compute(tf_c)
                features.extend([o, p])
            names.extend([f'ma_order_{tf_name}', f'ma_posture_{tf_name}'])
        
        return features, names
    
    def _build_timeframe_features(
        self, o: float, h: float, l: float, c: float, df: pd.DataFrame
    ) -> Tuple[List[float], List[str]]:
        """
        C4: 多时间框架表现
        
        当前价格在不同周期图表中的位置/趋势
        """
        features = []
        names = []
        
        tf_map = {'5min': 5, '15min': 15, '30min': 30, '1h': 60}
        
        for tf_name, tf_minutes in tf_map.items():
            if len(df) < tf_minutes * 2:
                continue
            
            # 取最近 tf_minutes 根 K 线的聚合
            recent = df.iloc[-tf_minutes:]
            tf_open = recent['open'].iloc[0]
            tf_high = recent['high'].max()
            tf_low = recent['low'].min()
            tf_close = recent['close'].iloc[-1]
            tf_range = tf_high - tf_low
            
            # 当前价格在时间框架中的位置
            for price, pname in [(o, 'open'), (h, 'high'), (l, 'low'), (c, 'close')]:
                # 价格在框架高低范围内的位置
                pos = (price - tf_low) / (tf_range + 1e-9)
                features.append(pos)
                names.append(f'{pname}_{tf_name}_position')
            
            # 框架趋势: 框架收盘 vs 框架开盘
            tf_trend = (tf_close - tf_open) / (tf_open + 1e-9)
            features.append(tf_trend)
            names.append(f'{tf_name}_trend')
        
        return features, names
    
    def _build_pattern_features(
        self, o: float, h: float, l: float, c: float, df: pd.DataFrame
    ) -> Tuple[List[float], List[str]]:
        """C5: 形态识别 (规则引擎)"""
        features = []
        names = []
        
        body_size = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l
        
        # 十字星: 实体极小
        doji = 1.0 if body_size / (total_range + 1e-9) < 0.1 else 0.0
        features.append(doji)
        names.append('pattern_doji')
        
        # 锤子线: 下影线 >> 实体，上影线很小
        hammer = 1.0 if (
            lower_shadow > 2 * body_size
            and upper_shadow < 0.3 * body_size
            and total_range > 0
        ) else 0.0
        features.append(hammer)
        names.append('pattern_hammer')
        
        # 吞没形态检测（简化版）
        engulfing = self._detect_engulfing(df)
        features.append(engulfing)
        names.append('pattern_engulfing')
        
        # 孕线检测
        harami = self._detect_harami(df)
        features.append(harami)
        names.append('pattern_harami')
        
        # 双顶双底简化检测
        double_top, double_bottom = self._detect_double_tb(df)
        features.extend([double_top, double_bottom])
        names.extend(['pattern_double_top', 'pattern_double_bottom'])
        
        # 抽头缩脚 (影线暴露意图)
        chop_head = 1.0 if upper_shadow > 2 * body_size and body_size > 0 else 0.0
        chop_foot = 1.0 if lower_shadow > 2 * body_size and body_size > 0 else 0.0
        features.extend([chop_head, chop_foot])
        names.extend(['pattern_chop_head', 'pattern_chop_foot'])
        
        return features, names
    
    def _detect_engulfing(self, df: pd.DataFrame) -> float:
        """检测吞没形态"""
        if len(df) < 3:
            return 0.0
        prev_o, prev_c = df['open'].iloc[-2], df['close'].iloc[-2]
        curr_o, curr_c = df['open'].iloc[-1], df['close'].iloc[-1]
        
        # 看涨吞没: 前阴后阳，后实体包前实体
        if (prev_c < prev_o and curr_c > curr_o
            and curr_o <= prev_c and curr_c >= prev_o):
            return 1.0
        # 看跌吞没: 前阳后阴
        if (prev_c > prev_o and curr_c < curr_o
            and curr_o >= prev_c and curr_c <= prev_o):
            return -1.0
        return 0.0
    
    def _detect_harami(self, df: pd.DataFrame) -> float:
        """检测孕线形态"""
        if len(df) < 3:
            return 0.0
        prev_o, prev_c = df['open'].iloc[-2], df['close'].iloc[-2]
        curr_o, curr_c = df['open'].iloc[-1], df['close'].iloc[-1]
        
        prev_range = abs(prev_c - prev_o)
        curr_range = abs(curr_c - curr_o)
        
        if prev_range > 0 and curr_range < 0.5 * prev_range:
            if curr_o > min(prev_o, prev_c) and curr_c < max(prev_o, prev_c):
                return 1.0
        return 0.0
    
    def _detect_double_tb(self, df: pd.DataFrame) -> Tuple[float, float]:
        """检测双顶双底（简化版）"""
        if len(df) < 30:
            return 0.0, 0.0
        
        highs = df['high'].values[-30:]
        lows = df['low'].values[-30:]
        
        # 找局部极值
        dt = 0.0
        db = 0.0
        
        # 双顶: 两个相近的局部高点
        peaks = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            if abs(p1[1] - p2[1]) / (p1[1] + 1e-9) < 0.02:  # 高度差 < 2%
                dt = 1.0
        
        # 双底
        troughs = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                troughs.append((i, lows[i]))
        
        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            if abs(t1[1] - t2[1]) / (t1[1] + 1e-9) < 0.02:
                db = 1.0
        
        return dt, db
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> Tuple[float, float]:
        """检测头肩顶底（简化版）"""
        if len(df) < 60:
            return 0.0, 0.0
        
        highs = df['high'].values[-60:]
        lows = df['low'].values[-60:]
        
        # 头肩顶: 左肩高 → 头更高 → 右肩约等于左肩
        # 简化: 找三个局部高点
        peaks = []
        for i in range(3, len(highs) - 3):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i-3] \
               and highs[i] > highs[i+1] and highs[i] > highs[i+2] and highs[i] > highs[i+3]:
                peaks.append((i, highs[i]))
        
        hs_top = 0.0
        if len(peaks) >= 3:
            p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
            if p2[1] > p1[1] and p2[1] > p3[1] and abs(p1[1] - p3[1]) / (p1[1] + 1e-9) < 0.05:
                hs_top = 1.0
        
        hs_bottom = 0.0
        return hs_top, hs_bottom
    
    def _build_hs_macd(
        self, df: pd.DataFrame
    ) -> Tuple[List[float], List[str]]:
        """C5b: 非标准头肩顶底 (MACD 交叉驱动)
        
        40根K线内模式匹配：
          头肩顶: 死→金→死, 距最后死叉≤25根 → -45→-44→...→0
          头肩底: 金→死→金, 距最后金叉≤25根 →  45→ 44→...→0
          无模式: -99
        """
        def _ema(arr, span):
            alpha = 2.0 / (span + 1)
            r = np.zeros_like(arr, dtype=np.float64)
            r[0] = float(arr[0])
            for i in range(1, len(arr)):
                r[i] = alpha * float(arr[i]) + (1 - alpha) * r[i-1]
            return r
        
        def _hs_value(closes):
            n = len(closes)
            if n < 40:
                return -99.0, -99.0
            ema12 = _ema(closes, 12)
            ema26 = _ema(closes, 26)
            dif = ema12 - ema26
            dea = _ema(dif, 9)
            macd_diff = dif - dea
            
            # Cross events in last 40 bars: (bars_ago, type: 1=golden, -1=dead)
            lookback = min(40, n - 1)
            cross_events = []
            for i in range(n - lookback, n):
                prev = macd_diff[i-1]
                curr = macd_diff[i]
                if prev < 0 and curr > 0:
                    cross_events.append((n - 1 - i, 1))
                elif prev > 0 and curr < 0:
                    cross_events.append((n - 1 - i, -1))
            
            def _eval_pattern(seq_types, max_dist, base):
                """Find pattern seq in events, return countdown value"""
                target = tuple(seq_types)
                chrono = tuple(e[1] for e in reversed(cross_events))
                for i in range(len(chrono) - len(target) + 1):
                    if chrono[i:i+len(target)] == target:
                        last = cross_events[-(i+len(target))]
                        if last[0] <= max_dist:
                            # base + bars_ago, clamp toward 0
                            val = max(base, -base)  # abs(base)
                            raw = val - last[0]
                            if base < 0:
                                return float(min(raw, 0))   # -45→...→0
                            else:
                                return float(max(raw, 0))   # 45→...→0
                return -99.0
            
            top_val = _eval_pattern([-1, 1, -1], 25, -45)  # dead→golden→dead
            bot_val = _eval_pattern([1, -1, 1], 25, 45)    # golden→dead→golden
            return float(top_val), float(bot_val)
        
        features = []
        names = []
        closes = df['close'].values
        
        t, b = _hs_value(closes)
        features.extend([t, b])
        names.extend(['hs_top_1m', 'hs_bot_1m'])
        
        for tf_name, tf_min in [('5m',5),('15m',15),('30m',30),('1h',60)]:
            n = len(closes) // tf_min
            if n < 40:
                features.extend([-99.0, -99.0])
            else:
                tf_closes = closes[-n * tf_min:][tf_min - 1::tf_min]
                t, b = _hs_value(tf_closes)
                features.extend([t, b])
            names.extend([f'hs_top_{tf_name}', f'hs_bot_{tf_name}'])
        
        return features, names
    
    def _build_talib_features(
        self, df: pd.DataFrame
    ) -> Tuple[List[float], List[str]]:
        """C6: ta-lib 技术指标特征"""
        features = []
        names = []
        
        if not HAS_TALIB:
            return features, names
        
        closes = df['close'].values.astype(np.float64)
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        
        if len(closes) < 30:
            return [0.0] * 12, ['talib_placeholder'] * 12
        
        # RSI(14)
        rsi = talib.RSI(closes, timeperiod=14)
        features.append(rsi[-1] if not np.isnan(rsi[-1]) else 50.0)
        names.append('rsi_14')
        
        # MACD
        macd, signal, hist = talib.MACD(closes)
        features.extend([
            macd[-1] if not np.isnan(macd[-1]) else 0.0,
            signal[-1] if not np.isnan(signal[-1]) else 0.0,
            hist[-1] if not np.isnan(hist[-1]) else 0.0,
        ])
        names.extend(['macd', 'macd_signal', 'macd_hist'])
        
        # 布林带
        upper, middle, lower = talib.BBANDS(closes)
        bb_position = (closes[-1] - lower[-1]) / (upper[-1] - lower[-1] + 1e-9)
        bb_width = (upper[-1] - lower[-1]) / (middle[-1] + 1e-9)
        features.extend([bb_position, bb_width])
        names.extend(['bb_position', 'bb_width'])
        
        # ATR(14)
        atr = talib.ATR(highs, lows, closes, timeperiod=14)
        features.append(atr[-1] if not np.isnan(atr[-1]) else 0.0)
        names.append('atr_14')
        
        # CCI(20)
        cci = talib.CCI(highs, lows, closes, timeperiod=20)
        features.append(cci[-1] if not np.isnan(cci[-1]) else 0.0)
        names.append('cci_20')
        
        # Stochastic
        slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
        features.extend([
            slowk[-1] if not np.isnan(slowk[-1]) else 50.0,
            slowd[-1] if not np.isnan(slowd[-1]) else 50.0,
        ])
        names.extend(['stoch_k', 'stoch_d'])
        
        return features, names
    
    def _build_macd_cross_features(
        self, o: float, h: float, l: float, c: float, df: pd.DataFrame
    ) -> Tuple[List[float], List[str]]:
        """C6b: MACD金叉/死叉距离
        
        1min: 距离0~10(金) / -1~-10(死) / 99(无)
        5min/15min/30min/1h: 同上但距离//2*2分组 (0,1→0; 2,3→2)
        """
        closes = df['close'].values
        
        def _ema(arr, span):
            alpha = 2.0 / (span + 1)
            result = np.zeros_like(arr, dtype=np.float64)
            result[0] = float(arr[0])
            for i in range(1, len(arr)):
                result[i] = alpha * float(arr[i]) + (1 - alpha) * result[i-1]
            return result
        
        def _cross_dist(cl):
            """Find MACD cross: +N=golden N bars ago, -N=dead N bars ago, 99=none"""
            if len(cl) < 35:
                return 99.0
            ema12 = _ema(cl, 12)
            ema26 = _ema(cl, 26)
            dif = ema12 - ema26
            dea = _ema(dif, 9)
            for dist in range(1, min(11, len(dif))):
                prev = dif[-dist-1] - dea[-dist-1]
                curr = dif[-dist] - dea[-dist]
                if prev < 0 and curr > 0:
                    return float(dist - 1)    # golden cross
                if prev > 0 and curr < 0:
                    return -float(dist - 1)   # dead cross
            return 99.0
        
        features = []
        names = []
        
        # 1min
        features.append(_cross_dist(closes))
        names.append('macd_cross_1m')
        
        # Higher timeframes: aggregate 1min closes
        for tf_name, tf_min in [('5m', 5), ('15m', 15), ('30m', 30), ('1h', 60)]:
            n = len(closes) // tf_min
            if n < 35:
                features.append(99.0)
            else:
                trimmed = closes[-n * tf_min:]
                tf_closes = trimmed[tf_min - 1::tf_min]  # close of each TF bar
                dist = _cross_dist(tf_closes)
                if dist != 99.0:
                    dist = (int(dist) // 2) * 2  # group: 0,1→0; 2,3→2
                features.append(float(dist))
            names.append(f'macd_cross_{tf_name}')
        
        # Additional MACD states (per timeframe)
        for tf_name, tf_closes in [('1m', closes)] + [
            (tn, closes[-len(closes)//tm*tm:][tm-1::tm])
            for tn, tm in [('5m',5),('15m',15),('30m',30),('1h',60)]
        ]:
            if len(tf_closes) < 35:
                features.extend([0.0]*5)
                names.extend([f'macd_difup_{tf_name}', f'macd_away_{tf_name}',
                              f'macd_opposite_{tf_name}', f'macd_hist_accel_{tf_name}',
                              f'macd_gap_grow_{tf_name}'])
                continue
            ema12 = _ema(tf_closes, 12)
            ema26 = _ema(tf_closes, 26)
            dif = ema12 - ema26
            dea = _ema(dif, 9)
            hist = dif - dea
            
            # 1. DIF above DEA
            features.append(1.0 if dif[-1] > dea[-1] else -1.0)
            names.append(f'macd_difup_{tf_name}')
            # 2. DIF moving away from zero AND hist same direction
            away = (dif[-1] > 0 and dif[-1] > dif[-2] and hist[-1] > 0) or \
                   (dif[-1] < 0 and dif[-1] < dif[-2] and hist[-1] < 0)
            features.append(1.0 if away else 0.0)
            names.append(f'macd_away_{tf_name}')
            # 3. DIF and DEA on opposite sides of zero
            features.append(1.0 if (dif[-1] > 0) != (dea[-1] > 0) else 0.0)
            names.append(f'macd_opposite_{tf_name}')
            # 4a. Histogram acceleration (curr - prev)
            features.append(float(hist[-1] - hist[-2]))
            names.append(f'macd_hist_accel_{tf_name}')
            # 4b. Gap |DEA| increasing
            features.append(1.0 if abs(dea[-1]) > abs(dea[-2]) else 0.0)
            names.append(f'macd_gap_grow_{tf_name}')
        
        return features, names
    
    def _build_volatility_features(
        self,
        c: float,
        timestamp,
        df: pd.DataFrame,
    ) -> Tuple[List[float], List[str]]:
        """C7: 波动率与时序特征"""
        features = []
        names = []
        
        closes = df['close'].values
        
        # 各时间窗口波动率
        for window, wname in [(5, '5min'), (15, '15min'), (30, '30min')]:
            if len(closes) >= window:
                vol = np.std(closes[-window:]) / (closes[-1] + 1e-9)  # 归一化
                features.append(vol)
                names.append(f'volatility_{wname}')
            else:
                features.append(0.0)
                names.append(f'volatility_{wname}')
        
        # 与前N根K线的收盘变化
        for n in [1, 2, 3, 5]:
            if len(closes) > n:
                change = (c - closes[-n-1]) / (closes[-n-1] + 1e-9)
                features.append(change)
                names.append(f'close_change_{n}bar')
            else:
                features.append(0.0)
                names.append(f'close_change_{n}bar')
        
        # 日内时间编码 (sin/cos 周期编码)
        if timestamp is not None:
            try:
                ts = pd.Timestamp(timestamp)
                minute_of_day = ts.hour * 60 + ts.minute
                day_of_week = ts.dayofweek
                
                features.append(np.sin(2 * np.pi * minute_of_day / 1440))
                features.append(np.cos(2 * np.pi * minute_of_day / 1440))
                features.append(np.sin(2 * np.pi * day_of_week / 5))
                features.append(np.cos(2 * np.pi * day_of_week / 5))
                names.extend(['time_sin', 'time_cos', 'dow_sin', 'dow_cos'])
            except Exception:
                features.extend([0.0] * 4)
                names.extend(['time_sin', 'time_cos', 'dow_sin', 'dow_cos'])
        
        return features, names


# ---- 特征维度预估 ----
def print_feature_summary():
    """打印特征维度预估"""
    print("""
    特征维度预估:
      C1 本体特征:          ~8维
      C2 影线/实体细分:     ~12维
      C3 多周期均线:        4参数 × 5均线 × 2指标 = 40维
      C4 多时间框架:        4参数 × 4框架 + 4趋势 = ~20维
      C5 形态识别:          ~12维
      C6 技术指标:          ~13维 (需ta-lib)
      C7 波动率+时序:       ~10维
    -------------------------------------
      总计:               ~105~115维 (不含ta-lib) / ~115~130维 (含ta-lib)
    """)


# ---- 快速测试 ----
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_feature_summary()
    
    # 生成模拟数据测试
    np.random.seed(42)
    n_bars = 500
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='1min')
    
    price = 1.10000 + np.cumsum(np.random.randn(n_bars) * 0.0001)
    df = pd.DataFrame({
        'open': price + np.random.randn(n_bars) * 0.00002,
        'high': price + np.abs(np.random.randn(n_bars) * 0.00005),
        'low': price - np.abs(np.random.randn(n_bars) * 0.00005),
        'close': price + np.random.randn(n_bars) * 0.00002,
        'tick_volume': np.random.randint(50, 500, n_bars),
    }, index=dates)
    
    # 确保 high>=open,close>=low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    builder = FeatureBuilder(symbol="EURUSD", point_size=0.00001)
    
    # 测试单条构建
    current = {'open': df['open'].iloc[-1], 'high': df['high'].iloc[-1],
               'low': df['low'].iloc[-1], 'close': df['close'].iloc[-1],
               'timestamp': df.index[-1]}
    
    vec = builder.build(current, df)
    print(f"\n单条向量维度: {vec.shape[1]}")
    print(f"特征名 (前10): {builder.feature_names[:10]}")
    print(f"特征值 (前10): {vec[0][:10]}")
    print(f"总特征数: {len(builder.feature_names)}")
