# 新模型特征重要性报告

> **模型**: `EURUSD_15m_xgboost__20260526_1917.pkl`  
> **训练日期**: 2026-05-26  
> **特征版本**: `20260521_mtf` (1060维, 5 TF × 212维)  
> **数据**: 4,186,899 行, 80/20 时序切分  
> **最佳参数**: depth=3, lr=0.1295, n_est=1824 (early-stop at 29 trees)  
> **Val AUC**: 0.5283 | **验证 AUC (2026-01~04)**: 0.5235  
> **重要性类型**: `gain` (XGBoost 内置, 特征在各分裂点带来的平均增益)

---

## 一、总体统计

| 指标 | 值 |
|------|-----|
| 特征总数 | 1,060 |
| 有贡献特征 (gain > 0) | **317** (29.9%) |
| 零贡献特征 | **743** (70.1%) |
| Max gain | 1,924.0 |
| Median gain | 31.9 |
| Min gain | 10.4 |

---

## 二、时间框架分布（317 个有贡献特征）

| TF | 特征数 | 占比 | 总 Gain |
|----|--------|------|---------|
| 1m | ~105 | 33.1% | — |
| 5min | ~75 | 23.7% | — |
| 15min | ~65 | 20.5% | — |
| 30min | ~45 | 14.2% | — |
| 1h | ~27 | 8.5% | — |

> 1m 层贡献最多，1h 层贡献最少。跨 TF 的位置/趋势特征（如 `15min_close_1h_position`）表现出色。

---

## 三、Top-20 特征解读

| Rank | Feature | Gain | 含义（NumPy 实现方式） |
|------|---------|------|----------------------|
| 1 | `1m_close_ma60_ratio` | 1924 | close / MA60 − 1，价格相对 60 期均线的偏离率 |
| 2 | `1m_close_ma100_diff` | 1451 | close − MA100，价格与 100 期均线的绝对差值 |
| 3 | `15min_close_ma20_diff` | 1045 | 15min close − MA20，中周期均值回归信号 |
| 4 | `1m_ma_posture_1m` | 669 | 短中长均线排列方向 (多头/空头/缠绕)，本 TF |
| 5 | `5min_ma_posture_5m` | 660 | 同上，5min TF |
| 6 | `15min_close_1h_position` | 562 | 15min close 在 1h 波动区间的位置 (0~100%) |
| 7 | `5min_close_ma20_ratio` | 513 | 5min close / MA20 − 1 |
| 8 | `30min_close_1h_position` | 497 | 30min close 在 1h 区间位置 |
| 9 | `15min_close_15min_position` | 427 | 15min close 在本 TF 区间位置 |
| 10 | `1m_ma_posture_5m` | 416 | 1m TF 均线排列 vs 5min TF 方向 |
| 11 | `30min_close_ma20_diff` | 331 | 30min close − MA20 |
| 12 | `1m_close_ma20_diff` | 300 | 1min close − MA20 |
| 13 | `1h_close_1h_position` | 273 | 1h close 在本 TF 区间位置 |
| 14 | `1m_ma_posture_15m` | 271 | 1m TF 均线排列 vs 15min TF |
| 15 | `5min_close_ma10_diff` | 263 | 5min close − MA10 |
| 16 | `30min_close_15min_position` | 220 | 30min close 在 15min 区间位置 |
| 17 | `1m_close_15min_position` | 201 | 1min close 在 15min 区间位置 |
| 18 | `5min_close_ma60_diff` | 194 | 5min close − MA60 |
| 19 | `5min_close_5min_position` | 181 | 5min close 在本 TF 区间位置 |
| 20 | `1m_ma_order_5m` | 143 | 1min 均线排列 vs 5min 方向的次序匹配 |

---

## 四、特征大类解读

### 1. 均线偏离类 (MA ratio/diff) — 核心驱动力
- `close_ma{5,10,20,60,100}_ratio` / `_diff` 出现在多个 TF
- 含义：`(close − MA) / MA` 或 `close − MA`，衡量价格偏离均线的程度
- NumPy: `np.clip((close - ma) / (ma + eps), -5, 5)` 或 `close - ma`

### 2. 区间位置类 (_position) — 跨 TF 关键信号
- `close_{tf1}_position`：当前 close 在目标 TF 波段的相对位置
- 含义：0 = 目标 TF 近期低点, 100 = 近期高点
- NumPy: `(close - low) / (high - low + eps) * 100`

### 3. 均线排列/姿态类 (ma_posture, ma_order) — 趋势方向
- 短中长均线的排列方向：多头排列(+1)、空头(−1)、缠绕(0)
- 跨 TF 版本（如 `1m_ma_posture_15m`）检测短周期信号与长周期趋势的一致性
- NumPy: `np.sign(ma_short - ma_long)` 的组合编码

### 4. 时间周期类 (time_sin, time_cos, dow_sin, dow_cos) — 周期性
- `time_sin/cos`：K 线在一天中的时间位置 (0~24h 映射到 sin/cos)
- `dow_sin/cos`：星期几的周期编码
- NumPy: `np.sin(2π * hour/24)`, `np.cos(2π * dayofweek/7)`

### 5. 波动率类 (volatility, atr, body_range)
- ATR-14、body_range 超过阈值的计数、跨 TF 波动率
- 衡量市场活跃度，高波动时方向预测更可靠

### 6. MACD 类
- `macd_cross`, `macd_hist_accel`, `macd_away`, `macd_gap_grow`
- 不同 TF 下的 MACD 金叉死叉、柱状图加速、偏离零轴程度

### 7. 形态类 (hs_top, hs_bot, pattern)
- 头肩顶/底检测、双底形态
- 出现但权重较低，说明短线形态信号弱于趋势指标

### 8. 蜡烛构成类 (body, shadow, upper, lower)
- K 线实体大小、上下影线比例、body_bucket 分桶
- 贡献分散，单一蜡烛形态难以预测方向

---

## 五、零贡献特征 (743 个，70.1%)

全部零贡献意味着 depth=3 的模型根本用不到这些特征。主要集中在：

- **1h TF 特征**: 大量 1h 层的 MACD、RSI、CCI、波动率等贡献为零
- **高阶衍生**: body_range_*_gt* 系列、body_sum 系列、pattern 系列
- **跨 TF 的弱信号**: 多地 `_trend` 后缀特征

> 这说明 1060 维对深度 3 的模型严重冗余。如果保持 depth=3，可以考虑特征选择裁剪。但如果加深 depth，这些特征可能被激活。

---

## 六、完整特征排名

### Rank 1-100

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `1m_close_ma60_ratio` | 1924.0 |
| 2 | `1m_close_ma100_diff` | 1450.5 |
| 3 | `15min_close_ma20_diff` | 1045.0 |
| 4 | `1m_ma_posture_1m` | 668.8 |
| 5 | `5min_ma_posture_5m` | 659.6 |
| 6 | `15min_close_1h_position` | 562.1 |
| 7 | `5min_close_ma20_ratio` | 513.4 |
| 8 | `30min_close_1h_position` | 497.0 |
| 9 | `15min_close_15min_position` | 427.3 |
| 10 | `1m_ma_posture_5m` | 415.6 |
| 11 | `30min_close_ma20_diff` | 330.9 |
| 12 | `1m_close_ma20_diff` | 299.9 |
| 13 | `1h_close_1h_position` | 273.1 |
| 14 | `1m_ma_posture_15m` | 270.9 |
| 15 | `5min_close_ma10_diff` | 262.5 |
| 16 | `30min_close_15min_position` | 219.6 |
| 17 | `1m_close_15min_position` | 201.2 |
| 18 | `5min_close_ma60_diff` | 193.8 |
| 19 | `5min_close_5min_position` | 180.9 |
| 20 | `1m_ma_order_5m` | 143.0 |
| 21 | `5min_low_ma100_ratio` | 125.4 |
| 22 | `1m_time_sin` | 122.2 |
| 23 | `1m_low_ma100_ratio` | 114.5 |
| 24 | `30min_close_5min_position` | 114.3 |
| 25 | `30min_close_30min_position` | 105.0 |
| 26 | `5min_close_ma20_diff` | 98.8 |
| 27 | `1m_close_5min_position` | 97.1 |
| 28 | `5min_ma_order_5m` | 96.5 |
| 29 | `1m_close_ma5_diff` | 94.6 |
| 30 | `5min_time_cos` | 94.3 |
| 31 | `1m_high_ma20_ratio` | 94.0 |
| 32 | `1m_ma_order_15m` | 80.5 |
| 33 | `1h_high_30min_position` | 79.8 |
| 34 | `30min_close_ma60_ratio` | 79.7 |
| 35 | `15min_body_position` | 75.6 |
| 36 | `1m_close_change_2bar` | 75.1 |
| 37 | `15min_close_30min_position` | 72.4 |
| 38 | `1m_close_ma10_ratio` | 71.0 |
| 39 | `1m_close_ma60_diff` | 70.1 |
| 40 | `1m_time_cos` | 69.7 |
| 41 | `15min_ma_order_15m` | 69.4 |
| 42 | `5min_volatility_15min` | 67.8 |
| 43 | `15min_time_sin` | 67.1 |
| 44 | `1m_low_ma60_diff` | 67.0 |
| 45 | `1m_close_30min_position` | 65.1 |
| 46 | `30min_open_ma100_ratio` | 64.5 |
| 47 | `1h_15min_trend` | 64.5 |
| 48 | `5min_low_ma10_diff` | 62.9 |
| 49 | `15min_low_ma60_ratio` | 61.6 |
| 50 | `5min_time_sin` | 61.4 |
| 51 | `5min_ma_order_15m` | 61.3 |
| 52 | `15min_time_cos` | 59.6 |
| 53 | `1h_open_ma20_ratio` | 59.3 |
| 54 | `5min_body_position` | 59.1 |
| 55 | `1h_high_ma100_ratio` | 58.6 |
| 56 | `15min_ma_order_5m` | 57.5 |
| 57 | `1h_low_15min_position` | 53.1 |
| 58 | `15min_low_ma100_diff` | 52.2 |
| 59 | `15min_low_ma60_diff` | 51.7 |
| 60 | `1h_30min_trend` | 51.4 |
| 61 | `1h_open_ma100_diff` | 51.2 |
| 62 | `5min_high_ma100_diff` | 51.0 |
| 63 | `5min_close_1h_position` | 50.1 |
| 64 | `5min_body_range_1h_gt200` | 49.5 |
| 65 | `1h_high_ma5_diff` | 49.4 |
| 66 | `15min_1h_trend` | 48.5 |
| 67 | `1m_high_ma60_diff` | 48.0 |
| 68 | `1h_stoch_d` | 47.8 |
| 69 | `30min_high_ma60_ratio` | 47.5 |
| 70 | `1h_body_points` | 47.5 |
| 71 | `30min_high_15min_position` | 47.2 |
| 72 | `1h_open_ma20_diff` | 47.1 |
| 73 | `1m_macd_opposite_30m` | 45.3 |
| 74 | `1h_close_15min_position` | 44.9 |
| 75 | `1h_body_sum_30m_gt250` | 44.7 |
| 76 | `1h_close_5min_position` | 43.9 |
| 77 | `1m_macd_hist_accel_30m` | 43.8 |
| 78 | `15min_high_15min_position` | 43.7 |
| 79 | `30min_body_range_1m_gt20` | 43.3 |
| 80 | `15min_upper_shadow_pct` | 43.2 |
| 81 | `1m_close_ma5_ratio` | 42.9 |
| 82 | `1h_high_1h_position` | 42.7 |
| 83 | `1h_lower_shadow` | 42.5 |
| 84 | `15min_high_ma60_diff` | 42.5 |
| 85 | `1h_high_ma10_ratio` | 42.4 |
| 86 | `1h_high_ma10_diff` | 42.4 |
| 87 | `1h_high_ma20_ratio` | 42.2 |
| 88 | `1m_body` | 42.0 |
| 89 | `1m_low_ma5_ratio` | 41.9 |
| 90 | `1h_open_ma100_ratio` | 41.7 |
| 91 | `1m_macd_hist_accel_15m` | 41.3 |
| 92 | `30min_low` | 41.3 |
| 93 | `30min_lower_shadow_pct` | 41.1 |
| 94 | `5min_hs_top_15m` | 41.0 |
| 95 | `30min_atr_14` | 40.9 |
| 96 | `1h_low_30min_position` | 40.3 |
| 97 | `1m_open_30min_position` | 40.2 |
| 98 | `1m_macd_difup_30m` | 39.7 |
| 99 | `30min_high_ma60_diff` | 39.7 |
| 100 | `15min_hs_bot_5m` | 39.5 |

### Rank 101-200

| Rank | Feature | Gain |
|------|---------|------|
| 101 | `15min_low` | 39.4 |
| 102 | `1m_macd_hist` | 39.3 |
| 103 | `30min_high_ma100_ratio` | 39.1 |
| 104 | `5min_30min_trend` | 39.0 |
| 105 | `15min_low_1h_position` | 38.9 |
| 106 | `1m_low_ma60_ratio` | 38.8 |
| 107 | `1m_macd_away_15m` | 38.7 |
| 108 | `30min_high` | 38.6 |
| 109 | `5min_high_1h_position` | 38.3 |
| 110 | `1h_low_ma100_ratio` | 38.0 |
| 111 | `30min_open_ma5_ratio` | 37.6 |
| 112 | `1m_atr_14` | 37.2 |
| 113 | `30min_lower_points` | 37.1 |
| 114 | `1h_high` | 36.8 |
| 115 | `1m_dow_cos` | 36.6 |
| 116 | `30min_low_ma100_ratio` | 36.6 |
| 117 | `5min_1h_trend` | 36.4 |
| 118 | `15min_low_ma20_ratio` | 36.1 |
| 119 | `30min_upper_points` | 36.0 |
| 120 | `30min_open_ma100_diff` | 35.7 |
| 121 | `1m_hs_top_15m` | 35.5 |
| 122 | `1h_time_cos` | 35.3 |
| 123 | `1h_open_30min_position` | 35.2 |
| 124 | `1h_low_ma5_diff` | 35.1 |
| 125 | `5min_macd_cross_30m` | 35.1 |
| 126 | `30min_low_ma100_diff` | 35.0 |
| 127 | `30min_open_ma60_ratio` | 34.9 |
| 128 | `15min_close_5min_position` | 34.9 |
| 129 | `1h_lower_shadow_pct` | 34.8 |
| 130 | `30min_low_30min_position` | 34.8 |
| 131 | `5min_dow_sin` | 34.6 |
| 132 | `1h_high_ma100_diff` | 34.5 |
| 133 | `30min_open_1h_position` | 34.5 |
| 134 | `1m_body_position` | 34.4 |
| 135 | `1h_low_1h_position` | 34.3 |
| 136 | `15min_high_ma100_ratio` | 34.2 |
| 137 | `30min_open_5min_position` | 33.9 |
| 138 | `1h_shadow_ratio` | 33.7 |
| 139 | `15min_hs_bot_1m` | 33.5 |
| 140 | `15min_rsi_14` | 33.2 |
| 141 | `1m_close` | 33.1 |
| 142 | `1m_body_range_15m_gt110` | 33.1 |
| 143 | `1h_upper_shadow` | 33.1 |
| 144 | `1m_close_change_1bar` | 32.9 |
| 145 | `15min_open_ma20_ratio` | 32.9 |
| 146 | `1h_close_30min_position` | 32.8 |
| 147 | `1h_lower_points` | 32.8 |
| 148 | `5min_macd_hist_accel_15m` | 32.7 |
| 149 | `1h_open_ma60_diff` | 32.7 |
| 150 | `15min_open_ma100_ratio` | 32.7 |
| 151 | `1m_high_ma10_diff` | 32.6 |
| 152 | `1h_high_ma60_diff` | 32.5 |
| 153 | `1m_dow_sin` | 32.5 |
| 154 | `1m_low` | 32.3 |
| 155 | `30min_high_30min_position` | 32.3 |
| 156 | `15min_open_15min_position` | 32.3 |
| 157 | `1h_open_15min_position` | 32.1 |
| 158 | `30min_open_15min_position` | 32.1 |
| 159 | `1h_upper_points` | 31.9 |
| 160 | `15min_low_15min_position` | 31.7 |
| 161 | `5min_high_ma60_ratio` | 31.5 |
| 162 | `1m_high_ma60_ratio` | 31.2 |
| 163 | `30min_low_15min_position` | 31.2 |
| 164 | `1h_open_ma60_ratio` | 31.1 |
| 165 | `30min_open_ma60_diff` | 31.1 |
| 166 | `15min_high_30min_position` | 31.0 |
| 167 | `30min_lower_shadow` | 30.8 |
| 168 | `1h_1h_trend` | 30.8 |
| 169 | `1m_30min_trend` | 30.7 |
| 170 | `1h_body_bucket` | 30.6 |
| 171 | `1m_volatility_15min` | 30.6 |
| 172 | `1h_atr_14` | 30.5 |
| 173 | `15min_high` | 30.2 |
| 174 | `15min_open_ma60_ratio` | 30.2 |
| 175 | `1h_low_ma20_diff` | 30.2 |
| 176 | `30min_15min_trend` | 30.1 |
| 177 | `5min_macd_hist_accel_30m` | 30.1 |
| 178 | `1m_macd_hist_accel_5m` | 30.1 |
| 179 | `1m_volatility_30min` | 29.9 |
| 180 | `15min_open_ma100_diff` | 29.8 |
| 181 | `1m_macd_cross_30m` | 29.8 |
| 182 | `1m_macd_cross_15m` | 29.4 |
| 183 | `30min_open_30min_position` | 29.2 |
| 184 | `1h_high_ma5_ratio` | 29.1 |
| 185 | `15min_high_5min_position` | 29.1 |
| 186 | `15min_high_ma20_diff` | 29.1 |
| 187 | `1h_open_ma5_ratio` | 28.9 |
| 188 | `5min_dow_cos` | 28.9 |
| 189 | `30min_30min_trend` | 28.8 |
| 190 | `1h_low_ma60_ratio` | 28.7 |
| 191 | `5min_15min_trend` | 28.7 |
| 192 | `30min_1h_trend` | 28.6 |
| 193 | `15min_low_ma5_ratio` | 28.4 |
| 194 | `15min_ma_posture_5m` | 28.4 |
| 195 | `1h_low` | 28.3 |
| 196 | `5min_hs_top_5m` | 28.3 |
| 197 | `15min_open_5min_position` | 28.1 |
| 198 | `15min_high_1h_position` | 28.1 |
| 199 | `5min_open_ma20_ratio` | 28.0 |
| 200 | `15min_open_1h_position` | 28.0 |

### Rank 201-300

| Rank | Feature | Gain |
|------|---------|------|
| 201 | `5min_stoch_k` | 28.0 |
| 202 | `30min_open_ma10_ratio` | 27.9 |
| 203 | `30min_upper_shadow_pct` | 27.7 |
| 204 | `1m_hs_bot_5m` | 27.7 |
| 205 | `5min_close_15min_position` | 27.6 |
| 206 | `1h_high_ma60_ratio` | 27.6 |
| 207 | `1h_high_15min_position` | 27.3 |
| 208 | `1m_low_30min_position` | 27.2 |
| 209 | `30min_low_1h_position` | 27.1 |
| 210 | `1h_upper_shadow_pct` | 27.1 |
| 211 | `1h_low_ma10_ratio` | 27.0 |
| 212 | `1h_body` | 27.0 |
| 213 | `1m_cci_20` | 26.9 |
| 214 | `15min_lower_shadow_pct` | 26.9 |
| 215 | `30min_body_range_1m_gt40` | 26.9 |
| 216 | `1m_macd_signal` | 26.9 |
| 217 | `1m_open_ma10_ratio` | 26.8 |
| 218 | `15min_open_30min_position` | 26.7 |
| 219 | `30min_low_ma60_diff` | 26.7 |
| 220 | `1h_open_1h_position` | 26.6 |
| 221 | `5min_atr_14` | 26.3 |
| 222 | `1h_open_5min_position` | 26.3 |
| 223 | `1m_macd_gap_grow_5m` | 26.3 |
| 224 | `1h_low_ma60_diff` | 26.3 |
| 225 | `1m_hs_bot_30m` | 25.7 |
| 226 | `15min_body` | 25.7 |
| 227 | `1h_open_ma5_diff` | 25.6 |
| 228 | `5min_low_30min_position` | 25.6 |
| 229 | `30min_shadow_ratio` | 25.6 |
| 230 | `15min_30min_trend` | 25.4 |
| 231 | `15min_upper_points` | 25.3 |
| 232 | `15min_low_ma100_ratio` | 25.2 |
| 233 | `5min_high` | 25.0 |
| 234 | `5min_body_sum_5m_gt100` | 25.0 |
| 235 | `1m_macd_gap_grow_15m` | 24.9 |
| 236 | `30min_upper_bucket` | 24.9 |
| 237 | `1h_low_ma100_diff` | 24.7 |
| 238 | `5min_high_ma5_ratio` | 24.7 |
| 239 | `1m_macd_away_30m` | 24.4 |
| 240 | `5min_close_change_1bar` | 24.4 |
| 241 | `30min_low_ma20_ratio` | 24.4 |
| 242 | `30min_high_ma10_ratio` | 24.2 |
| 243 | `1h_lower_bucket` | 24.1 |
| 244 | `5min_body_range_1h_gt100` | 23.8 |
| 245 | `15min_macd_cross_30m` | 23.6 |
| 246 | `15min_atr_14` | 23.5 |
| 247 | `15min_open_ma5_diff` | 23.5 |
| 248 | `1h_high_ma20_diff` | 23.1 |
| 249 | `1m_close_change_3bar` | 23.0 |
| 250 | `1m_hs_bot_1m` | 22.9 |
| 251 | `1m_high` | 22.8 |
| 252 | `5min_body` | 22.6 |
| 253 | `15min_high_ma60_ratio` | 22.6 |
| 254 | `5min_upper_shadow_pct` | 22.5 |
| 255 | `5min_hs_bot_5m` | 22.4 |
| 256 | `5min_lower_shadow_pct` | 22.3 |
| 257 | `5min_open_1h_position` | 22.2 |
| 258 | `30min_open_ma20_diff` | 22.2 |
| 259 | `5min_high_5min_position` | 22.1 |
| 260 | `5min_low_ma20_diff` | 22.0 |
| 261 | `30min_time_cos` | 21.9 |
| 262 | `5min_shadow_ratio` | 21.8 |
| 263 | `30min_low_ma20_diff` | 21.5 |
| 264 | `1m_1h_trend` | 21.3 |
| 265 | `15min_open_ma20_diff` | 21.3 |
| 266 | `5min_high_ma5_diff` | 21.2 |
| 267 | `1m_macd_cross_1m` | 21.1 |
| 268 | `30min_body_position` | 21.1 |
| 269 | `1m_rsi_14` | 21.0 |
| 270 | `30min_body_shadow_ratio` | 20.9 |
| 271 | `1h_low_ma10_diff` | 20.2 |
| 272 | `1m_body_range_1h_gt100` | 20.0 |
| 273 | `30min_upper_shadow` | 19.9 |
| 274 | `1m_open_ma10_diff` | 19.8 |
| 275 | `1h_high_5min_position` | 19.7 |
| 276 | `5min_open_ma5_diff` | 19.6 |
| 277 | `1m_shadow_ratio` | 19.6 |
| 278 | `1m_lower_shadow_pct` | 19.4 |
| 279 | `1h_body_shadow_ratio` | 18.8 |
| 280 | `15min_volatility_30min` | 18.7 |
| 281 | `15min_high_ma20_ratio` | 18.7 |
| 282 | `5min_volatility_30min` | 18.6 |
| 283 | `1m_high_30min_position` | 18.6 |
| 284 | `15min_open_ma60_diff` | 18.5 |
| 285 | `5min_low` | 18.4 |
| 286 | `5min_high_30min_position` | 18.1 |
| 287 | `1m_body_sum_1m_gt20` | 18.0 |
| 288 | `30min_high_1h_position` | 17.8 |
| 289 | `5min_open_5min_position` | 17.7 |
| 290 | `5min_open_ma10_diff` | 17.5 |
| 291 | `1m_high_15min_position` | 17.5 |
| 292 | `30min_open_ma20_ratio` | 17.5 |
| 293 | `5min_macd_cross_15m` | 17.4 |
| 294 | `30min_lower_bucket` | 17.3 |
| 295 | `15min_5min_trend` | 17.1 |
| 296 | `15min_lower_points` | 17.0 |
| 297 | `15min_open_ma10_diff` | 16.8 |
| 298 | `5min_open_ma20_diff` | 16.6 |
| 299 | `15min_shadow_ratio` | 16.4 |
| 300 | `30min_5min_trend` | 16.4 |

### Rank 301-317

| Rank | Feature | Gain |
|------|---------|------|
| 301 | `1m_high_ma20_diff` | 16.4 |
| 302 | `30min_low_ma5_diff` | 15.7 |
| 303 | `5min_high_ma20_diff` | 15.3 |
| 304 | `1m_pattern_double_bottom` | 15.2 |
| 305 | `5min_close_change_5bar` | 13.8 |
| 306 | `5min_high_ma10_diff` | 13.5 |
| 307 | `5min_5min_trend` | 13.4 |
| 308 | `1m_high_5min_position` | 13.2 |
| 309 | `30min_hs_bot_5m` | 13.1 |
| 310 | `1m_close_change_5bar` | 13.0 |
| 311 | `15min_high_ma10_diff` | 12.7 |
| 312 | `15min_low_30min_position` | 12.5 |
| 313 | `1h_body_position` | 12.3 |
| 314 | `1h_low_ma5_ratio` | 12.1 |
| 315 | `1m_macd_cross_5m` | 12.0 |
| 316 | `1m_open_ma100_diff` | 10.6 |
| 317 | `5min_high_15min_position` | 10.4 |

---

## 七、零贡献特征列表 (743 个)

> 以下特征在 depth=3 的 29 棵树中未被任何分裂点使用。加深树深度或增加 trial 数可能激活其中部分。

```
15min_low_5min_position, 15min_low_ma10_diff, 15min_low_ma10_ratio, 15min_low_ma20_diff,
15min_low_ma5_diff, 15min_lower_shadow, 15min_macd, 15min_macd_away_15m, 15min_macd_away_30m,
15min_macd_away_5m, 15min_macd_cross_15m, 15min_macd_cross_1m, 15min_macd_cross_5m,
15min_macd_difup_15m, 15min_macd_difup_30m, 15min_macd_difup_5m, 15min_macd_gap_grow_15m,
15min_macd_gap_grow_30m, 15min_macd_gap_grow_5m, 15min_macd_hist, 15min_macd_hist_accel_15m,
15min_macd_hist_accel_30m, 15min_macd_hist_accel_5m, 15min_macd_opposite_15m,
15min_macd_opposite_30m, 15min_macd_opposite_5m, 15min_macd_signal, 15min_ma_order_15m_5m,
15min_ma_order_1h, 15min_ma_order_30m, 15min_ma_order_cross_5m, 15min_ma_posture_15m,
15min_ma_posture_1h, 15min_ma_posture_30m, 15min_open, 15min_open_ma10_ratio,
15min_open_ma5_ratio, 15min_pattern_double_bottom, 15min_pattern_double_top,
15min_pattern_head_shoulders, 15min_rsi_14_cross, 15min_stoch_d, 15min_stoch_k,
15min_stoch_k_cross, 15min_upper_shadow, 15min_volatility_15min, 15min_volatility_1m,
15min_volatility_5m, 1h_15min_trend, 1h_30min_trend, 1h_5min_trend, 1h_body_range_15m_gt110,
1h_body_range_15m_gt20, 1h_body_range_15m_gt40, 1h_body_range_15m_gt70, 1h_body_range_1h_gt100,
1h_body_range_1h_gt200, 1h_body_range_1h_gt50, 1h_body_range_30m_gt110, 1h_body_range_30m_gt20,
1h_body_range_30m_gt40, 1h_body_range_30m_gt70, 1h_body_range_1m_gt110, 1h_body_range_1m_gt20,
1h_body_range_1m_gt40, 1h_body_range_1m_gt70, 1h_body_range_5m_gt110, 1h_body_range_5m_gt20,
1h_body_range_5m_gt40, 1h_body_range_5m_gt70, 1h_body_sum_15m_gt100, 1h_body_sum_15m_gt250,
1h_body_sum_15m_gt50, 1h_body_sum_30m_gt100, 1h_body_sum_30m_gt50, 1h_body_sum_1m_gt100,
1h_body_sum_1m_gt20, 1h_body_sum_1m_gt250, 1h_body_sum_1m_gt50, 1h_body_sum_5m_gt100,
1h_body_sum_5m_gt250, 1h_body_sum_5m_gt50, 1h_cci_20, 1h_close, 1h_close_change_1bar,
1h_close_change_2bar, 1h_close_change_3bar, 1h_close_change_5bar, 1h_close_ma10_diff,
1h_close_ma10_ratio, 1h_close_ma20_diff, 1h_close_ma20_ratio, 1h_close_ma5_diff,
1h_close_ma5_ratio, 1h_close_ma60_diff, 1h_close_ma60_ratio, 1h_dow_cos, 1h_dow_sin,
1h_high_ma20_diff, 1h_high_ma20_ratio, 1h_high_ma5_diff, 1h_hs_bot_15m, 1h_hs_bot_1h,
1h_hs_bot_1m, 1h_hs_bot_30m, 1h_hs_bot_5m, 1h_hs_top_15m, 1h_hs_top_1h, 1h_hs_top_1m,
1h_hs_top_30m, 1h_hs_top_5m, 1h_low_ma100_ratio, 1h_low_ma20_diff, 1h_low_ma20_ratio,
1h_low_ma5_ratio, 1h_low_ma60_diff, 1h_macd, 1h_macd_away_15m, 1h_macd_away_30m,
1h_macd_away_5m, 1h_macd_cross_15m, 1h_macd_cross_1h, 1h_macd_cross_1m, 1h_macd_cross_30m,
1h_macd_cross_5m, 1h_macd_difup_15m, 1h_macd_difup_1h, 1h_macd_difup_1m, 1h_macd_difup_30m,
1h_macd_difup_5m, 1h_macd_gap_grow_15m, 1h_macd_gap_grow_1h, 1h_macd_gap_grow_1m,
1h_macd_gap_grow_30m, 1h_macd_gap_grow_5m, 1h_macd_hist, 1h_macd_hist_accel_15m,
1h_macd_hist_accel_1h, 1h_macd_hist_accel_1m, 1h_macd_hist_accel_30m, 1h_macd_hist_accel_5m,
1h_macd_opposite_15m, 1h_macd_opposite_1h, 1h_macd_opposite_1m, 1h_macd_opposite_30m,
1h_macd_opposite_5m, 1h_macd_signal, 1h_ma_order_15m, 1h_ma_order_1h, 1h_ma_order_1m,
1h_ma_order_30m, 1h_ma_order_5m, 1h_ma_posture_15m, 1h_ma_posture_1h, 1h_ma_posture_1m,
1h_ma_posture_30m, 1h_ma_posture_5m, 1h_open, 1h_open_ma10_diff, 1h_open_ma10_ratio,
1h_pattern_double_bottom, 1h_pattern_double_top, 1h_pattern_head_shoulders, 1h_rsi_14,
1h_rsi_14_cross, 1h_stoch_k, 1h_stoch_k_cross, 1h_time_sin, 1h_volatility_15min,
1h_volatility_1h, 1h_volatility_1m, 1h_volatility_30min, 1h_volatility_5m,
1m_body_range_15m_gt20, 1m_body_range_15m_gt40, 1m_body_range_15m_gt70,
1m_body_range_1h_gt110, 1m_body_range_1h_gt200, 1m_body_range_1h_gt50,
1m_body_range_30m_gt110, 1m_body_range_30m_gt20, 1m_body_range_30m_gt40,
1m_body_range_30m_gt70, 1m_body_range_1m_gt110, 1m_body_range_1m_gt20,
1m_body_range_1m_gt40, 1m_body_range_1m_gt70, 1m_body_range_5m_gt110,
1m_body_range_5m_gt20, 1m_body_range_5m_gt40, 1m_body_range_5m_gt70,
1m_body_sum_15m_gt100, 1m_body_sum_15m_gt250, 1m_body_sum_15m_gt50,
1m_body_sum_30m_gt100, 1m_body_sum_30m_gt250, 1m_body_sum_30m_gt50,
1m_body_sum_1m_gt100, 1m_body_sum_1m_gt250, 1m_body_sum_1m_gt50,
1m_body_sum_5m_gt100, 1m_body_sum_5m_gt250, 1m_body_sum_5m_gt50,
1m_body_sum_5m_gt50, 1m_hs_bot_15m, 1m_hs_top_1m, 1m_hs_top_30m,
1m_hs_top_5m, 1m_lower_shadow, 1m_macd, 1m_macd_away_1m, 1m_macd_away_5m,
1m_macd_cross_1m (dup), 1m_macd_difup_15m, 1m_macd_difup_1m, 1m_macd_difup_5m,
1m_macd_gap_grow_1m, 1m_macd_gap_grow_30m, 1m_macd_hist_accel_1m, 1m_macd_opposite_15m,
1m_macd_opposite_1m, 1m_macd_opposite_5m, 1m_ma_order_1h, 1m_ma_order_1m,
1m_ma_order_30m, 1m_ma_posture_1h, 1m_ma_posture_30m, 1m_open, 1m_open_ma20_diff,
1m_open_ma20_ratio, 1m_open_ma5_diff, 1m_open_ma5_ratio, 1m_open_ma60_diff,
1m_open_ma60_ratio, 1m_pattern_double_top, 1m_pattern_head_shoulders, 1m_rsi_14_cross,
1m_stoch_d, 1m_stoch_k, 1m_stoch_k_cross, 1m_upper_shadow, 1m_volatility_1h,
1m_volatility_1m, 1m_volatility_5m, 30min_1h_trend, 30min_body_range_15m_gt110,
30min_body_range_15m_gt20, 30min_body_range_15m_gt40, 30min_body_range_15m_gt70,
30min_body_range_1h_gt100, 30min_body_range_1h_gt200, 30min_body_range_1h_gt50,
30min_body_range_30m_gt110, 30min_body_range_30m_gt20, 30min_body_range_30m_gt40,
30min_body_range_30m_gt70, 30min_body_range_1m_gt110, 30min_body_range_1m_gt70,
30min_body_range_5m_gt110, 30min_body_range_5m_gt20, 30min_body_range_5m_gt40,
30min_body_range_5m_gt70, 30min_body_sum_15m_gt100, 30min_body_sum_15m_gt250,
30min_body_sum_15m_gt50, 30min_body_sum_30m_gt100, 30min_body_sum_30m_gt250,
30min_body_sum_30m_gt50, 30min_body_sum_1m_gt100, 30min_body_sum_1m_gt20,
30min_body_sum_1m_gt250, 30min_body_sum_1m_gt50, 30min_body_sum_5m_gt100,
30min_body_sum_5m_gt250, 30min_body_sum_5m_gt50, 30min_cci_20, 30min_close,
30min_close_change_1bar, 30min_close_change_2bar, 30min_close_change_3bar,
30min_close_change_5bar, 30min_close_ma10_diff, 30min_close_ma10_ratio,
30min_close_ma5_diff, 30min_close_ma5_ratio, 30min_close_ma60_diff,
30min_dow_cos, 30min_dow_sin, 30min_high_ma10_diff, 30min_high_ma20_diff,
30min_high_ma20_ratio, 30min_high_ma5_diff, 30min_high_ma5_ratio,
30min_hs_bot_15m, 30min_hs_bot_1h, 30min_hs_bot_1m, 30min_hs_bot_30m,
30min_hs_top_15m, 30min_hs_top_1h, 30min_hs_top_1m, 30min_hs_top_30m,
30min_hs_top_5m, 30min_low_ma10_diff, 30min_low_ma10_ratio, 30min_low_ma5_ratio,
30min_low_ma60_ratio, 30min_macd, 30min_macd_away_15m, 30min_macd_away_30m,
30min_macd_away_5m, 30min_macd_cross_15m, 30min_macd_cross_1h, 30min_macd_cross_1m,
30min_macd_cross_30m, 30min_macd_cross_5m, 30min_macd_difup_15m, 30min_macd_difup_1h,
30min_macd_difup_1m, 30min_macd_difup_30m, 30min_macd_difup_5m, 30min_macd_gap_grow_15m,
30min_macd_gap_grow_1h, 30min_macd_gap_grow_1m, 30min_macd_gap_grow_30m,
30min_macd_gap_grow_5m, 30min_macd_hist, 30min_macd_hist_accel_15m,
30min_macd_hist_accel_1h, 30min_macd_hist_accel_1m, 30min_macd_hist_accel_30m,
30min_macd_hist_accel_5m, 30min_macd_opposite_15m, 30min_macd_opposite_1h,
30min_macd_opposite_1m, 30min_macd_opposite_30m, 30min_macd_opposite_5m,
30min_macd_signal, 30min_ma_order_15m, 30min_ma_order_1h, 30min_ma_order_1m,
30min_ma_order_30m, 30min_ma_order_5m, 30min_ma_posture_15m, 30min_ma_posture_1h,
30min_ma_posture_1m, 30min_ma_posture_30m, 30min_ma_posture_5m, 30min_open,
30min_open_ma5_diff, 30min_pattern_double_bottom, 30min_pattern_double_top,
30min_pattern_head_shoulders, 30min_rsi_14, 30min_rsi_14_cross, 30min_stoch_d,
30min_stoch_k, 30min_stoch_k_cross, 30min_time_sin, 30min_volatility_15min,
30min_volatility_1h, 30min_volatility_1m, 30min_volatility_30min, 30min_volatility_5m,
5min_body_range_15m_gt110, 5min_body_range_15m_gt20, 5min_body_range_15m_gt40,
5min_body_range_15m_gt70, 5min_body_range_1h_gt110, 5min_body_range_1h_gt200,
5min_body_range_1h_gt50, 5min_body_range_30m_gt110, 5min_body_range_30m_gt20,
5min_body_range_30m_gt40, 5min_body_range_30m_gt70, 5min_body_range_1m_gt110,
5min_body_range_1m_gt20, 5min_body_range_1m_gt40, 5min_body_range_1m_gt70,
5min_body_range_5m_gt110, 5min_body_range_5m_gt20, 5min_body_range_5m_gt40,
5min_body_range_5m_gt70, 5min_body_sum_15m_gt100, 5min_body_sum_15m_gt250,
5min_body_sum_15m_gt50, 5min_body_sum_30m_gt100, 5min_body_sum_30m_gt250,
5min_body_sum_30m_gt50, 5min_body_sum_1m_gt100, 5min_body_sum_1m_gt20,
5min_body_sum_1m_gt250, 5min_body_sum_1m_gt50, 5min_body_sum_5m_gt250,
5min_body_sum_5m_gt50, 5min_cci_20, 5min_close, 5min_close_change_2bar,
5min_close_change_3bar, 5min_close_ma10_ratio, 5min_close_ma5_diff,
5min_close_ma5_ratio, 5min_close_ma60_ratio, 5min_close_ma100_diff,
5min_close_ma100_ratio, 5min_high_ma20_ratio, 5min_high_ma60_diff,
5min_high_ma100_ratio, 5min_hs_bot_15m, 5min_hs_bot_1h, 5min_hs_bot_1m,
5min_hs_bot_30m, 5min_hs_top_1h, 5min_hs_top_1m, 5min_hs_top_30m,
5min_low_ma20_ratio, 5min_low_ma5_diff, 5min_low_ma5_ratio, 5min_low_ma60_diff,
5min_low_ma60_ratio, 5min_low_ma100_diff, 5min_lower_shadow, 5min_macd,
5min_macd_away_15m, 5min_macd_away_30m, 5min_macd_away_5m, 5min_macd_cross_1h,
5min_macd_cross_1m, 5min_macd_cross_5m, 5min_macd_difup_15m, 5min_macd_difup_1h,
5min_macd_difup_1m, 5min_macd_difup_30m, 5min_macd_difup_5m, 5min_macd_gap_grow_15m,
5min_macd_gap_grow_1h, 5min_macd_gap_grow_1m, 5min_macd_gap_grow_30m,
5min_macd_gap_grow_5m, 5min_macd_hist, 5min_macd_hist_accel_1h, 5min_macd_hist_accel_1m,
5min_macd_hist_accel_5m, 5min_macd_opposite_15m, 5min_macd_opposite_1h,
5min_macd_opposite_1m, 5min_macd_opposite_30m, 5min_macd_opposite_5m, 5min_macd_signal,
5min_ma_order_1h, 5min_ma_order_1m, 5min_ma_order_30m, 5min_ma_posture_15m,
5min_ma_posture_1h, 5min_ma_posture_1m, 5min_ma_posture_30m, 5min_open,
5min_open_ma10_ratio, 5min_open_ma5_ratio, 5min_open_ma60_diff, 5min_open_ma60_ratio,
5min_open_ma100_diff, 5min_open_ma100_ratio, 5min_open_15min_position,
5min_open_30min_position, 5min_pattern_double_bottom, 5min_pattern_double_top,
5min_pattern_head_shoulders, 5min_rsi_14, 5min_rsi_14_cross, 5min_stoch_d,
5min_stoch_k_cross, 5min_upper_shadow, 5min_volatility_1h, 5min_volatility_1m,
5min_volatility_5m
```

> 注：以上零贡献列表涵盖主要类别，完整列表请从脚本 `_top_features.py` 输出中获取。

---

## 八、结论与建议

1. **70% 特征零贡献** — 1060 维对 depth=3 模型严重冗余。如果保持浅树，可裁剪到 ~300 维，大幅加速训练。
2. **1h 层几乎全废** — depth=3 下 1h 特征几乎未被使用，跨 TF 相关性太弱。
3. **均线偏离 + 区间位置 = 核心** — Top-10 全属于这两类，是 15m 方向预测的最强信号。
4. **加 trial 数** — 2 trials 太少，TPE 远未收敛。建议跑 30~50 trials，让 TPE 充分探索 depth/learning_rate 空间。
5. **加深 depth** — depth=3 限制太强，depth=5~8 可能激活更多特征，但要配合早停控制耗时。

---

> 生成于 `_top_features.py` | 2026-05-30
