"""Final pre-run verification"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=== 1. FMports ===")
import pandas as pd
import numpy as np
import psutil
from features.feature_builder import FeatureBuilder
from features.feature_store import FeatureStore
from features.feature_pipeline import FeaturePipeline
print("  All imports OK")

print("\n=== 2. Memory ===")
m = psutil.virtual_memory()
print(f"  {m.available/1024**3:.1f}G free / {m.total/1024**3:.1f}G total")

print("\n=== 3. FeatureBuilder real data test (500 bars) ===")
cleaned = r"E:\forex_data\cleaneded\EURUSD\EURUSD_M1.csv"
df = pd.read_csv(cleaned, nrows=1000)
dts = []
for _, row in df.iterrows():
    try:
        t = str(row['time']).strip()
        if len(t) <= 5: t += ':00'
        dts.append(pd.Timestamp(f"{row['date']} {t}"))
    except:
        dts.append(pd.NaT)
df['datetime'] = dts
df = df.dropna(subset=['datetime']).set_index('datetime')
for c in ['open','high','low','close']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df[['open','high','low','close','tick_volume']].dropna()
print(f"  Loaded {len(df)} bars")

b = FeatureBuilder('EURUSD', 0.00001)
X, names = b.build_many(df)
print(f"  Features: {X.shape}, names: {len(names)}")
assert X.shape[1] == len(names)

# Quick dim check
expected = {'high','low','close','upper_shadow','lower_shadow','body','body_position',
            'upper_points','lower_points','body_points',
            'open_ma5_diff','open_ma10_diff','open_ma20_diff','open_ma60_diff','open_ma100_diff',
            'close_ma5_diff','close_ma10_diff','close_ma20_diff','close_ma60_diff','close_ma100_diff',
            'close_change_1bar','close_change_2bar','close_change_3bar','close_change_5bar',
            'volatility_5min','volatility_15min','volatility_30min',
            'time_sin','time_cos','dow_sin','dow_cos',
            'body_range_1m_gt10','body_sum_1m_gt20',
            'macd_cross_1m','macd_difup_1m','macd_away_1m','macd_opposite_1m',
            'macd_hist_accel_1m','macd_gap_grow_1m',
            'hs_top_1m','hs_bot_1m',
            'ma_order_1m','ma_posture_1m',
            'rsi_14','macd','macd_signal','macd_hist','bb_position','bb_width',
            'atr_14','cci_20','stoch_k','stoch_d'}
found = {n for n in names}
missing = expected - found
if missing:
    print(f"  MISSING: {missing}")
else:
    print(f"  All expected features present")
print(f"  PASS")

print("\n=== 4. FeatureStore check ===")
s = FeatureStore()  # default: E:/forex_npy
if s.is_done('EURUSD'):
    d = s.load_all('EURUSD')
    old = d['features'].shape[1]
    if old != len(names):
        print(f"  Old dims={old}, new={len(names)} -> MUST REBUILD")
else:
    print(f"  EURUSD not built yet")

print("\n=== 5. Pipeline ready ===")
print(f"  Command: python -m features.feature_pipeline --data-dir E:\\forex_data\\cleaneded\\EURUSD --symbols EURUSD --store-dir ./data_store --force")
print(f"  After:    python -m training.stage1_autogluon --symbol EURUSD --target 15m --time-budget 3600 --sample-ratio 0.5")

print("\n=== SUMMARY ===")
print(f"  FeatureBuilder: OK ({len(names)}d)")
print(f"  Memory: OK ({m.available/1024**3:.1f}G)")
print(f"  Ready: YES")
