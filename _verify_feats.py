"""Verify feature dimensions"""
import sys
sys.path.insert(0, r'd:\dapanAllThings\codes\forex_predictor')
import numpy as np
import pandas as pd
from features.feature_builder import FeatureBuilder

np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=800, freq='1min')
price = 1.10 + np.cumsum(np.random.randn(800) * 0.0001)
df = pd.DataFrame({
    'open': price + np.random.randn(800) * 0.00002,
    'high': price + abs(np.random.randn(800) * 0.00005),
    'low': price - abs(np.random.randn(800) * 0.00005),
    'close': price + np.random.randn(800) * 0.00002,
    'tick_volume': np.random.randint(50, 500, 800),
}, index=dates)
df['high'] = df[['open','high','close']].max(axis=1)
df['low'] = df[['open','low','close']].min(axis=1)

b = FeatureBuilder('EURUSD', 0.00001)
X, names = b.build_many(df)

print(f"\nTotal features: {len(names)}")
print(f"First 5 names: {names[:5]}")
print(f"Last 5 names:  {names[-5:]}")

# Quick category counting
cs = {'C1':0,'C1b_body':0,'C2':0,'C3':0,'C4':0,'C5':0,'C6_talib':0,'C6b_cross':0,'C6b_state':0,'C7':0}
for n in names:
    if n in ('high','low','close','upper_shadow','lower_shadow','body','body_position'):
        cs['C1']+=1
    elif n.startswith('body_range') or n.startswith('body_sum'):
        cs['C1b_body']+=1
    elif n.startswith(('upper_points','lower_points','body_points','upper_bucket','lower_bucket','body_bucket','shadow_ratio','body_shadow_ratio','is_bullish','upper_shadow_pct','lower_shadow_pct','body_pct')):
        cs['C2']+=1
    elif '_ma' in n:
        cs['C3']+=1
    elif n.endswith('_position') or n.endswith('_trend'):
        cs['C4']+=1
    elif n.startswith('pattern_'):
        cs['C5']+=1
    elif n in ('rsi_14','macd','macd_signal','macd_hist','bb_position','bb_width','atr_14','cci_20','stoch_k','stoch_d'):
        cs['C6_talib']+=1
    elif n.startswith('macd_cross_'):
        cs['C6b_cross']+=1
    elif n.startswith(('macd_difup_','macd_away_','macd_opposite_','macd_hist_accel_','macd_gap_grow_')):
        cs['C6b_state']+=1
    elif n.startswith(('volatility_','close_change_','time_','dow_')):
        cs['C7']+=1
    else:
        print(f"  UNKNOWN: {n}")

for k,v in sorted(cs.items()):
    print(f"  {k}: {v}d")
print(f"  Total: {sum(cs.values())}d")
