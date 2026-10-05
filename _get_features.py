"""Get feature names from FeatureBuilder"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd, numpy as np
from features.feature_builder import FeatureBuilder

builder = FeatureBuilder(symbol="EURUSD", point_size=0.00001)

# Generate minimal data to trigger feature name generation
df = pd.DataFrame({
    'open':  [1.1 + i*0.00001 for i in range(1000)],
    'high':  [1.1001 + i*0.00001 for i in range(1000)],
    'low':   [1.0999 + i*0.00001 for i in range(1000)],
    'close': [1.10005 + i*0.00001 for i in range(1000)],
    'tick_volume': [100]*1000,
})
df.index = pd.date_range('2025-01-01', periods=1000, freq='1min')
df['time'] = df.index

X, names = builder.build_many(df)
print(f"Total features: {len(names)}")
for i, name in enumerate(names):
    if i < 10 or i >= len(names)-10 or name in ['macd','macd_signal','macd_hist','rsi_14']:
        print(f"  f{i:>3}: {name}")
# Print all
for i, name in enumerate(names):
    print(f"FEATURE|{i}|{name}")
