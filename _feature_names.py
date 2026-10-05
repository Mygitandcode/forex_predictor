"""Map feature index to name using FeatureBuilder"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from features.feature_builder import FeatureBuilder

builder = FeatureBuilder(symbol="EURUSD", point_size=0.00001)
# Trigger feature name generation by building a single sample
import pandas as pd, numpy as np
df = pd.DataFrame({
    'open': [1.1 + i*0.00001 for i in range(1000)],
    'high': [1.1001 + i*0.00001 for i in range(1000)],
    'low': [1.0999 + i*0.00001 for i in range(1000)],
    'close': [1.10005 + i*0.00001 for i in range(1000)],
    'tick_volume': [100] * 1000,
})
df.index = pd.date_range('2025-01-01', periods=1000, freq='1min')
df['time'] = df.index

X, names = builder.build_many(df)

print(f"Total features: {len(names)}")
print()

# Print specific feature indices mentioned in SHAP top results
targets = [209, 208, 141, 136, 131, 167, 115, 96, 116, 154, 95, 139, 126, 189, 87, 112, 142, 134, 2, 114, 203, 187, 117, 13, 184, 202, 163, 135, 158, 170]
for idx in targets:
    if idx < len(names):
        print(f"  f{idx:>3}: {names[idx]}")
