"""Recover labels/timestamps/weights after feature save bug fix.
Features already computed (4041890, 212). Just need aligned metadata.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from features.feature_store import FeatureStore
from training.data_weights import compute_time_weights

store = FeatureStore()  # default: E:/forex_npy
symbol = "EURUSD"

# Load existing features
X = np.load(store.symbol_dir(symbol) / "features.npy")
print(f"Features: {X.shape}")

# Reload the same data the pipeline used (2015-01-01 ~ 2025-12-31)
cleaned = r"E:\forex_data\cleaneded\EURUSD\EURUSD_M1.csv"
print("Loading CSV...")
df = pd.read_csv(cleaned)
print(f"Total rows: {len(df)}")

# Build datetime index same as pipeline
dts = []
for _, row in df.iterrows():
    try:
        t = str(row['time']).strip()
        if len(t) <= 5:
            t += ':00'
        dts.append(pd.Timestamp(f"{row['date']} {t}"))
    except:
        dts.append(pd.NaT)
df['datetime'] = dts
df = df.dropna(subset=['datetime']).set_index('datetime')
for c in ['open','high','low','close']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df[['open','high','low','close','tick_volume']].dropna()

# Filter to 2015-2025
df = df[df.index >= pd.Timestamp('2015-01-01')]
df = df[df.index <= pd.Timestamp('2025-12-31')]
print(f"Filtered: {len(df)} rows, {df.index[0]} ~ {df.index[-1]}")

# Align timestamps: features[0] corresponds to df.index[200]
n_features = len(X)
timestamps = df.index[200:200 + n_features].values
print(f"Timestamps: {len(timestamps)}, {timestamps[0]} ~ {timestamps[-1]}")

# Compute labels
closes = df['close'].values
windows = [7, 10, 15, 30, 60]
labels = {}
for w in windows:
    lbl = np.full(n_features, -1, dtype=np.int8)
    for i in range(n_features):
        fi = 200 + i + w
        if fi < len(closes):
            lbl[i] = 1 if closes[fi] > closes[200 + i] else 0
    labels[f"{w}m"] = lbl
    print(f"Labels {w}m: {(lbl>=0).sum()} valid")

# Compute weights
weights = compute_time_weights(timestamps, lambda_decay=1.0)

# Save all
store.save_features(symbol, X, timestamps)
store.save_labels(symbol, labels)
store.save_weights(symbol, weights)
store.mark_done(symbol)
print("\nAll saved. .done marked.")
