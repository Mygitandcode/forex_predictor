"""Fast recovery — only load tail of CSV, skip full read"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from features.feature_store import FeatureStore
from training.data_weights import compute_time_weights

store = FeatureStore()  # default: E:/forex_npy
X = np.load(store.symbol_dir("EURUSD") / "features.npy")
n = len(X)
print(f"Features: ({n}, {X.shape[1]})")

# Only load what we need: 200 skipped + n features + max_label_window(60) rows before end
# The feature data covers 2015-2025. We need the original CSV rows
# indexed 200 to 200+n+60 from the 2015-start position.
# Strategy: load the full file but use skiprows/read in 2 passes

cleaned = r"E:\forex_data\cleaneded\EURUSD\EURUSD_M1.csv"

# Find where 2015-01-01 starts (use date column quick scan)
print("Scanning for 2015 start...")
chunk_size = 500000
offset = 0
for chunk in pd.read_csv(cleaned, chunksize=chunk_size):
    mask = chunk['date'].str.startswith('2015')
    if mask.any():
        first_idx = mask.idxmax()
        offset += first_idx
        break
    offset += len(chunk)

print(f"2015 starts at row {offset}")

# Now load from offset onwards
print("Loading 2015-2025...")
all_rows = []
for chunk in pd.read_csv(cleaned, skiprows=range(1, offset+1), chunksize=chunk_size):
    mask = ~chunk['date'].str.startswith('2026')
    all_rows.append(chunk[mask])
    if not mask.all():  # hit 2026
        break
df = pd.concat(all_rows, ignore_index=True)
print(f"Loaded {len(df)} rows")

# Build datetime index
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
print(f"Cleaned: {len(df)} bars, {df.index[0]} ~ {df.index[-1]}")

# Timestamps
timestamps = df.index[200:200 + n].values
print(f"Timestamps: {len(timestamps)}")

# Labels
closes = df['close'].values
for w in [7, 10, 15, 30, 60]:
    lbl = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        fi = 200 + i + w
        if fi < len(closes):
            lbl[i] = 1 if closes[fi] > closes[200 + i] else 0
    store.save_labels("EURUSD", {f"{w}m": lbl})
    print(f"Labels {w}m: {(lbl>=0).sum()} valid")

# Weights
store.save_weights("EURUSD", compute_time_weights(timestamps, 1.0))

# Features (already there but re-save to trigger atomic write)
store.save_features("EURUSD", X, timestamps)
store.mark_done("EURUSD")
print("DONE")
