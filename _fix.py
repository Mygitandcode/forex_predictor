import numpy as np, pandas as pd
from pathlib import Path
# Load features
X = np.load(r"d:\dapanAllThings\codes\forex_predictor\data_store\EURUSD\features.npy")
n = len(X)
# Quick load CSV
df = pd.read_csv(r"E:\forex_data\cleaneded\EURUSD\EURUSD_M1.csv")
df = df[df['date'] >= '2015.01.01']
df = df[df['date'] <= '2025.12.31']
print(f"Loaded {len(df)} rows, features={n}")
# Datetime index
dts = []
for _, r in df.iterrows():
    t = str(r['time']).strip()
    if len(t) <= 5: t += ':00'
    dts.append(pd.Timestamp(f"{r['date']} {t}"))
df['dt'] = dts
df = df.dropna(subset=['dt']).set_index('dt')
df = df[['open','high','low','close','tick_volume']].dropna()
c = df['close'].values
# Timestamps
ts = df.index[200:200+n].values
np.save(r"d:\dapanAllThings\codes\forex_predictor\data_store\EURUSD\timestamps.npy", ts)
# Labels
for w in [7,10,15,30,60]:
    lbl = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        fi = 200 + i + w
        if fi < len(c): lbl[i] = 1 if c[fi] > c[200+i] else 0
    np.save(rf"d:\dapanAllThings\codes\forex_predictor\data_store\EURUSD\labels_{w}m.npy", lbl)
# Weights
age = (ts[-1] - ts) / np.timedelta64(1, 'D')
wgt = np.exp(-1.0 * age / max(age.max(),1)).astype(np.float32)
np.save(r"d:\dapanAllThings\codes\forex_predictor\data_store\EURUSD\weights.npy", wgt)
# Done
Path(r"d:\dapanAllThings\codes\forex_predictor\data_store\EURUSD\.done").touch()
print("DONE")
