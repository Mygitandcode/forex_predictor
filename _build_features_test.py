"""Test: build features for 1 week of EURUSD 15m candles → features_15m"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import pymysql, pandas as pd, numpy as np
from pymysql.cursors import DictCursor
from features.feature_builder import FeatureBuilder

MYSQL = dict(host="localhost",port=3305,user="root",password="",database="forex_predictor",charset="utf8mb4")

# Load 1 week of 15m candles (no DictCursor — pandas needs raw tuples for parse_dates)
conn = pymysql.connect(**MYSQL)
df = pd.read_sql(
    "SELECT timestamp, open, high, low, close, tick_volume FROM candles_15m "
    "WHERE symbol='EURUSD' AND timestamp >= '2026-05-01' AND timestamp < '2026-05-08' "
    "ORDER BY timestamp",
    conn, parse_dates=['timestamp']
)
conn.close()
print(f"Loaded {len(df)} candles: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")

if len(df) < 200:
    print("ERROR: need at least 200 bars for indicators")
    exit(1)

df.set_index('timestamp', inplace=True)
df['time'] = df.index

# Build features
builder = FeatureBuilder(symbol="EURUSD", point_size=0.00001)
print("Building features...")
X, feature_names = builder.build_many(df)
print(f"Feature matrix: {X.shape}, names: {len(feature_names)}")

# Build INSERT SQL
safe_names = [n.replace(' ','_').replace('-','_').replace('(','').replace(')','').replace('/','_') for n in feature_names]
cols_sql = ", ".join([f"`{sn}`" for sn in safe_names])
sql = f"INSERT INTO features_15m (timestamp, symbol, {cols_sql}, target_15m) VALUES (%s,%s,{','.join(['%s']*len(feature_names))},%s)"

# Write to DB
conn = pymysql.connect(**MYSQL, cursorclass=DictCursor)
cur = conn.cursor()
written = 0
for i in range(len(X)):
    ts = df.index[i].strftime('%Y-%m-%d %H:%M:%S')
    vals = [float(v) for v in X[i]] + [-1]  # target=-1 placeholder
    try:
        cur.execute(sql, [ts, "EURUSD"] + vals)
        written += 1
    except Exception as e:
        if i < 3:
            print(f"  Row {i} error: {e}")
        continue
conn.commit()

# Verify
cur.execute("SELECT COUNT(*) cnt, MIN(timestamp) mn, MAX(timestamp) mx FROM features_15m")
r = cur.fetchone()
print(f"\nfeatures_15m: {r['cnt']} rows, {r['mn']} ~ {r['mx']}")

# Show sample
cur.execute("SELECT * FROM features_15m LIMIT 1")
sample = cur.fetchone()
if sample:
    non_null = sum(1 for k,v in sample.items() if v is not None)
    print(f"Sample row: {non_null} non-null values out of {len(sample)} cols")

cur.close(); conn.close()
print("TEST DONE")
