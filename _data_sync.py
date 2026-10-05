"""
EURUSD data sync: local CSV files -> MySQL
  Source 1: E:/forex_data/cleaned/EURUSD/ (histdata.com M1 2015-2026-04)
  Source 2: E:/forex_data/userDealByHands/ (MT4 recent gap fill)
  Run: python _data_sync.py
  Safe to re-run: INSERT IGNORE handles duplicates
"""
import pymysql, csv, os, sys, argparse
from pymysql.cursors import DictCursor
from datetime import datetime

MYSQL = dict(host="localhost", port=3305, user="root", password="", database="forex_predictor", charset="utf8mb4")
BATCH_SIZE = 50000
DEAL_DIR = r"E:\forex_data\userDealByHands"

_parser = argparse.ArgumentParser(description="Data sync: local CSV -> MySQL candles (parameterized by symbol)")
_parser.add_argument("--symbol", default="EURUSD")
_args = _parser.parse_args()
SYMBOL = _args.symbol
CLEANED_DIR = rf"E:\forex_data\cleaned\{SYMBOL}"

TF_MAP = {
    "candles_5m":  f"{SYMBOL}bo5.csv",
    "candles_15m": f"{SYMBOL}bo15.csv",
    "candles_30m": f"{SYMBOL}bo30.csv",
    "candles_1h":  f"{SYMBOL}bo60.csv",
}


def import_csv(filepath, table, clear_before=False):
    """Import a single CSV file into a MySQL table.
    CSV format: YYYY.MM.DD,HH:MM,Open,High,Low,Close,Volume (no header)
    """
    fname = os.path.basename(filepath)
    conn = pymysql.connect(**MYSQL, cursorclass=DictCursor)
    cur = conn.cursor()

    if clear_before:
        cur.execute(f"DELETE FROM {table} WHERE symbol=%s", (SYMBOL,))
        conn.commit()

    rows = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for r in csv.reader(f):
            if len(r) < 7: continue
            ts = f"{r[0].replace('.','-')} {r[1]}:00"
            try:
                o, h, l, c, v = float(r[2]), float(r[3]), float(r[4]), float(r[5]), int(r[6])
            except: continue
            rows.append((ts, SYMBOL, o, h, l, c, v))

    n = len(rows)
    sql = f"INSERT IGNORE INTO {table} (timestamp,symbol,open,high,low,close,tick_volume) VALUES (%s,%s,%s,%s,%s,%s,%s)"
    for i in range(0, n, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        cur.executemany(sql, batch)
        conn.commit()
        pct = min(100, (i + len(batch)) * 100 // n)
        print(f"  {fname:40s} {i+len(batch):>8,}/{n:>8,} ({pct:3d}%)", end="\r")
    print(f"  {fname:40s} DONE {n:>8,} rows")

    cur.close(); conn.close()
    return n


# ============================================================
# STEP 1: Import M1 from cleaned histdata (2015~2026-04)
# ============================================================
print("=" * 60)
print("STEP 1: Import M1 from cleaned histdata (2015~2026-04)")
print("=" * 60)

m1_files = []
_m1_prefix = f"DAT_MT_{SYMBOL}_M1_"
for root, dirs, fnames in os.walk(CLEANED_DIR):
    for fn in fnames:
        if _m1_prefix in fn and fn.endswith(".csv"):
            try:
                yr = int(fn.replace(_m1_prefix, "").replace(".csv", "")[:4])
                if yr >= 2015:
                    m1_files.append(os.path.join(root, fn))
            except: pass
m1_files.sort()
print(f"Found {len(m1_files)} M1 files: {[os.path.basename(f) for f in m1_files]}")

# Import M1 files (INSERT IGNORE — safe to re-run, skips existing rows)
total_m1 = 0
for fp in m1_files:
    total_m1 += import_csv(fp, "candles_1m")
print(f"\nM1 total: {total_m1:,} rows\n")


# ============================================================
# STEP 2: Resample 1m → 5m/15m/30m/1h
# ============================================================
print("=" * 60)
print("STEP 2: Resample M1 → higher timeframes")
print("=" * 60)

import pandas as pd

conn = pymysql.connect(**MYSQL)
df = pd.read_sql(
    f"SELECT timestamp, open, high, low, close, tick_volume FROM candles_1m WHERE symbol='{SYMBOL}' ORDER BY timestamp",
    conn, parse_dates=['timestamp']
)
df.set_index('timestamp', inplace=True)
if df.empty:
    print(f"No M1 data for {SYMBOL} (cleaned dir: {CLEANED_DIR}) — nothing to resample")
    sys.exit(0)
print(f"Loaded {len(df):,} M1 bars: {df.index[0]} ~ {df.index[-1]}")

TF_RESAMPLE = {'candles_5m': 5, 'candles_15m': 15, 'candles_30m': 30, 'candles_1h': 60}

for table, tf_min in TF_RESAMPLE.items():
    r = df.resample(f'{tf_min}min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
    }).dropna()
    print(f"  {table}: {len(r):,} candles")

    cur = conn.cursor()

    sql = f"REPLACE INTO {table} (timestamp,symbol,open,high,low,close,tick_volume) VALUES (%s,%s,%s,%s,%s,%s,%s)"
    batch = []
    for idx, row in r.iterrows():
        batch.append((idx.strftime('%Y-%m-%d %H:%M:%S'), SYMBOL, float(row['open']), float(row['high']), float(row['low']), float(row['close']), int(row['tick_volume'])))
        if len(batch) >= BATCH_SIZE:
            cur.executemany(sql, batch); conn.commit(); batch = []
    if batch:
        cur.executemany(sql, batch); conn.commit()
    cur.close()
    print(f"    wrote OK")

conn.close()
print("Resample done\n")


# ============================================================
# STEP 3: Fill gap from userDealByHands (2026-04 ~ today)
# ============================================================
print("=" * 60)
print("STEP 3: Fill recent gap from userDealByHands CSV")
print("=" * 60)

# Get latest timestamp from each table to determine if gap filling is needed
conn = pymysql.connect(**MYSQL, cursorclass=DictCursor)
cur = conn.cursor()

for table, csv_name in {**TF_MAP, "candles_1m": f"{SYMBOL}bo1.csv"}.items():
    filepath = os.path.join(DEAL_DIR, csv_name)
    if not os.path.exists(filepath):
        print(f"  SKIP {table}: {csv_name} not found")
        continue

    cur.execute(f"SELECT MAX(timestamp) mx FROM {table} WHERE symbol=%s", (SYMBOL,))
    db_max = cur.fetchone()["mx"]
    print(f"  {table}: DB latest={str(db_max)[:16]}")

    # Import the CSV as-is (INSERT IGNORE handles overlap)
    n = import_csv(filepath, table)
    print(f"  {table}: +{n:,} candidate rows from {csv_name}")

cur.close(); conn.close()

# ============================================================
# FINAL CHECK
# ============================================================
print("\n" + "=" * 60)
print("FINAL STATE")
print("=" * 60)

conn = pymysql.connect(**MYSQL, cursorclass=DictCursor)
cur = conn.cursor()
for table in ['candles_1m','candles_5m','candles_15m','candles_30m','candles_1h']:
    cur.execute(f"SELECT COUNT(*) c, MIN(timestamp) mn, MAX(timestamp) mx FROM {table} WHERE symbol=%s", (SYMBOL,))
    r = cur.fetchone()
    print(f"  {table:15s} {r['c']:>10,} rows  {str(r['mn'])[:16]} ~ {str(r['mx'])[:16]}")
cur.close(); conn.close()
print("\nDONE")
