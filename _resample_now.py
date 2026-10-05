"""Resample candles_1m -> 5m/15m/30m/1h for EURUSD"""
import pymysql, sys
import pandas as pd

MYSQL = dict(host="localhost",port=3305,user="root",password="",database="forex_predictor",charset="utf8mb4")
SYMBOL = "EURUSD"
BATCH = 50000

print("Loading 1m data...", flush=True)
conn = pymysql.connect(**MYSQL)
df = pd.read_sql("SELECT timestamp,open,high,low,close,tick_volume FROM candles_1m WHERE symbol='EURUSD' ORDER BY timestamp", conn, parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
print(f"Loaded {len(df):,} M1 bars: {df.index[0]} ~ {df.index[-1]}", flush=True)

TF_MAP = {'candles_5m':5, 'candles_15m':15, 'candles_30m':30, 'candles_1h':60}

for table, tf_min in TF_MAP.items():
    print(f"\nResampling {table} ({tf_min}min)...", flush=True)
    r = df.resample(f'{tf_min}min').agg({'open':'first','high':'max','low':'min','close':'last','tick_volume':'sum'}).dropna()
    print(f"  Result: {len(r):,} candles", flush=True)

    cur = conn.cursor()
    cur.execute(f"DELETE FROM {table} WHERE symbol=%s", (SYMBOL,))
    conn.commit()
    print(f"  Cleared old data", flush=True)

    sql = f"INSERT INTO {table} (timestamp,symbol,open,high,low,close,tick_volume) VALUES (%s,%s,%s,%s,%s,%s,%s)"
    batch = []
    total = 0
    for idx, row in r.iterrows():
        batch.append((idx.strftime('%Y-%m-%d %H:%M:%S'), SYMBOL, float(row['open']), float(row['high']), float(row['low']), float(row['close']), int(row['tick_volume'])))
        if len(batch) >= BATCH:
            cur.executemany(sql, batch); conn.commit()
            total += len(batch)
            print(f"    {total:,}/{len(r):,}", end="\r", flush=True)
            batch = []
    if batch:
        cur.executemany(sql, batch); conn.commit()
        total += len(batch)
    cur.close()
    print(f"  {table}: {total:,} rows written", flush=True)

# Fill gap to today from userDealByHands
print("\nFilling gap to today from userDealByHands...", flush=True)
import csv, os
DEAL = r"E:\forex_data\userDealByHands"
for table, csv_name in {'candles_5m':'EURUSDbo5.csv','candles_15m':'EURUSDbo15.csv','candles_30m':'EURUSDbo30.csv','candles_1h':'EURUSDbo60.csv'}.items():
    fp = os.path.join(DEAL, csv_name)
    if not os.path.exists(fp): continue
    rows = []
    with open(fp, 'r') as f:
        for r in csv.reader(f):
            if len(r) < 7: continue
            ts = f"{r[0].replace('.','-')} {r[1]}:00"
            try: o,h,l,c,v = float(r[2]),float(r[3]),float(r[4]),float(r[5]),int(r[6])
            except: continue
            rows.append((ts,SYMBOL,o,h,l,c,v))
    cur = conn.cursor()
    sql = f"INSERT IGNORE INTO {table} (timestamp,symbol,open,high,low,close,tick_volume) VALUES (%s,%s,%s,%s,%s,%s,%s)"
    for i in range(0, len(rows), BATCH):
        cur.executemany(sql, rows[i:i+BATCH]); conn.commit()
    cur.close()
    print(f"  {table}: +{len(rows)} candidate, latest from userDealByHands", flush=True)

conn.close()
print("\nDONE — all timeframes complete with gap fill", flush=True)
