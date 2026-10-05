"""
特征构建（固定蜡烛快照版，支持多 TF）
使用: python _build_features.py --tf 15m --mode test_week|test_month|full
"""
import sys, argparse
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import pymysql, pandas as pd, numpy as np
from pymysql.cursors import DictCursor
from features.feature_builder import FeatureBuilder

MYSQL = dict(host="localhost",port=3305,user="root",password="",database="forex_predictor",charset="utf8mb4")
SYMBOL = "EURUSD"
BATCH = 5000


def build_and_insert(tf, date_from, date_to=None, clear_before=True):
    """Build features for a date range and insert into features_{tf}"""
    candle_table = f"candles_{tf}"
    feature_table = f"features_{tf}"
    tf_num = tf.replace('m','').replace('h','60')  # for target column

    # Load candles (regular cursor for pandas)
    conn = pymysql.connect(**MYSQL)
    where = f"WHERE symbol='{SYMBOL}' AND timestamp >= '{date_from}'"
    if date_to:
        where += f" AND timestamp < '{date_to}'"
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, tick_volume FROM {candle_table} {where} ORDER BY timestamp",
        conn, parse_dates=['timestamp']
    )
    conn.close()
    print(f"  Loaded {len(df):,} candles: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}", flush=True)

    if len(df) < 100:
        print("  ERROR: need >= 100 bars")
        return 0

    df.set_index('timestamp', inplace=True)
    df['time'] = df.index

    # Build features
    builder = FeatureBuilder(symbol=SYMBOL, point_size=0.00001)
    X, feature_names = builder.build_many(df)
    print(f"  Features: {X.shape[0]:,} rows x {X.shape[1]} dims", flush=True)

    # Prepare INSERT
    safe_names = [n.replace(' ','_').replace('-','_').replace('(','').replace(')','').replace('/','_') for n in feature_names]
    cols = ", ".join([f"`{sn}`" for sn in safe_names])
    placeholders = ",".join(["%s"] * (len(feature_names) + 1))
    target_col = f"target_{tf}" if tf != '1h' else "target_1h"
    sql = f"INSERT INTO {feature_table} (timestamp, symbol, {cols}, {target_col}) VALUES (%s, %s, {placeholders})"

    # Clear old data
    conn = pymysql.connect(**MYSQL, cursorclass=DictCursor)
    cur = conn.cursor()
    if clear_before:
        del_sql = f"DELETE FROM {feature_table} WHERE symbol='{SYMBOL}'"
        if date_to:
            del_sql += f" AND timestamp >= '{date_from}' AND timestamp < '{date_to}'"
        cur.execute(del_sql)
        conn.commit()

    # Batch insert
    batch = []
    total = 0
    for i in range(len(X)):
        ts = df.index[i].strftime('%Y-%m-%d %H:%M:%S')
        vals = [float(v) for v in X[i]] + [-1]
        batch.append((ts, SYMBOL) + tuple(vals))
        if len(batch) >= BATCH:
            cur.executemany(sql, batch)
            conn.commit()
            total += len(batch)
            print(f"    {total:,}/{X.shape[0]:,}", end="\r", flush=True)
            batch = []
    if batch:
        cur.executemany(sql, batch)
        conn.commit()
        total += len(batch)

    # Verify
    cur.execute(f"SELECT COUNT(*) c, MIN(timestamp) mn, MAX(timestamp) mx FROM {feature_table}")
    r = cur.fetchone()
    print(f"\n  {feature_table}: {r['c']:,} rows, {r['mn']} ~ {r['mx']}", flush=True)
    cur.close(); conn.close()
    return total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tf", default="15m", help="1m|5m|15m|30m|1h")
    p.add_argument("--mode", choices=["test_week","test_month","full"], default="test_week")
    args = p.parse_args()

    if args.mode == "test_week":
        # 1h needs 4 weeks for indicator warmup
        end_date = "2026-05-20" if args.tf == "1h" else "2026-05-08"
        print(f"=== {args.tf} TEST: {'4 weeks' if args.tf=='1h' else '1 week'} ===")
        n = build_and_insert(args.tf, "2026-05-01", end_date, clear_before=True)
        print(f"DONE: {n:,} rows\n")

    elif args.mode == "test_month":
        print(f"=== {args.tf} TEST: 1 month ===")
        n = build_and_insert(args.tf, "2026-04-01", "2026-05-01", clear_before=True)
        print(f"DONE: {n:,} rows\n")

    elif args.mode == "full":
        print(f"=== {args.tf} FULL: 2015-01-01 ~ today ===")
        n = build_and_insert(args.tf, "2015-01-01", clear_before=True)
        print(f"DONE: {n:,} rows\n")
        if args.tf == "15m":
            conn = pymysql.connect(**MYSQL, cursorclass=DictCursor)
            cur = conn.cursor()
            cur.execute("""INSERT INTO feature_versions (version_key,dimension,feature_count,status,description,build_module)
                VALUES (%s,212,212,'active','212-dim fixed candle snapshot','feature_sets/212_snapshot.py')
                ON DUPLICATE KEY UPDATE updated_at=NOW()""", ("20260521_212",))
            conn.commit(); cur.close(); conn.close()
            print("feature_versions registered: 20260521_212")
