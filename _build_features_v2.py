"""
多 TF 联合特征构建（每 1m 行 × 5 层 TF 特征）
==============================================
每行: 1m(212维) + 5m(212维) + 15m(212维) + 30m(212维) + 1h(212维) ≈ 1060维
高 TF 特征使用未闭合蜡烛的当前累积状态（与 MT4 实盘一致）。
"""
import sys, argparse
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import pymysql, pandas as pd, numpy as np
from pymysql.cursors import DictCursor
from features.feature_builder import FeatureBuilder

MYSQL = dict(host="localhost",port=3305,user="root",password="",database="forex_predictor",charset="utf8mb4")
BATCH = 5000
WARMUP = 200  # FeatureBuilder needs ~200 bars for indicator warmup
TF_MAP = {'5min':5, '15min':15, '30min':30, '1h':60}


def build_many_safe(builder, df, label=""):
    """Run build_many with progress logging"""
    if len(df) < WARMUP:
        return np.array([]).reshape(0, len(builder.feature_names or [])), []
    print(f"  {label}: {len(df):,} rows → ", end="", flush=True)
    X, names = builder.build_many(df)
    print(f"{X.shape[0]:,} features", flush=True)
    return X, names


def make_synthetic_tf(df_1m, tf_minutes, builder):
    """
    从 1m 蜡烛构造高 TF 的 synthetic DataFrame。
    每行 = 当前 1m 时刻该 TF bar 的累积状态 (O/H/L/C/V)。
    """
    tf_label = f"{tf_minutes}min"
    bucket = df_1m.index.floor(tf_label)

    syn = pd.DataFrame(index=df_1m.index)
    syn['open'] = df_1m.groupby(bucket)['open'].transform('first')
    syn['high'] = df_1m.groupby(bucket)['high'].cummax()
    syn['low'] = df_1m.groupby(bucket)['low'].cummin()
    syn['close'] = df_1m['close']
    syn['tick_volume'] = df_1m.groupby(bucket)['tick_volume'].cumsum()
    syn['time'] = syn.index

    return syn


def build_all_tf(date_from, date_to=None, test_only=False, symbol="EURUSD", point_size=0.00001):
    """Build ALL 5 TF features for a date range → INSERT into `features` table"""
    # ---- Load 1m candles ----
    conn = pymysql.connect(**MYSQL)
    where = f"WHERE symbol='{symbol}' AND timestamp >= '{date_from}'"
    if date_to:
        where += f" AND timestamp < '{date_to}'"
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, tick_volume FROM candles_1m {where} ORDER BY timestamp",
        conn, parse_dates=['timestamp']
    )
    conn.close()
    if df.empty:
        print(f"No candles for symbol={symbol} in [{date_from}, {date_to or 'now'}) — skip build", flush=True)
        return 0
    print(f"Loaded {len(df):,} 1m candles: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}", flush=True)

    df.set_index('timestamp', inplace=True)
    df['time'] = df.index
    n_total = len(df)

    # ---- 1m 特征 ----
    builder_1m = FeatureBuilder(symbol=symbol, point_size=point_size)
    X_1m, names_1m = build_many_safe(builder_1m, df, "1m")
    n_feat = len(names_1m)

    # ---- 合成高 TF 特征 ----
    tf_data = {}  # {tf_name: X_array}
    for tf_name, tf_min in TF_MAP.items():
        syn = make_synthetic_tf(df, tf_min, builder_1m)
        builder_tf = FeatureBuilder(symbol=symbol, point_size=point_size)
        X_tf, names_tf = build_many_safe(builder_tf, syn, tf_name)
        tf_data[tf_name] = X_tf

    # ---- 对齐行数（各 TF 的 build_many 输出行数 = n_total - WARMUP）----
    expected_rows = max(0, n_total - WARMUP)
    for key, arr in [('1m', X_1m)] + [(k, v) for k, v in tf_data.items()]:
        if len(arr) != expected_rows and len(arr) > 0:
            print(f"  WARNING: {key} has {len(arr)} rows, expected {expected_rows}")

    # ---- 拆成两张表 (MariaDB InnoDB 上限 ~1000 列) ----
    # features_a: 1m(212) + 5m(212) = 424 cols
    # features_b: 15m(212) + 30m(212) + 1h(212) = 636 cols
    def safe_col(prefix, name):
        return prefix + '_' + name.replace(' ','_').replace('-','_').replace('(','').replace(')','').replace('/','_')

    tables = {
        'features_a': ['1m', '5min'],
        'features_b': ['15min', '30min', '1h'],
    }

    base_sql = "CREATE TABLE IF NOT EXISTS {tbl} (timestamp DATETIME NOT NULL, symbol VARCHAR(10) NOT NULL, {cols}, target_15m TINYINT, PRIMARY KEY (timestamp, symbol), INDEX idx_symbol (symbol)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"

    conn = pymysql.connect(**MYSQL)
    cur = conn.cursor()
    for tbl_name, prefixes in tables.items():
        cols = [f"`{safe_col(p, n)}` FLOAT" for p in prefixes for n in names_1m]
        ddl = base_sql.format(tbl=tbl_name, cols=", ".join(cols))
        cur.execute(ddl)
        # 多 symbol 共存：仅清除本 symbol 旧段，保留其他 symbol 数据。
        # 列一致性防护：CREATE IF NOT EXISTS 不会更新旧表结构，特征定义演进后须显式报错而非静默漏列
        cur.execute("""SELECT column_name FROM information_schema.columns
            WHERE table_schema=DATABASE() AND table_name=%s""", (tbl_name,))
        actual = {r[0] for r in cur.fetchall()}
        expected = {'timestamp', 'symbol', 'target_15m'} | {c.split('`')[1] for c in cols}
        if actual != expected:
            missing = sorted(expected - actual)[:5]
            extra = sorted(actual - expected)[:5]
            raise RuntimeError(
                f"{tbl_name} 列结构与当前特征定义不一致 (missing={missing}, extra={extra}) — "
                f"特征定义已演进，需人工评估 DROP 重建")
        cur.execute(f"DELETE FROM {tbl_name} WHERE symbol=%s", (symbol,))
        conn.commit()
        print(f"  {tbl_name}: {len(cols)} cols (rebuild segment symbol={symbol})", flush=True)
    cur.close(); conn.close()

    # ---- 批量 INSERT 到两张表 ----
    # Align data
    n_rows = min(len(arr) for arr in ([X_1m] + list(tf_data.values())) if len(arr) > 0)
    if n_rows == 0:
        print("No feature rows to insert")
        return 0

    conn = pymysql.connect(**MYSQL, cursorclass=DictCursor)
    cur = conn.cursor()
    total = 0

    for tbl_name, prefixes in tables.items():
        cols_names = [safe_col(p, n) for p in prefixes for n in names_1m]
        cols_sql = ", ".join([f"`{c}`" for c in cols_names])
        ph = ",".join(["%s"] * len(cols_names))
        sql = f"INSERT IGNORE INTO {tbl_name} (timestamp, symbol, {cols_sql}, target_15m) VALUES (%s, %s, {ph}, %s)"

        batch = []
        for i in range(n_rows):
            real_idx = i + WARMUP
            ts = df.index[real_idx].strftime('%Y-%m-%d %H:%M:%S')
            vals = [float(v) for p in prefixes for v in tf_data.get(p, X_1m)[i]] if '1m' in prefixes else [float(v) for p in prefixes for v in tf_data[p][i]]
            # For features_a (1m+5m):
            if tbl_name == 'features_a':
                vals = [float(v) for v in X_1m[i]] + [float(v) for v in tf_data['5min'][i]]
            else:
                vals = [float(v) for v in tf_data['15min'][i]] + [float(v) for v in tf_data['30min'][i]] + [float(v) for v in tf_data['1h'][i]]

            future_idx = real_idx + 15
            target = 1 if future_idx < n_total and df['close'].iloc[future_idx] > df['close'].iloc[real_idx] else (0 if future_idx < n_total else -1)
            batch.append((ts, symbol) + tuple(vals) + (target,))
            if len(batch) >= BATCH:
                cur.executemany(sql, batch)
                conn.commit()
                total += len(batch)
                print(f"  {tbl_name}: {total:,}/{n_rows:,}", end="\r", flush=True)
                batch = []
        if batch:
            cur.executemany(sql, batch)
            conn.commit()
            total = 0
        print(f"\n  {tbl_name}: done", flush=True)

    for tbl_name in tables:
        cur.execute(f"SELECT COUNT(*) c, MIN(timestamp) mn, MAX(timestamp) mx FROM {tbl_name} WHERE symbol=%s", (symbol,))
        r = cur.fetchone()
        print(f"  {tbl_name} [{symbol}]: {r['c']:,} rows, {r['mn']} ~ {r['mx']}", flush=True)

    cur.close(); conn.close()
    return n_rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["test_day","test_week","full"], default="test_day")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--point-size", type=float, default=0.00001,
                   help="报价最小变动点（5位小数对=0.00001，JPY报价对=0.001）")
    args = p.parse_args()

    if args.mode == "test_day":
        print("=== TEST: 1 day (2026-05-20) ===")
        n = build_all_tf("2026-05-20", "2026-05-21", symbol=args.symbol, point_size=args.point_size)
        print(f"DONE: {n:,} rows\n")

    elif args.mode == "test_week":
        print("=== TEST: 1 week (2026-05-01 ~ 2026-05-08) ===")
        n = build_all_tf("2026-05-01", "2026-05-08", symbol=args.symbol, point_size=args.point_size)
        print(f"DONE: {n:,} rows\n")

    elif args.mode == "full":
        print("=== FULL: 2015-01-01 ~ today ===")
        n = build_all_tf("2015-01-01", symbol=args.symbol, point_size=args.point_size)
        print(f"DONE: {n:,} rows\n")
        # Register feature version
        conn = pymysql.connect(**MYSQL, cursorclass=DictCursor)
        cur = conn.cursor()
        cur.execute("""INSERT INTO feature_versions (version_key,dimension,feature_count,status,description,build_module)
            VALUES (%s,%s,%s,'active','multi-TF joint wide table (5 TF x 212 dim)','_build_features_v2.py')
            ON DUPLICATE KEY UPDATE updated_at=NOW()""",
            ("20260521_mtf", 1060, 1060))
        conn.commit(); cur.close(); conn.close()
        print("feature_versions registered: 20260521_mtf")
