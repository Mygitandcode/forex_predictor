"""Resample candles_1m -> 5m/15m/30m/1h"""
import pymysql, pandas as pd

conn = pymysql.connect(host='localhost',port=3305,user='root',password='',database='forex_predictor',charset='utf8mb4')

symbols = pd.read_sql("SELECT DISTINCT symbol FROM candles_1m", conn)['symbol'].tolist()
print(f"Symbols: {symbols}")

TF_MAP = {'5m':5,'15m':15,'30m':30,'1h':60}

for sym in symbols:
    df = pd.read_sql(f"SELECT timestamp,open,high,low,close,tick_volume FROM candles_1m WHERE symbol='{sym}' ORDER BY timestamp", conn, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    print(f"{sym}: {len(df)} rows, {df.index[0]} ~ {df.index[-1]}")

    for tf_name, tf_min in TF_MAP.items():
        r = df.resample(f'{tf_min}min').agg({'open':'first','high':'max','low':'min','close':'last','tick_volume':'sum'}).dropna()
        print(f"  {tf_name}: {len(r)} candles")

        cur = conn.cursor()
        cur.execute(f"DELETE FROM candles_{tf_name} WHERE symbol=%s", (sym,))
        conn.commit()

        sql = f"INSERT INTO candles_{tf_name} (timestamp,symbol,open,high,low,close,tick_volume) VALUES (%s,%s,%s,%s,%s,%s,%s)"
        batch = []
        for idx, row in r.iterrows():
            batch.append((idx.strftime('%Y-%m-%d %H:%M:%S'), sym, float(row['open']), float(row['high']), float(row['low']), float(row['close']), int(row['tick_volume'])))
            if len(batch) >= 5000:
                cur.executemany(sql, batch); conn.commit(); batch = []
        if batch:
            cur.executemany(sql, batch); conn.commit()
        cur.close()
        print(f"    wrote OK")

conn.close()
print("DONE")
