import pymysql
from pymysql.cursors import DictCursor

conn = pymysql.connect(host="localhost",port=3305,user="root",password="",database="forex_predictor",charset="utf8mb4",cursorclass=DictCursor)
cur = conn.cursor()

for table in ['candles_1m','candles_5m','candles_15m','candles_30m','candles_1h']:
    cur.execute(f"SELECT COUNT(*) c, MIN(timestamp) mn, MAX(timestamp) mx FROM {table} WHERE symbol='EURUSD'")
    r = cur.fetchone()
    print(f"{table:15s} {r['c']:>10,} rows  {str(r['mn'])[:16]} ~ {str(r['mx'])[:16]}")

# Check for gaps in 1m (more than 2 min gap = issue)
cur.execute("SELECT timestamp FROM candles_1m WHERE symbol='EURUSD' ORDER BY timestamp")
rows = cur.fetchall()
gaps = 0
for i in range(1, len(rows)):
    t1 = rows[i-1]["timestamp"]
    t2 = rows[i]["timestamp"]
    diff = (t2 - t1).total_seconds()
    if diff > 120:  # more than 2 minutes
        if gaps < 10:
            print(f"  GAP: {t1} -> {t2} ({diff/60:.0f} min)")
        gaps += 1
print(f"Total gaps > 2min in 1m: {gaps}")

# Check higher TFs alignment
for table in ['candles_5m','candles_15m','candles_30m','candles_1h']:
    cur.execute(f"SELECT timestamp FROM {table} WHERE symbol='EURUSD' ORDER BY timestamp")
    rows2 = cur.fetchall()
    gaps2 = 0
    for i in range(1, len(rows2)):
        t1 = rows2[i-1]["timestamp"]
        t2 = rows2[i]["timestamp"]
        diff = (t2 - t1).total_seconds()
        expected = {"candles_5m":300,"candles_15m":900,"candles_30m":1800,"candles_1h":3600}
        exp = expected.get(table, 0)
        if diff > exp * 1.1:  # more than 10% extra
            if gaps2 < 5:
                print(f"  {table} GAP: {t1} -> {t2} ({diff/60:.0f} min)")
            gaps2 += 1
    if gaps2 == 0:
        print(f"  {table}: no gaps")

cur.close(); conn.close()
