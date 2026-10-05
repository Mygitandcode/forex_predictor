import pymysql; from pymysql.cursors import DictCursor
c=pymysql.connect(host="localhost",port=3305,user="root",password="",database="forex_predictor",charset="utf8mb4",cursorclass=DictCursor)
cur=c.cursor()
for tf in ['1h','30m','15m','5m']:
    tbl = f"features_{tf}"
    cur.execute(f"SELECT COUNT(*) c, MIN(timestamp) mn, MAX(timestamp) mx FROM {tbl}")
    r=cur.fetchone()
    print(f"features_{tf}: {r['c']:>8,} rows  {str(r['mn'])[:16]} ~ {str(r['mx'])[:16]}")
    # Check macd column exists
    cur.execute(f"SELECT COUNT(*) c FROM {tbl} WHERE rsi_14 IS NOT NULL")
    print(f"  rsi_14 non-null: {cur.fetchone()['c']:,}")
cur.close();c.close()
