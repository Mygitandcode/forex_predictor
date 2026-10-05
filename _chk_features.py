import pymysql; from pymysql.cursors import DictCursor
c=pymysql.connect(host="localhost",port=3305,user="root",password="",database="forex_predictor",charset="utf8mb4",cursorclass=DictCursor)
cur=c.cursor()
cur.execute("SELECT COUNT(*) c, MIN(timestamp) mn, MAX(timestamp) mx FROM features_15m")
r=cur.fetchone(); print(f"features_15m: {r['c']} rows, {r['mn']} ~ {r['mx']}")
cur.execute("SELECT * FROM features_15m LIMIT 1")
s=cur.fetchone()
if s:
    nn=sum(1 for v in s.values() if v is not None)
    print(f"Sample: {nn}/{len(s)} non-null cols")
for col in ["macd","macd_signal","macd_hist","rsi_14"]:
    cur.execute(f"SELECT COUNT(*) c FROM features_15m WHERE `{col}` IS NOT NULL")
    print(f"  {col}: {cur.fetchone()['c']} non-null")
cur.close();c.close()
