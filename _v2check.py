import pymysql; from pymysql.cursors import DictCursor
c=pymysql.connect(host="localhost",port=3305,user="root",password="",database="forex_predictor",charset="utf8mb4",cursorclass=DictCursor)
cur=c.cursor()
for t in ["features_a","features_b"]:
    cur.execute(f"SELECT COUNT(*) c, MIN(timestamp) mn, MAX(timestamp) mx FROM {t}")
    r=cur.fetchone(); print(f"{t}: {r['c']} rows, {r['mn']} ~ {r['mx']}")
    cur.execute(f"SELECT * FROM {t} LIMIT 1")
    s=cur.fetchone(); nn=sum(1 for v in s.values() if v is not None)
    print(f"  sample: {nn}/{len(s)} non-null cols")
    cur.execute(f"SELECT COUNT(*) c FROM {t} WHERE target_15m >= 0")
    print(f"  target valid: {cur.fetchone()['c']}")
c.close()
