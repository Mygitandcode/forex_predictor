"""_mysql_probe.py — MySQL 连通性 + 建库建表 + 插入读取全链路测试

用法: python _mysql_probe.py
若失败: 检查 XAMPP MySQL 是否启动 (XAMPP Control Panel → MySQL Start)
"""
import pymysql
import sys
from datetime import datetime

HOST = "127.0.0.1"
PORT = 3305
USER = "root"
PASSWORD = ""
DB = "forex_predictor"

def connect(db=None):
    kwargs = dict(host=HOST, port=PORT, user=USER, password=PASSWORD,
                  connect_timeout=3, read_timeout=5, charset='utf8mb4')
    if db:
        kwargs['database'] = db
    return pymysql.connect(**kwargs)

print("=" * 60)
print("MySQL Connectivity Full-Link Test")
print(f"  {HOST}:{PORT}  user={USER or '(no password)'}")
print("=" * 60)

# Step 1 — Basic connectivity
try:
    conn = connect()
    print("Step 1 [OK] Basic connectivity")
except Exception as e:
    print(f"Step 1 [FAIL] Basic connectivity: {e}")
    print("\n-> Please start MySQL in XAMPP Control Panel first, then retry.")
    sys.exit(1)

cur = conn.cursor()
cur.execute("SHOW DATABASES")
dbs = [r[0] for r in cur.fetchall()]
print(f"  Existing databases: {dbs}")
cur.close()
conn.close()

# Step 2 — Create database
try:
    conn = connect()
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB} CHARACTER SET utf8mb4")
    conn.commit()
    cur.close()
    conn.close()
    print(f"Step 2 [OK] Database `{DB}` ready")
except Exception as e:
    print(f"Step 2 [FAIL] Create DB failed: {e}")
    sys.exit(1)

# Step 3 — Create test table in forex_predictor
try:
    conn = connect(DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS _probe_test (
            id        BIGINT AUTO_INCREMENT PRIMARY KEY,
            ts        DATETIME NOT NULL,
            val       FLOAT,
            tag       VARCHAR(50)
        ) ENGINE=InnoDB
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Step 3 [OK] Test table `_probe_test` ready")
except Exception as e:
    print(f"Step 3 [FAIL] Create table failed: {e}")
    sys.exit(1)

# Step 4 — Python insert data
try:
    conn = connect(DB)
    cur = conn.cursor()
    rows = [
        (datetime(2026, 5, 16, 14, 0), 1.1032, "insert_test_01"),
        (datetime(2026, 5, 16, 14, 1), 1.1035, "insert_test_02"),
        (datetime(2026, 5, 16, 14, 2), 1.1031, "insert_test_03"),
    ]
    cur.executemany(
        "INSERT INTO _probe_test (ts, val, tag) VALUES (%s, %s, %s)",
        rows
    )
    conn.commit()
    print(f"Step 4 [OK] Inserted {cur.rowcount} rows")
    cur.close()
    conn.close()
except Exception as e:
    print(f"Step 4 [FAIL] Insert failed: {e}")
    sys.exit(1)

# Step 5 — Python read data
try:
    conn = connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT id, ts, val, tag FROM _probe_test ORDER BY ts")
    result = cur.fetchall()
    print(f"Step 5 [OK] Read {len(result)} rows:")
    for r in result:
        print(f"  id={r[0]}  ts={r[1]}  val={r[2]}  tag={r[3]}")
    cur.close()
    conn.close()
except Exception as e:
    print(f"Step 5 [FAIL] Read failed: {e}")
    sys.exit(1)

# Step 6 — pandas read_sql (common pattern)
try:
    import pandas as pd
    conn = connect(DB)
    df = pd.read_sql("SELECT * FROM _probe_test ORDER BY ts", conn)
    conn.close()
    print(f"Step 6 [OK] pandas read_sql: {len(df)} rows, columns={list(df.columns)}")
    print(f"  val mean: {df['val'].mean():.4f}")
except Exception as e:
    print(f"Step 6 [WARN] pandas read failed (non-critical): {e}")

# Step 7 — Cleanup test table
try:
    conn = connect(DB)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS _probe_test")
    conn.commit()
    cur.close()
    conn.close()
    print("Step 7 [OK] Test table cleaned up")
except Exception as e:
    print(f"Step 7 [WARN] Cleanup failed (non-critical): {e}")

print("=" * 60)
print("Full-Link PASSED — MySQL is ready for use")
print(f"  Connection: {USER}@{HOST}:{PORT}/{DB}")
print("=" * 60)
