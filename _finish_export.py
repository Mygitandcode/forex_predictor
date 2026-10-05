"""补全导出元数据：timestamps/weights/.json（_recover_export 之后用）

游标分页流式写入，避免 418 万行全量内存加载。
"""
import numpy as np, os, json, pymysql
from datetime import datetime

DST = "E:/forex_npy/EURUSD"
VERSION_KEY = "20260521_mtf"
BATCH = 50000
MYSQL = dict(host="localhost", port=3305, user="root", password="",
             database="forex_predictor", charset="utf8mb4",
             connect_timeout=10, read_timeout=600)

# 1. 行数
conn = pymysql.connect(**MYSQL)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM features_a")
n = cur.fetchone()[0]
cur.close(); conn.close()
print(f"Rows: {n:,}")

# 2. 时间戳 — 游标分页，逐段写入 mmap
ts_mmap_path = os.path.join(DST, "tmp.timestamps.mmap")
ts = np.memmap(ts_mmap_path, dtype='U26', mode='w+', shape=(n,))  # U26 = '2026-05-20 17:49:00'
written = 0
last_ts = "0000-01-01 00:00:00"

conn = pymysql.connect(**MYSQL)
cur = conn.cursor()
print("Reading timestamps (cursor pagination)...")

while True:
    cur.execute(f"""SELECT timestamp FROM features_a
        WHERE timestamp > %s ORDER BY timestamp LIMIT {BATCH}""", (last_ts,))
    rows = cur.fetchall()
    if not rows:
        break
    chunk = np.array([str(r[0]) for r in rows], dtype='U26')
    end = written + len(chunk)
    ts[written:end] = chunk
    written = end
    last_ts = str(rows[-1][0])
    print(f"  timestamps: {written:,}/{n:,} ({written*100//n}%)", flush=True)

cur.close(); conn.close()
ts.flush()
t_min, t_max = str(ts[0]), str(ts[-1])
print(f"  Range: {t_min} ~ {t_max}")

# 3. 列名
conn = pymysql.connect(**MYSQL)
cur = conn.cursor()
for tbl in ["features_a", "features_b"]:
    cur.execute(f"""SELECT column_name FROM information_schema.columns
        WHERE table_schema='forex_predictor' AND table_name='{tbl}'
        AND column_name NOT IN ('timestamp','symbol','target_15m')
        ORDER BY ordinal_position""")
    fetched = [r[0] for r in cur.fetchall()]
    if tbl == "features_a":
        cols_a = fetched
    else:
        cols_b = fetched
cur.close(); conn.close()

feature_names = cols_a + cols_b
n_cols = len(feature_names)
print(f"Features: {len(cols_a)}+{len(cols_b)}={n_cols} cols")

# 4. 写 timestamps.npy（从 mmap 转存，释放 mmap 后删临时文件）
print("Saving timestamps.npy ...")
np.save(os.path.join(DST, "timestamps.npy"), ts)
del ts  # 释放 memmap 文件句柄，否则 Windows 下 os.remove 会 PermissionError
os.remove(ts_mmap_path)

# 5. 写入其他文件
np.save(os.path.join(DST, "weights.npy"), np.ones(n, dtype=np.float32))
print("weights.npy saved")

with open(os.path.join(DST, "feature_names.json"), "w", encoding="utf-8") as f:
    json.dump(feature_names, f, ensure_ascii=False, indent=2)

meta = {
    "version_key": VERSION_KEY,
    "n_rows": n, "n_cols": n_cols,
    "selected_features": None,
    "feature_names": feature_names,
    "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "t_min": t_min, "t_max": t_max,
}
with open(os.path.join(DST, "export_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("feature_names.json + export_meta.json saved")

# 6. 验证
X = np.load(os.path.join(DST, "features.npy"), mmap_mode='r')
print(f"Verify: X={X.shape}, n_cols={n_cols} {'OK' if X.shape[1] == n_cols else 'MISMATCH!'}")
del X
print("DONE")
