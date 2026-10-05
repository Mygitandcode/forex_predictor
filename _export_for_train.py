"""Export features_a + features_b → data_store/EURUSD/ numpy files (multi-threaded)"""
import pymysql, numpy as np, os
from pymysql.cursors import DictCursor
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

MYSQL_CONF = dict(host="localhost",port=3305,user="root",password="",database="forex_predictor",charset="utf8mb4")
DST = "data_store/EURUSD"
WORKERS = 4

# Get column names (shared)
conn = pymysql.connect(**MYSQL_CONF)
cur = conn.cursor()
def get_cols(tbl):
    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_schema='forex_predictor' AND table_name='{tbl}' AND column_name NOT IN ('timestamp','symbol','target_15m') ORDER BY ordinal_position")
    return [r[0] for r in cur.fetchall()]
cols_a = get_cols("features_a")
cols_b = get_cols("features_b")
cur.close(); conn.close()
print(f"Cols: a={len(cols_a)}, b={len(cols_b)}, total={len(cols_a)+len(cols_b)}")

# Count
conn = pymysql.connect(**MYSQL_CONF, cursorclass=DictCursor)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) c, MIN(timestamp) mn, MAX(timestamp) mx FROM features_a")
r = cur.fetchone()
n_total = r["c"]; t_min = str(r["mn"]); t_max = str(r["mx"])
cur.close(); conn.close()
print(f"Rows: {n_total:,}  {t_min} ~ {t_max}")

# Parallel read by time range
chunk_size = n_total // WORKERS
conn = pymysql.connect(**MYSQL_CONF, cursorclass=DictCursor); cur = conn.cursor()
boundaries = [t_min]
for i in range(1, WORKERS):
    cur.execute(f"SELECT timestamp FROM features_a ORDER BY timestamp LIMIT 1 OFFSET {i * chunk_size}")
    boundaries.append(str(cur.fetchone()["timestamp"]))
boundaries.append("2099-12-31 23:59:59")  # future date for last chunk
cur.close(); conn.close()
ranges = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

print(f"Ranges: {len(ranges)} chunks")

lock = threading.Lock()
results = {}  # {idx: (features, labels, timestamps)}

def read_range(idx, t_from, t_to):
    conn = pymysql.connect(**MYSQL_CONF, cursorclass=DictCursor)
    cur = conn.cursor()
    feats, labs, times = [], [], []
    cur.execute(f"""
        SELECT a.timestamp, a.target_15m, {','.join(['a.' + c for c in cols_a])}, {','.join(['b.' + c for c in cols_b])}
        FROM features_a a JOIN features_b b ON a.timestamp=b.timestamp
        WHERE a.timestamp >= %s AND a.timestamp < %s
        ORDER BY a.timestamp
    """, (t_from, t_to))
    for r in cur.fetchall():
        vals = [float(r[c] if r[c] is not None else 0) for c in cols_a]
        vals += [float(r[c] if r[c] is not None else 0) for c in cols_b]
        feats.append(vals)
        labs.append(int(r["target_15m"]))
        times.append(str(r["timestamp"]))
    cur.close(); conn.close()
    with lock:
        results[idx] = (feats, labs, times)
        done = sum(len(v[0]) for v in results.values())
        print(f"  chunk {idx+1}/{len(ranges)}: {len(feats):,} rows, total {done:,}/{n_total:,}", flush=True)

# Run
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = {ex.submit(read_range, i, r[0], r[1]): i for i, r in enumerate(ranges)}
    for f in as_completed(futures):
        f.result()

# Merge in order
print("Merging...")
all_feats, all_labs, all_times = [], [], []
for i in sorted(results.keys()):
    f, l, t = results[i]
    all_feats.extend(f); all_labs.extend(l); all_times.extend(t)

X = np.array(all_feats, dtype=np.float32)
y = np.array(all_labs, dtype=np.int8)
t_arr = np.array(all_times)
print(f"X={X.shape}, y={y.shape}")

os.makedirs(DST, exist_ok=True)
done = os.path.join(DST, ".done")
if os.path.exists(done): os.remove(done)
np.save(os.path.join(DST, "features.npy"), X)
np.save(os.path.join(DST, "labels_15m.npy"), y)
np.save(os.path.join(DST, "timestamps.npy"), t_arr)
np.save(os.path.join(DST, "weights.npy"), np.ones(len(y), dtype=np.float32))
Path(done).touch()
print(f"Saved to {DST}/ — DONE")
