"""Export features_a JOIN features_b -> numpy (cursor pagination, mmap, low memory)

Usage:
    # Export all 1060 features (default)
    python _export_v2.py

    # Export selected features from JSON
    python _export_v2.py --features E:/forex_npy/EURUSD/top_features.json

    # From Python (e.g. training pipeline)
    from _export_v2 import export_features
    meta = export_features(selected_features=["1m_rsi_14", "5min_macd"], dst_dir="E:/forex_npy/EURUSD")

约定: 只增特征不删特征。selected_features 为空 = SELECT ALL。
feature_names.json 会随 .npy 一起保存，用于追溯每个维度对应的特征名。

日志: 全部 JSON Lines(经 pipeline.jsonlog.jlog 输出到 stdout, 由 stage3 重定向到日志文件)。
"""

import pymysql, numpy as np, os, json, sys
from pymysql.cursors import DictCursor
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

from pipeline.jsonlog import jlog  # JSON Lines 日志(替代 print)

MYSQL = dict(host="localhost", port=3305, user="root", password="",
             database="forex_predictor", charset="utf8mb4",
             connect_timeout=10, read_timeout=600, write_timeout=600)
BATCH = 20000
REFRESH_EVERY = 100000  # reconnect every 100K rows to prevent timeout
VERSION_KEY = "20260521_mtf"  # 特征版本，来自 _build_features_v2.py


def _get_all_columns(conn, target="15m"):
    """获取 features_a 和 features_b 中所有特征列名"""
    label_col = f"target_{target}"
    cur = conn.cursor()
    cur.execute("""SELECT column_name FROM information_schema.columns
        WHERE table_schema='forex_predictor' AND table_name='features_a'
        AND column_name NOT IN ('timestamp','symbol',%s)
        ORDER BY ordinal_position""", (label_col,))
    cols_a = [r[0] for r in cur.fetchall()]
    cur.execute("""SELECT column_name FROM information_schema.columns
        WHERE table_schema='forex_predictor' AND table_name='features_b'
        AND column_name NOT IN ('timestamp','symbol',%s)
        ORDER BY ordinal_position""", (label_col,))
    cols_b = [r[0] for r in cur.fetchall()]
    cur.close()
    return cols_a, cols_b


def _filter_columns(cols_a, cols_b, selected_features):
    """按 selected_features 过滤列，保持原始顺序。不匹配的特征会被静默跳过。"""
    selected_set = set(selected_features)
    filtered_a = [c for c in cols_a if c in selected_set]
    filtered_b = [c for c in cols_b if c in selected_set]
    missing = selected_set - set(cols_a) - set(cols_b)
    if missing:
        jlog("features_not_found", count=len(missing), sample=sorted(missing)[:20])
    return filtered_a, filtered_b


def _build_sql(cols_a, cols_b):
    """构建 SELECT 子句"""
    parts = [f"a.`{c}`" for c in cols_a] + [f"b.`{c}`" for c in cols_b]
    return ", ".join(parts) if parts else "1 AS placeholder"


def export_features(
    selected_features: Optional[List[str]] = None,
    dst_dir: str = "E:/forex_npy/EURUSD",
    dry_run: bool = False,
    symbol: str = "EURUSD",
    target: str = "15m",
) -> Dict:
    """
    从 MySQL 导出特征向量到 .npy 文件

    Args:
        selected_features: 要导出的特征名列表，None 或空列表 = 导出全部
        dst_dir: 输出目录
        dry_run: 仅打印，不实际写入
        symbol: 导出的货币对（按 features_a/b 的 symbol 列过滤）
        target: 标签窗口（列 target_{target}，文件 labels_{target}.npy）

    Returns:
        {
            'n_rows': int, 'n_cols': int,
            'feature_names': [str],  # 实际导出的特征名列表（保持顺序）
            'selected_features': [str] or None,  # 入参
            'version_key': str,
            'dst_dir': str,
            'symbol': str,
            'target': str,
        }
    """
    # ---- 获取列名 ----
    conn = pymysql.connect(**MYSQL)
    all_a, all_b = _get_all_columns(conn, target)
    conn.close()

    if selected_features and len(selected_features) > 0:
        cols_a, cols_b = _filter_columns(all_a, all_b, selected_features)
        used_selection = True
    else:
        cols_a, cols_b = all_a, all_b
        used_selection = False

    total_cols = len(cols_a) + len(cols_b)
    feature_names = cols_a + cols_b  # 保持 a 后 b 的顺序

    if total_cols == 0:
        raise ValueError("No features to export (all filtered out?)")

    jlog("cols", a=len(cols_a), a_total=len(all_a), b=len(cols_b), b_total=len(all_b), total=total_cols)

    # ---- 行数 ----
    conn = pymysql.connect(**MYSQL, cursorclass=DictCursor)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) c FROM features_a WHERE symbol=%s", (symbol,))
    n = cur.fetchone()["c"]
    cur.close(); conn.close()
    jlog("rows", count=n, symbol=symbol)

    if n == 0:
        raise ValueError(
            f"symbol '{symbol}' 在 features_a 无特征行 — "
            f"先运行特征构建: python -u _build_features_v2.py --mode full --symbol {symbol}")

    if dry_run:
        jlog("dry_run", rows=n, cols=total_cols, dst=dst_dir, symbol=symbol, target=target)
        return {
            'n_rows': n, 'n_cols': total_cols,
            'feature_names': feature_names,
            'selected_features': selected_features if used_selection else None,
            'version_key': VERSION_KEY,
            'dst_dir': dst_dir,
            'symbol': symbol,
            'target': target,
        }

    # ---- 清理旧文件 ----
    for f in ["features.npy", f"labels_{target}.npy", "timestamps.npy",
              "weights.npy", ".done", "features.mmap", "labels.mmap",
              "feature_names.json", "export_meta.json"]:
        p = os.path.join(dst_dir, f)
        if os.path.exists(p):
            os.remove(p)

    # ---- mmap 预分配 ----
    os.makedirs(dst_dir, exist_ok=True)
    X_mmap_path = os.path.join(dst_dir, "features.mmap")
    y_mmap_path = os.path.join(dst_dir, "labels.mmap")
    X = np.memmap(X_mmap_path, dtype=np.float32, mode='w+', shape=(n, total_cols))
    y = np.memmap(y_mmap_path, dtype=np.int8, mode='w+', shape=(n,))
    t_list = []
    written = 0
    last_ts = "0000-01-01 00:00:00"

    col_sel = _build_sql(cols_a, cols_b)

    conn = pymysql.connect(**MYSQL, cursorclass=DictCursor)
    cur = conn.cursor()

    while True:
        if written > 0 and written % REFRESH_EVERY == 0:
            cur.close(); conn.close()
            conn = pymysql.connect(**MYSQL, cursorclass=DictCursor)
            cur = conn.cursor()

        cur.execute(f"""SELECT a.timestamp, a.`target_{target}`, {col_sel}
            FROM features_a a JOIN features_b b ON a.timestamp=b.timestamp AND a.symbol=b.symbol
            WHERE a.symbol=%s AND a.timestamp > %s
            ORDER BY a.timestamp LIMIT {BATCH}""", (symbol, last_ts))

        rows = cur.fetchall()
        if not rows:
            break

        chunk_feats, chunk_labs, chunk_times = [], [], []
        for r in rows:
            vals = [float(r[c] if r[c] is not None else 0) for c in cols_a]
            vals += [float(r[c] if r[c] is not None else 0) for c in cols_b]
            chunk_feats.append(vals)
            chunk_labs.append(int(r[f"target_{target}"]))
            chunk_times.append(str(r["timestamp"]))

        batch_size = len(rows)
        end = written + batch_size
        X[written:end] = np.array(chunk_feats, dtype=np.float32)
        y[written:end] = np.array(chunk_labs, dtype=np.int8)
        t_list.extend(chunk_times)
        written = end
        last_ts = chunk_times[-1]
        jlog("progress", written=written, total=n, pct=written*100//n, last_ts=str(last_ts))

    cur.close(); conn.close()

    # ---- Flush mmap -> disk ----
    X.flush(); del X
    y.flush(); del y

    # ---- mmap -> .npy ----
    jlog("converting_mmap_to_npy")
    X = np.memmap(X_mmap_path, dtype=np.float32, mode='r', shape=(n, total_cols))
    y_final = np.memmap(y_mmap_path, dtype=np.int8, mode='r', shape=(n,))
    t_arr = np.array(t_list); del t_list
    jlog("shape", x_shape=list(X.shape), x_dtype=str(X.dtype), y_shape=list(y_final.shape), dims=X.shape[1])

    done = os.path.join(dst_dir, ".done")
    if os.path.exists(done):
        os.remove(done)

    np.save(os.path.join(dst_dir, "features.npy"), X)
    np.save(os.path.join(dst_dir, f"labels_{target}.npy"), y_final)
    np.save(os.path.join(dst_dir, "timestamps.npy"), t_arr)
    np.save(os.path.join(dst_dir, "weights.npy"), np.ones(n, dtype=np.float32))
    del X, y_final

    # ---- 保存特征名 (追溯用) ----
    with open(os.path.join(dst_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    jlog("feature_names_saved", count=len(feature_names))

    # ---- 保存导出元数据 ----
    meta = {
        'version_key': VERSION_KEY,
        'symbol': symbol,
        'target': target,
        'n_rows': n,
        'n_cols': total_cols,
        'selected_features': selected_features if used_selection else None,
        'feature_names': feature_names,
        'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        't_min': t_arr[0],
        't_max': t_arr[-1],
    }
    with open(os.path.join(dst_dir, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- 清理 mmap ----
    for f in ["features.mmap", "labels.mmap"]:
        p = os.path.join(dst_dir, f)
        if os.path.exists(p):
            os.remove(p)

    Path(done).touch()

    # ---- 验证 ----
    X2 = np.load(os.path.join(dst_dir, "features.npy"))
    y2 = np.load(os.path.join(dst_dir, f"labels_{target}.npy"))
    jlog("verify", x_shape=list(X2.shape), y_shape=list(y2.shape), dims=X2.shape[1])
    assert X2.shape[1] == total_cols, f"WRONG DIMS: {X2.shape[1]} vs {total_cols}"
    jlog("done", status="dims_correct")
    return meta


# ---- CLI ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export features from MySQL to .npy (low memory mmap)")
    parser.add_argument("--features", type=str, default=None,
                        help="Path to JSON file with selected feature names list")
    parser.add_argument("--dst", type=str, default="E:/forex_npy/EURUSD",
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print what would be done")
    parser.add_argument("--symbol", type=str, default="EURUSD",
                        help="Symbol to export (filters features_a/b by symbol column)")
    parser.add_argument("--target", type=str, default="15m",
                        help="Target label window (column target_{target}, file labels_{target}.npy)")
    args = parser.parse_args()

    selected = None
    if args.features:
        with open(args.features, "r", encoding="utf-8") as f:
            selected = json.load(f)
        if not isinstance(selected, list):
            jlog("error", msg="--features JSON must be a list of feature names")
            sys.exit(1)
        jlog("selected_features", count=len(selected), src=args.features)

    export_features(selected_features=selected, dst_dir=args.dst, dry_run=args.dry_run,
                    symbol=args.symbol, target=args.target)
