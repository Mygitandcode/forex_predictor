"""从 mmap 临时文件恢复 .npy 导出（_export_v2.py 中断后使用）

Usage:
    python _recover_export.py                    # 恢复 EURUSD
    python _recover_export.py --dir E:/forex_npy/EURUSD
"""
import numpy as np, os, json, shutil
from pathlib import Path
from datetime import datetime

VERSION_KEY = "20260521_mtf"
CHUNK_ROWS = 50000  # 分块写入，避免一次性 17GB 写入 OOM/文件系统截断


def _save_npy_chunked(dst, src_memmap, n_rows, n_cols, dtype, chunk_rows=CHUNK_ROWS):
    """分块流式写入 .npy：手动写 header，然后逐块 memmap→file，不经 np.save 内存副本"""
    import numpy.lib.format as fmt

    with open(dst, "wb") as f:
        header = fmt.header_data_from_array_1_0(
            np.empty((n_rows, n_cols), dtype=dtype))
        fmt.write_array_header_1_0(f, header)

        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            src_memmap[start:end].tofile(f)
            pct = end * 100 // n_rows
            print(f"  writing {dst[-30:]}: {end:,}/{n_rows:,} ({pct}%)", flush=True)
    print(f"  {dst} done ({n_rows} x {n_cols})")


def save_1d_chunked(dst, src_memmap, n_rows, dtype, chunk_rows=CHUNK_ROWS):
    """1D 数组分块写入"""
    import numpy.lib.format as fmt

    with open(dst, "wb") as f:
        header = fmt.header_data_from_array_1_0(
            np.empty((n_rows,), dtype=dtype))
        fmt.write_array_header_1_0(f, header)

        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            src_memmap[start:end].tofile(f)
            if end % (chunk_rows * 10) == 0:
                print(f"  writing {dst[-30:]}: {end:,}/{n_rows:,}", flush=True)
    print(f"  {dst} done ({n_rows})")


def recover(dst_dir="E:/forex_npy/EURUSD"):
    x_mmap = os.path.join(dst_dir, "features.mmap")
    y_mmap = os.path.join(dst_dir, "labels.mmap")
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.exists(x_mmap):
        print(f"ERROR: {x_mmap} not found — nothing to recover")
        return

    # 尝试读 feature_names.json 获取列数
    fn_path = os.path.join(dst_dir, "feature_names.json")
    if os.path.exists(fn_path):
        with open(fn_path) as f:
            feature_names = json.load(f)
        total_cols = len(feature_names)
    else:
        total_cols = 1060
        feature_names = []

    # 从文件大小反推行数
    x_size = os.path.getsize(x_mmap)
    n_from_x = x_size // (total_cols * 4)
    y_size = os.path.getsize(y_mmap)
    n_from_y = y_size

    if n_from_x != n_from_y:
        print(f"WARNING: row mismatch X→{n_from_x:,} y→{n_from_y:,}")
    n = min(n_from_x, n_from_y)

    # 磁盘空间检查
    needed_gb = (n * total_cols * 4 + n * 1) / (1024**3) + 1
    free_gb = shutil.disk_usage(dst_dir).free / (1024**3)
    print(f"Recovering {n:,} rows x {total_cols} cols, need ~{needed_gb:.0f} GB, free ~{free_gb:.0f} GB")

    X = np.memmap(x_mmap, dtype=np.float32, mode='r', shape=(n, total_cols))
    y = np.memmap(y_mmap, dtype=np.int8, mode='r', shape=(n,))
    print(f"X={X.shape} {X.dtype}, y={y.shape}")

    # ----- 清理旧输出 -----
    done = os.path.join(dst_dir, ".done")
    for f in ["features.npy", "labels_15m.npy", ".done"]:
        p = os.path.join(dst_dir, f)
        if os.path.exists(p):
            os.remove(p)

    # ----- 直接写入目标目录 -----
    dst_feat = os.path.join(dst_dir, "features.npy")
    dst_labs = os.path.join(dst_dir, "labels_15m.npy")

    print(f"Saving features.npy ...")
    _save_npy_chunked(dst_feat, X, n, total_cols, np.float32)

    print(f"Saving labels_15m.npy ...")
    save_1d_chunked(dst_labs, y, n, np.int8)

    # ----- 清理 mmap -----
    for f in ["features.mmap", "labels.mmap"]:
        p = os.path.join(dst_dir, f)
        if os.path.exists(p):
            os.remove(p)
    print("mmap temp files cleaned")

    # ----- weights -----
    w_path = os.path.join(dst_dir, "weights.npy")
    if not os.path.exists(w_path):
        np.save(w_path, np.ones(n, dtype=np.float32))
        print("weights.npy (default=1.0)")

    # ----- export_meta.json -----
    em_path = os.path.join(dst_dir, "export_meta.json")
    if not os.path.exists(em_path):
        t_path = os.path.join(dst_dir, "timestamps.npy")
        t_min = t_max = "unknown"
        if os.path.exists(t_path):
            t_arr = np.load(t_path)
            t_min = str(t_arr[0])
            t_max = str(t_arr[-1])

        meta = {
            'version_key': VERSION_KEY,
            'n_rows': n,
            'n_cols': total_cols,
            'selected_features': None,
            'feature_names': feature_names,
            'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            't_min': t_min,
            't_max': t_max,
        }
        with open(em_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"export_meta.json created")

    # ----- 验证（用 memmap 避免全量加载到内存）-----
    print("Verifying...")
    X2 = np.load(dst_feat, mmap_mode='r')
    y2 = np.load(dst_labs, mmap_mode='r')
    print(f"Verify: X={X2.shape}, y={y2.shape}")
    assert X2.shape[1] == total_cols, f"DIM MISMATCH: {X2.shape[1]} != {total_cols}"
    del X2, y2
    Path(done).touch()
    print("DONE")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="E:/forex_npy/EURUSD")
    args = p.parse_args()
    recover(args.dir)
