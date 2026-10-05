"""Pipeline status checker — single source of truth for AI and human.

Usage:
    python _pipeline_status.py          # Full status report
    python _pipeline_status.py --json   # Machine-readable JSON for AI consumers

Guarantees:
    - Idempotent, read-only, no side effects
    - Reports which steps are done / in-progress / pending
    - Flags inconsistencies (missing .done, stale files, dim mismatches)
    - Tells the next step to execute
"""

import os, sys, json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
DATA_STORE = Path("E:/forex_npy")  # 主数据目录（大文件 npy）


# ── helpers ──

def _db_conn(port=3305):
    """Quick DB connection, returns connection or None"""
    try:
        import pymysql
        conn = pymysql.connect(
            host="localhost", port=port, user="root", password="",
            database="forex_predictor", charset="utf8mb4",
            connect_timeout=3,
        )
        return conn
    except Exception:
        return None


def _db_fetch(query, port=3305):
    conn = _db_conn(port)
    if conn is None:
        return None, f"DB({port}) unreachable"
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows, None
    except Exception as e:
        try:
            conn.close()
        except Exception:
            pass
        return None, str(e)


def _npy_info(path):
    """Return (shape, dtype, size_mb) or None"""
    try:
        import numpy as np
        if not os.path.exists(str(path)):
            return None
        arr = np.load(str(path), mmap_mode='r')
        mb = os.path.getsize(str(path)) / 1024 / 1024
        return (arr.shape, arr.dtype, mb)
    except Exception:
        return None


# ── check functions ──

def check_db_running():
    rows, err = _db_fetch("SELECT 1")
    return {"running": err is None, "error": str(err) if err else None}


def check_db_tables():
    rows, err = _db_fetch(
        "SELECT table_name, table_rows, ROUND((data_length+index_length)/1024/1024,2) mb "
        "FROM information_schema.tables WHERE table_schema='forex_predictor' "
        "ORDER BY (data_length+index_length) DESC")
    if err:
        return {"error": err}
    return [{"table": r[0], "rows": r[1], "mb": float(r[2])} for r in rows]


def check_export(symbol="EURUSD"):
    """Check export status for a symbol"""
    d = DATA_STORE / symbol
    result = {
        "symbol": symbol,
        "dir_exists": d.exists(),
        "done": (d / ".done").exists(),
        "meta": None,
        "features_npy": None,
        "labels_npy": None,
        "stale_files": [],
        "errors": [],
    }

    # Read export_meta.json
    meta_path = d / "export_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                result["meta"] = json.load(f)
        except Exception as e:
            result["errors"].append(f"export_meta.json parse error: {e}")

    # Check .npy files
    for name, key in [("features.npy", "features_npy"),
                      ("labels_15m.npy", "labels_npy")]:
        info = _npy_info(d / name)
        if info:
            shape, dtype, mb = info
            result[key] = {"shape": list(shape), "dtype": str(dtype), "mb": round(mb, 1)}

    # Check for stale temp files
    for stale in ["features.mmap", "labels.mmap", "tmp.features.npy",
                  "tmp.labels_15m.npy", "tmp.timestamps.npy"]:
        p = d / stale
        if p.exists():
            result["stale_files"].append({"name": stale, "mb": round(
                os.path.getsize(str(p)) / 1024 / 1024, 1)})

    # Consistency checks
    if result["meta"] and result["features_npy"]:
        meta_cols = result["meta"].get("n_cols")
        npy_cols = result["features_npy"]["shape"][1] if len(result["features_npy"]["shape"]) > 1 else result["features_npy"]["shape"][0]
        if meta_cols != npy_cols:
            result["errors"].append(
                f"Dimension mismatch: meta.n_cols={meta_cols} vs .npy dim={npy_cols}")

    if result["meta"] and result["labels_npy"]:
        meta_rows = result["meta"].get("n_rows")
        npy_rows = result["labels_npy"]["shape"][0]
        if meta_rows != npy_rows:
            result["errors"].append(
                f"Row mismatch: meta.n_rows={meta_rows} vs .npy rows={npy_rows}")

    if result["done"] and not result["features_npy"]:
        result["errors"].append(".done exists but features.npy missing")

    return result


def check_training(symbol="EURUSD", target_tf="15m"):
    """Check training status via model_runs table"""
    rows, err = _db_fetch(f"""
        SELECT run_id, symbol, target_tf, dimension, feature_version_key,
               auc, trials_completed, selected_features, created_at
        FROM model_runs WHERE symbol='{symbol}' AND target_tf='{target_tf}'
        ORDER BY created_at DESC LIMIT 10
    """)
    if err:
        return {"error": err}

    result = {"symbol": symbol, "target_tf": target_tf, "runs": []}
    for r in rows:
        selected = None
        if r[7]:
            try:
                selected = json.loads(r[7])
            except: pass
        result["runs"].append({
            "run_id": r[0], "dimension": r[3], "version_key": r[4],
            "auc": r[5], "trials": r[6],
            "selected_count": len(selected) if selected else None,
            "created_at": str(r[8]) if r[8] else None,
        })
    return result


def check_feature_versions():
    rows, err = _db_fetch(
        "SELECT version_key, dimension, feature_count, status, description, created_at "
        "FROM feature_versions ORDER BY created_at DESC")
    if err:
        return {"error": err}
    return [{"key": r[0], "dim": r[1], "count": r[2], "status": r[3],
             "desc": r[4], "created": str(r[5])} for r in rows]


def check_optuna_study(symbol="EURUSD", target_tf="15m", model_type="xgboost"):
    """Check Optuna SQLite study status"""
    db_path = ROOT / f"optuna_{symbol}_{target_tf}_{model_type}.db"
    if not db_path.exists():
        return {"exists": False, "path": str(db_path)}

    try:
        import optuna
        storage = f"sqlite:///{db_path}"
        study = optuna.load_study(
            study_name=f"{symbol}_{target_tf}_{model_type}",
            storage=storage)
        trials = study.trials
        completed = [t for t in trials if t.state.name == "COMPLETE"]
        return {
            "exists": True,
            "path": str(db_path),
            "total_trials": len(trials),
            "completed": len(completed),
            "best_value": study.best_value if completed else None,
        }
    except Exception as e:
        return {"exists": True, "path": str(db_path), "error": str(e)}


# -- main report --

def full_report(json_output=False):
    report = {
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sections": {},
        "next_steps": [],
    }

    # 1. DB
    report["sections"]["db"] = check_db_running()
    if report["sections"]["db"]["running"]:
        report["sections"]["tables"] = check_db_tables()
        report["sections"]["feature_versions"] = check_feature_versions()

    # 2. Export
    report["sections"]["export"] = check_export()

    # 3. Training
    report["sections"]["training"] = check_training()
    report["sections"]["optuna"] = check_optuna_study()

    # 4. Determine next steps
    export = report["sections"]["export"]
    training = report["sections"]["training"]

    if not export["done"]:
        if export.get("features_npy") or export.get("stale_files"):
            report["next_steps"].append("EXPORT IN PROGRESS: wait for _export_v2.py to finish")
        else:
            report["next_steps"].append("EXPORT PENDING: run `python _export_v2.py`")
    elif not training["runs"]:
        report["next_steps"].append("TRAINING PENDING: run `python -m training.stage2_optuna --symbol EURUSD --target 15m --n-trials 1`")
    elif any(r["auc"] is None for r in training["runs"]):
        report["next_steps"].append("TRAINING IN PROGRESS: model_runs has pending (auc=NULL) records, wait for completion")
    else:
        report["next_steps"].append("ALL DONE: review results, run SHAP analysis, plan Phase 5 TS-CV")

    # Stale file warnings
    if export.get("stale_files"):
        report["next_steps"].insert(0,
            f"[!] CLEANUP: {len(export['stale_files'])} stale temp files in E:/forex_npy/EURUSD/ "
            f"({sum(f['mb'] for f in export['stale_files']):.0f} MB total)")

    # Dimension mismatch warnings
    if export.get("errors"):
        for e in export["errors"]:
            report["next_steps"].insert(0, f"[!!] CONSISTENCY: {e}")

    if json_output:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        _print_pretty(report)

    return report


def _print_pretty(report):
    print("=" * 60)
    print(f"  PIPELINE STATUS  @  {report['checked_at']}")
    print("=" * 60)

    # DB
    db = report["sections"]["db"]
    status = "[OK] RUNNING" if db["running"] else f"[FAIL] DOWN ({db['error']})"
    print(f"\n  MariaDB 3305: {status}")

    # Tables
    if "tables" in report["sections"]:
        print("\n  -- Database Tables --")
        for t in report["sections"]["tables"][:6]:
            print(f"    {t['table']:20s}  {t['rows']:>12,} rows  {t['mb']:>10,.1f} MB")

    # Feature versions
    fv = report["sections"].get("feature_versions", [])
    if fv:
        print("\n  -- Feature Versions --")
        for v in fv:
            print(f"    {v['key']:20s}  dim={v['dim']}  [{v['status']}]  {v['desc']}")

    # Export
    exp = report["sections"]["export"]
    print(f"\n  -- Export ({exp['symbol']}) --")
    done = exp["done"]
    print(f"    Status: {'[OK] DONE' if done else '[..] INCOMPLETE'}")

    if exp["features_npy"]:
        s = exp["features_npy"]
        print(f"    features.npy:  {s['shape']}  {s['dtype']}  {s['mb']} MB")
    if exp["labels_npy"]:
        print(f"    labels_15m.npy: {exp['labels_npy']['shape']}  {exp['labels_npy']['mb']} MB")
    if exp["meta"]:
        m = exp["meta"]
        sel = f"selected={len(m['selected_features'])}" if m.get("selected_features") else "ALL"
        print(f"    Meta:  {m['n_rows']:,} rows x {m['n_cols']} cols  [{sel}]  ver={m['version_key']}")
    if exp["stale_files"]:
        print(f"    [!] Stale temp files ({sum(f['mb'] for f in exp['stale_files']):.0f} MB):")
        for f in exp["stale_files"]:
            print(f"       {f['name']} ({f['mb']} MB)")
    if exp["errors"]:
        for e in exp["errors"]:
            print(f"    [!!] {e}")

    # Training
    tr = report["sections"]["training"]
    print(f"\n  -- Training --")
    if tr.get("error"):
        print(f"    [!!] {tr['error']}")
    elif tr.get("runs") is None:
        print("    (no training runs yet)")
    elif tr["runs"]:
        for r in tr["runs"]:
            auc_str = f"{r['auc']:.4f}" if r['auc'] else "pending"
            sel_str = f"selected={r['selected_count']}" if r['selected_count'] else "all"
            print(f"    {r['run_id'][:12]}...  dim={r['dimension']}  AUC={auc_str}  trials={r['trials']}  [{sel_str}]")
    else:
        print("    (no training runs yet)")

    # Optuna
    opt = report["sections"]["optuna"]
    print(f"\n  -- Optuna Study --")
    if opt["exists"]:
        print(f"    DB: {opt['path']}")
        if opt.get("error"):
            print(f"    [!] {opt['error']}")
        else:
            best_str = f"Best: {opt['best_value']:.4f}" if opt['best_value'] else ""
            print(f"    Trials: {opt['completed']}/{opt['total_trials']} completed  {best_str}")
    else:
        print("    (study not started)")

    # Next steps
    print(f"\n  -- Next Steps --")
    for i, step in enumerate(report["next_steps"], 1):
        print(f"    {i}. {step}")
    print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Pipeline status checker")
    p.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--target", default="15m")
    args = p.parse_args()

    full_report(json_output=args.json)
