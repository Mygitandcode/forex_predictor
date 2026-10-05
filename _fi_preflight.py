"""_fi_preflight.py - 特征迭代 Layer0 环境检查

无状态 AI 在 START 前调用, 检查环境就绪。
输出 JSON Lines 到 stdout, 退出码 0=全就绪 1=有缺失。

用法: python _fi_preflight.py
"""

import json
import os
import shutil
import socket
import sys
from datetime import datetime

# 确保代码根在 path(脚本在代码根, __file__ 的父目录就是代码根)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.jsonlog import jlog


def _check_port(port):
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def main():
    results = {}

    # 端口
    results["mariadb_3305"] = _check_port(3305)
    results["redis_6379"] = _check_port(6379)
    results["mlflow_5001"] = _check_port(5001)

    # E盘空间
    try:
        u = shutil.disk_usage("E:/")
        results["e_free_gb"] = round(u.free / 1e9, 1)
        results["e_ok"] = u.free / 1e9 > 5
    except Exception as e:
        results["e_error"] = str(e)
        results["e_ok"] = False

    # venv
    results["venv"] = os.path.exists(
        "D:/dapanAllThings/codes/forex_predictor/venv/Scripts/python.exe")

    # EURUSD 数据(MySQL)
    try:
        import pymysql
        conn = pymysql.connect(
            host="localhost", port=3305, user="root", password="",
            database="forex_predictor", charset="utf8mb4",
            connect_timeout=3)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM candles_1m WHERE symbol='EURUSD'")
        results["eurusd_rows"] = cur.fetchone()[0]
        results["eurusd_ok"] = results["eurusd_rows"] > 1000000
        conn.close()
    except Exception as e:
        results["eurusd_error"] = str(e)
        results["eurusd_ok"] = False

    # 总判定
    results["all_ok"] = all([
        results["mariadb_3305"], results["redis_6379"], results["mlflow_5001"],
        results.get("e_ok", False), results["venv"],
        results.get("eurusd_ok", False),
    ])

    jlog("preflight", **results)
    sys.exit(0 if results["all_ok"] else 1)


if __name__ == "__main__":
    main()
