"""_mysql_proxy_test.py — 通过 FastAPI HTTP 代理查询 MySQL（绕开 IDE 超时）

前置: FastAPI 必须先启动 (uvicorn serving.fastapi_app:app --port 8000)
用法: python _mysql_proxy_test.py
"""
import urllib.request, json, sys

BASE = "http://localhost:8000/api/mysql"

def get(path):
    try:
        r = urllib.request.urlopen(f"{BASE}{path}", timeout=5)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def post(path, body):
    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(f"{BASE}{path}", data=data,
                                     headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=5)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

print("=" * 50)
print("MySQL Proxy Test (via FastAPI HTTP)")
print("=" * 50)

# Step 1: 健康检查
r = get("/health")
print(f"Step 1 健康检查: {r}")

# Step 2: 列出所有表
r = get("/tables")
if "tables" in r:
    print(f"Step 2 表列表: {len(r['tables'])} 张表")
    for t in r["tables"]:
        print(f"  {t['table']:30s} {t['rows']:>10} rows")
else:
    print(f"Step 2 失败: {r}")

# Step 3: 执行查询
r = post("/query", {"sql": "SELECT 1 AS test, NOW() AS server_time"})
print(f"Step 3 查询测试: {r}")

# Step 4: 查 schema
r = post("/query", {"sql": "SHOW DATABASES"})
print(f"Step 4 数据库列表: {r}")

print("=" * 50)
print("全链路完成" if "error" not in str(r) else "部分失败（FastAPI 是否启动？）")
print("=" * 50)
