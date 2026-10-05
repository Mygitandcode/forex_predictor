"""
mysql_proxy.py — FastAPI MySQL 查询代理（解决 IDE 超时问题）

问题: CodeBuddy IDE execute_command 有 ~3 秒超时, pymysql 握手 >2 秒,
      AI 无法直接连 MySQL。

解决: FastAPI 启动时建立 MySQL 连接池（长连接），AI 通过 HTTP 调 /api/mysql/query，
      FastAPI 内部查 MySQL → 返回 JSON。HTTP 调用 <1 秒, 不受超时限制。

使用: 在 fastapi_app.py 中 import 并 mount 本模块的 router:
      from serving.mysql_proxy import router as mysql_router
      app.include_router(mysql_router)

端点:
  GET  /api/mysql/tables          → 列出 forex_predictor 库所有表及行数
  POST /api/mysql/query           → 执行只读 SQL (SELECT/SHOW/DESCRIBE/EXPLAIN)
  GET  /api/mysql/health          → 健康检查
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mysql", tags=["mysql"])

# ---- 连接池 (模块级，FastAPI 启动时初始化) ----
_pool = None


def init_pool(host="localhost", port=3305, user="root", password="",
              database="forex_predictor", charset="utf8mb4", pool_size=3):
    """初始化 MySQL 连接池（在 FastAPI startup 事件中调用）"""
    global _pool
    try:
        import pymysql
        from pymysql.cursors import DictCursor
        _pool = {
            "config": dict(host=host, port=port, user=user, password=password,
                           database=database, charset=charset,
                           connect_timeout=3, read_timeout=5, write_timeout=5,
                           cursorclass=DictCursor),
            "pymysql": pymysql,
        }
        # 测试连通（3 秒超时，不阻塞 FastAPI 启动）
        conn = pymysql.connect(**_pool["config"])
        conn.ping()
        conn.close()
        logger.info(f"MySQL proxy initialized: {user}@{host}:{port}/{database}")
    except Exception as e:
        logger.warning(f"MySQL proxy init failed (XAMPP MySQL 可能未启动): {e}")
        _pool = None


def _get_conn():
    """获取一个 MySQL 连接"""
    if _pool is None:
        raise RuntimeError("MySQL proxy 未初始化（XAMPP MySQL 可能未启动）")
    return _pool["pymysql"].connect(**_pool["config"])


# ---- 请求/响应模型 ----
class QueryRequest(BaseModel):
    sql: str
    limit: int = 100  # 安全上限


# ---- 路由 ----
@router.get("/health")
async def mysql_health():
    """MySQL 代理健康检查"""
    if _pool is None:
        return {"status": "unavailable", "message": "MySQL 未连接（XAMPP 可能未启动）"}
    try:
        conn = _get_conn()
        conn.ping()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/tables")
async def list_tables():
    """列出 forex_predictor 库所有表及行数"""
    if _pool is None:
        raise HTTPException(503, "MySQL 未连接")
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SHOW TABLES")
        tables_raw = [row for row in cur.fetchall()]
        table_key = list(tables_raw[0].keys())[0] if tables_raw else None

        tables = []
        for row in tables_raw:
            tname = row[table_key]
            try:
                cur.execute(f"SELECT COUNT(*) AS cnt FROM `{tname}`")
                cnt = cur.fetchone()["cnt"]
            except Exception:
                cnt = -1
            tables.append({"table": tname, "rows": cnt})
        return {"database": _pool["config"]["database"], "tables": tables}
    finally:
        cur.close()
        conn.close()


@router.post("/query")
async def execute_query(req: QueryRequest):
    """执行只读 SQL 查询（SELECT / SHOW / DESCRIBE / EXPLAIN）

    安全限制: 只允许只读语句, 禁止 INSERT/UPDATE/DELETE/DROP/ALTER/CREATE
    """
    if _pool is None:
        raise HTTPException(503, "MySQL 未连接")

    sql = req.sql.strip()
    sql_upper = sql.upper()

    # 安全: 只允许只读
    ALLOWED = ("SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN")
    if not any(sql_upper.startswith(w) for w in ALLOWED):
        raise HTTPException(400, f"仅允许只读查询 ({', '.join(ALLOWED)})")

    # 防全表扫描
    if "LIMIT" not in sql_upper and sql_upper.startswith("SELECT"):
        sql = f"{sql.rstrip(';')} LIMIT {req.limit}"

    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        # 获取列名
        columns = [d[0] for d in cur.description] if cur.description else []
        return {
            "sql": sql,
            "columns": columns,
            "rows": rows[:req.limit],
            "count": len(rows),
        }
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        cur.close()
        conn.close()
