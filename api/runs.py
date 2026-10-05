"""GET /api/runs — training run list"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from api.db import get_conn, VALID_SORT_FIELDS_RUNS, VALID_ORDERS

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["runs"])


@router.get("/runs")
async def get_runs(
    symbol: str = Query(..., description="Currency pair"),
    dimension: Optional[int] = Query(None),
    target_tf: Optional[str] = Query(None),
    feature_version_key: Optional[str] = Query(None),
    sort_by: str = Query("auc", description="auc / created_at / live_accuracy"),
    order: str = Query("desc"),
    limit: int = Query(20, ge=1, le=100),
):
    if sort_by not in VALID_SORT_FIELDS_RUNS:
        raise HTTPException(400, f"Invalid sort_by. Must be one of: {', '.join(sorted(VALID_SORT_FIELDS_RUNS))}")
    if order not in VALID_ORDERS:
        raise HTTPException(400, f"Invalid order. Must be asc or desc")

    conditions = ["symbol = %s"]
    params = [symbol]

    if dimension is not None:
        conditions.append("dimension = %s")
        params.append(dimension)
    if target_tf:
        conditions.append("target_tf = %s")
        params.append(target_tf)
    if feature_version_key:
        conditions.append("feature_version_key = %s")
        params.append(feature_version_key)

    where = " AND ".join(conditions)
    sql = (
        f"SELECT run_id, symbol, target_tf, dimension, feature_version_key, "
        f"auc, live_count, live_accuracy, live_status, trials_completed, description, created_at "
        f"FROM model_runs WHERE {where} ORDER BY {sort_by} {order} LIMIT %s"
    )
    params.append(limit)

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()

        runs = []
        for row in rows:
            runs.append({
                "run_id": row["run_id"],
                "symbol": row["symbol"],
                "target_tf": row["target_tf"],
                "dimension": row["dimension"],
                "feature_version_key": row["feature_version_key"],
                "auc": float(row["auc"]) if row["auc"] is not None else None,
                "live_count": row["live_count"],
                "live_accuracy": float(row["live_accuracy"]) if row["live_accuracy"] is not None else None,
                "live_status": row["live_status"],
                "trials_completed": row["trials_completed"],
                "description": row["description"],
                "created_at": str(row["created_at"]) if row["created_at"] else None,
            })

        return {"runs": runs}

    except Exception as e:
        log.error(f"runs error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        conn.close()
