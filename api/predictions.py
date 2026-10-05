"""GET /api/predictions — prediction records"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from api.db import get_conn

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["predictions"])


@router.get("/predictions")
async def get_predictions(
    symbol: str = Query(..., description="Currency pair"),
    run_id: Optional[str] = Query(None),
    target_tf: Optional[str] = Query(None, description="Filter by target timeframe"),
    mode: str = Query("natural", description="training or natural"),
    is_adopted: Optional[int] = Query(None, description="0=not adopted, 1=adopted"),
    from_ts: Optional[str] = Query(None, alias="from"),
    to_ts: Optional[str] = Query(None, alias="to"),
    limit: int = Query(500, ge=1, le=2000),
):
    conditions = ["symbol = %s"]
    params = [symbol]

    if run_id:
        conditions.append("run_id = %s")
        params.append(run_id)
    if target_tf:
        conditions.append("target_tf = %s")
        params.append(target_tf)
    if mode and mode in ("training", "natural"):
        conditions.append("mode = %s")
        params.append(mode)
    if is_adopted is not None:
        conditions.append("is_adopted = %s")
        params.append(is_adopted)
    if from_ts:
        conditions.append("timestamp >= %s")
        params.append(from_ts)
    if to_ts:
        conditions.append("timestamp <= %s")
        params.append(to_ts)

    where = " AND ".join(conditions)
    sql = f"SELECT * FROM model_predictions WHERE {where} ORDER BY timestamp DESC LIMIT %s"
    params.append(limit)

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()

        predictions = []
        for row in rows:
            predictions.append({
                "id": row["id"],
                "run_id": row["run_id"],
                "symbol": row["symbol"],
                "target_tf": row["target_tf"],
                "timestamp": str(row["timestamp"]),
                "mode": row["mode"],
                "prob_up": float(row["prob_up"]) if row["prob_up"] is not None else None,
                "signal": row["signal"],
                "is_adopted": row["is_adopted"],
                "adopted_at": str(row["adopted_at"]) if row["adopted_at"] else None,
                "true_label": row["true_label"],
                "is_correct": row["is_correct"],
                "dimension": row["dimension"],
            })

        return {"predictions": predictions, "count": len(predictions)}

    except Exception as e:
        log.error(f"predictions error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        conn.close()
