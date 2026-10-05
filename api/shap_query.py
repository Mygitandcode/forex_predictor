"""GET /api/shap — SHAP feature contributions"""
import logging
from fastapi import APIRouter, HTTPException, Query
from api.db import get_conn

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["shap"])


@router.get("/shap")
async def get_shap(
    run_id: str = Query(..., description="MLflow run ID"),
    limit: int = Query(20, ge=1, le=50),
):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT feature_id, feature_name, importance, rank "
            "FROM shap_contributions WHERE run_id=%s "
            "ORDER BY rank ASC LIMIT %s",
            (run_id, limit),
        )
        rows = cur.fetchall()
        cur.close()

        features = [
            {
                "feature_id": row["feature_id"],
                "feature_name": row["feature_name"],
                "importance": float(row["importance"]) if row["importance"] is not None else 0,
                "rank": row["rank"],
            }
            for row in rows
        ]

        return {"run_id": run_id, "features": features}

    except Exception as e:
        log.error(f"shap error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        conn.close()
