"""POST /api/adopt — adopt a prediction"""
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.db import get_conn

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["adopt"])


class AdoptRequest(BaseModel):
    run_id: str
    symbol: str
    target_tf: str
    timestamp: str


@router.post("/adopt")
async def adopt_prediction(body: AdoptRequest):
    conn = get_conn()
    try:
        cur = conn.cursor()

        # 1. Find the prediction
        cur.execute(
            "SELECT id, is_adopted FROM model_predictions "
            "WHERE run_id=%s AND symbol=%s AND target_tf=%s AND timestamp=%s AND mode='natural'",
            (body.run_id, body.symbol, body.target_tf, body.timestamp),
        )
        pred = cur.fetchone()
        if not pred:
            raise HTTPException(404, "Prediction record not found")
        if pred["is_adopted"]:
            return {"message": "already adopted", "prediction_id": pred["id"]}

        # 2. Check within last 5 candles for this TF
        table = f"candles_{body.target_tf}"
        cur.execute(
            f"SELECT timestamp FROM `{table}` WHERE symbol=%s ORDER BY timestamp DESC LIMIT 5",
            (body.symbol,),
        )
        recent_ts = {str(row["timestamp"]) for row in cur.fetchall()}

        if body.timestamp not in recent_ts:
            raise HTTPException(400, "Prediction not within last 5 candles, cannot adopt")

        # 3. Update
        cur.execute(
            "UPDATE model_predictions SET is_adopted=1, adopted_at=NOW() WHERE id=%s",
            (pred["id"],),
        )
        conn.commit()
        cur.close()

        return {"message": "adopted", "prediction_id": pred["id"]}

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"adopt error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        conn.close()
