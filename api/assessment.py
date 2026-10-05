"""GET /api/assessment — assessment panel data"""
import logging
from fastapi import APIRouter, HTTPException, Query
from api.db import get_conn

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["assessment"])


@router.get("/assessment")
async def get_assessment(
    symbol: str = Query(..., description="Currency pair"),
    run_ids: str = Query(..., description="Comma-separated run IDs"),
):
    run_id_list = [r.strip() for r in run_ids.split(",") if r.strip()]
    if not run_id_list:
        raise HTTPException(400, "At least one run_id required")

    conn = get_conn()
    try:
        cur = conn.cursor()

        # 1. Get target_tf for each run
        placeholders = ",".join(["%s"] * len(run_id_list))
        cur.execute(
            f"SELECT run_id, target_tf, feature_version_key FROM model_runs WHERE run_id IN ({placeholders})",
            run_id_list,
        )
        run_tf_map = {row["run_id"]: row["target_tf"] for row in cur.fetchall()}

        # 2. For each run, get latest natural prediction
        tfs = {}
        for run_id in run_id_list:
            tf = run_tf_map.get(run_id, "15m")
            cur.execute(
                "SELECT run_id, target_tf, timestamp, prob_up, `signal`, is_adopted "
                "FROM model_predictions "
                "WHERE run_id=%s AND symbol=%s AND mode='natural' "
                "ORDER BY timestamp DESC LIMIT 1",
                (run_id, symbol),
            )
            row = cur.fetchone()
            if not row:
                continue

            signal_text = "涨" if row["signal"] == 1 else ("跌" if row["signal"] == 0 else "震荡")
            if tf not in tfs:
                tfs[tf] = []
            tfs[tf].append({
                "run_id": row["run_id"],
                "feature_version_key": None,
                "signal": signal_text,
                "prob_up": float(row["prob_up"]) if row["prob_up"] is not None else None,
                "is_adopted": bool(row["is_adopted"]),
                "timestamp": str(row["timestamp"]),
            })

        # 3. Ensure all 4 TFs present
        for tf in ["5m", "15m", "30m", "1h"]:
            if tf not in tfs:
                tfs[tf] = []

        # 4. Summary
        summary = {}
        for tf, items in tfs.items():
            up = sum(1 for i in items if i["signal"] == "涨")
            down = sum(1 for i in items if i["signal"] == "跌")
            summary[tf] = f"{up}涨{down}跌" if up + down > 0 else "无信号"

        cur.close()
        return {"symbol": symbol, "tfs": tfs, "summary": summary}

    except Exception as e:
        log.error(f"assessment error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        conn.close()
