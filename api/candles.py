"""GET /api/candles — candle OHLC data with pagination"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.db import get_conn, VALID_TFS

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["candles"])

# TF → minutes mapping
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}


@router.get("/candles")
async def get_candles(
    symbol: str = Query(..., description="Currency pair"),
    tf: str = Query(..., description="Timeframe: 1m/5m/15m/30m/1h"),
    before: Optional[str] = Query(None, description="Bars before this timestamp (older)"),
    after: Optional[str] = Query(None, description="Bars after this timestamp (newer)"),
    from_ts: Optional[str] = Query(None, alias="from", description="Start of time range"),
    limit: int = Query(100, ge=1, le=2000, description="Max bars"),
):
    if tf not in VALID_TFS:
        raise HTTPException(400, f"Invalid tf. Must be one of: {', '.join(sorted(VALID_TFS))}")

    table = f"candles_{tf}"
    params_provided = sum(1 for x in [before, after, from_ts] if x)
    if params_provided > 1:
        raise HTTPException(400, "Use only one of: before, after, from")

    conn = get_conn()
    try:
        cur = conn.cursor()

        # Total count (always return)
        cur.execute(f"SELECT COUNT(*) AS cnt FROM `{table}` WHERE symbol=%s", (symbol,))
        total_count = cur.fetchone()["cnt"]

        # Query
        if before:
            cur.execute(
                f"SELECT timestamp, open, high, low, close, tick_volume "
                f"FROM `{table}` WHERE symbol=%s AND timestamp < %s "
                f"ORDER BY timestamp DESC LIMIT %s",
                (symbol, before, limit + 1),
            )
            rows = cur.fetchall()
            has_more = len(rows) > limit
            if has_more:
                rows = rows[:limit]
            rows.reverse()

        elif after:
            cur.execute(
                f"SELECT timestamp, open, high, low, close, tick_volume "
                f"FROM `{table}` WHERE symbol=%s AND timestamp > %s "
                f"ORDER BY timestamp ASC LIMIT %s",
                (symbol, after, limit + 1),
            )
            rows = cur.fetchall()
            has_more = len(rows) > limit
            if has_more:
                rows = rows[:limit]

        elif from_ts:
            # Time range: from from_ts, limit bars forward
            cur.execute(
                f"SELECT timestamp, open, high, low, close, tick_volume "
                f"FROM `{table}` WHERE symbol=%s AND timestamp >= %s "
                f"ORDER BY timestamp ASC LIMIT %s",
                (symbol, from_ts, limit + 1),
            )
            rows = cur.fetchall()
            has_more = len(rows) > limit
            if has_more:
                rows = rows[:limit]

        else:
            cur.execute(
                f"SELECT timestamp, open, high, low, close, tick_volume "
                f"FROM `{table}` WHERE symbol=%s "
                f"ORDER BY timestamp DESC LIMIT %s",
                (symbol, limit),
            )
            rows = cur.fetchall()
            rows.reverse()
            has_more = False

        cur.close()

        candles = [
            {
                "timestamp": str(row["timestamp"]),
                "bar_start": str(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "tick_volume": int(row["tick_volume"]),
            }
            for row in rows
        ]

        return {
            "candles": candles,
            "count": len(candles),
            "has_more": has_more,
            "total_count": total_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"candles error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        conn.close()
