"""CRUD /api/feature-versions — feature set management"""
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from api.db import get_conn, VALID_SORT_FIELDS_FV, VALID_ORDERS

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["feature-versions"])


# ---- Request models ----
class FeatureVersionCreate(BaseModel):
    version_key: str
    dimension: int
    feature_names: Optional[List[str]] = None
    feature_count: int
    description: Optional[str] = None
    build_module: Optional[str] = None
    status: Optional[str] = "experimental"


class FeatureVersionUpdate(BaseModel):
    description: Optional[str] = None
    status: Optional[str] = None
    build_module: Optional[str] = None


# ---- GET list ----
@router.get("/feature-versions")
async def list_feature_versions(
    status: str = Query("active", description="active / deprecated / experimental / all"),
    dimension: Optional[int] = Query(None),
    sort_by: str = Query("created_at"),
    order: str = Query("desc"),
    limit: int = Query(20, ge=1, le=100),
):
    if sort_by not in VALID_SORT_FIELDS_FV:
        raise HTTPException(400, f"Invalid sort_by. Must be one of: {', '.join(sorted(VALID_SORT_FIELDS_FV))}")
    if order not in VALID_ORDERS:
        raise HTTPException(400, f"Invalid order. Must be asc or desc")

    conditions = []
    params = []

    if status and status != "all":
        conditions.append("fv.status = %s")
        params.append(status)
    if dimension is not None:
        conditions.append("fv.dimension = %s")
        params.append(dimension)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = (
        f"SELECT fv.*, "
        f"(SELECT COUNT(*) FROM model_runs mr WHERE mr.feature_version_key = fv.version_key) AS run_count "
        f"FROM feature_versions fv WHERE {where} ORDER BY fv.{sort_by} {order} LIMIT %s"
    )
    params.append(limit)

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()

        import json
        versions = []
        for row in rows:
            versions.append({
                "version_key": row["version_key"],
                "dimension": row["dimension"],
                "feature_names": json.loads(row["feature_names"]) if row["feature_names"] else None,
                "feature_count": row["feature_count"],
                "status": row["status"],
                "description": row["description"],
                "mlflow_run_id": row["mlflow_run_id"],
                "build_module": row["build_module"],
                "first_trained_at": str(row["first_trained_at"]) if row["first_trained_at"] else None,
                "created_at": str(row["created_at"]) if row["created_at"] else None,
                "updated_at": str(row["updated_at"]) if row["updated_at"] else None,
                "run_count": row["run_count"],
            })

        return {"versions": versions}

    except Exception as e:
        log.error(f"feature-versions list error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        conn.close()


# ---- POST create ----
@router.post("/feature-versions", status_code=201)
async def create_feature_version(body: FeatureVersionCreate):
    import json
    conn = get_conn()
    try:
        cur = conn.cursor()

        # Check if already exists
        cur.execute("SELECT version_key FROM feature_versions WHERE version_key=%s", (body.version_key,))
        if cur.fetchone():
            raise HTTPException(409, f"version_key '{body.version_key}' already exists")

        feature_names_json = json.dumps(body.feature_names) if body.feature_names else None
        cur.execute(
            "INSERT INTO feature_versions (version_key, dimension, feature_names, feature_count, status, description, build_module) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (body.version_key, body.dimension, feature_names_json, body.feature_count,
             body.status or "experimental", body.description, body.build_module),
        )
        conn.commit()
        cur.close()

        return {
            "version_key": body.version_key,
            "dimension": body.dimension,
            "status": body.status or "experimental",
            "created_at": str(body.version_key),  # placeholder
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"feature-versions create error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        conn.close()


# ---- PUT update ----
@router.put("/feature-versions/{version_key}")
async def update_feature_version(version_key: str, body: FeatureVersionUpdate):
    if body.status and body.status not in ("active", "deprecated", "experimental"):
        raise HTTPException(400, "status must be: active, deprecated, or experimental")

    conn = get_conn()
    try:
        cur = conn.cursor()

        cur.execute("SELECT * FROM feature_versions WHERE version_key=%s", (version_key,))
        existing = cur.fetchone()
        if not existing:
            raise HTTPException(404, f"version_key '{version_key}' not found")

        updates = []
        params = []
        if body.description is not None:
            updates.append("description = %s")
            params.append(body.description)
        if body.status is not None:
            updates.append("status = %s")
            params.append(body.status)
        if body.build_module is not None:
            updates.append("build_module = %s")
            params.append(body.build_module)

        if updates:
            params.append(version_key)
            cur.execute(f"UPDATE feature_versions SET {', '.join(updates)} WHERE version_key=%s", params)
            conn.commit()

        cur.execute("SELECT * FROM feature_versions WHERE version_key=%s", (version_key,))
        row = cur.fetchone()
        cur.close()

        import json
        return {
            "version_key": row["version_key"],
            "dimension": row["dimension"],
            "status": row["status"],
            "description": row["description"],
            "updated_at": str(row["updated_at"]) if row["updated_at"] else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"feature-versions update error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        conn.close()


# ---- DELETE (soft delete) ----
@router.delete("/feature-versions/{version_key}")
async def delete_feature_version(version_key: str):
    conn = get_conn()
    try:
        cur = conn.cursor()

        cur.execute("SELECT version_key, status FROM feature_versions WHERE version_key=%s", (version_key,))
        existing = cur.fetchone()
        if not existing:
            raise HTTPException(404, f"version_key '{version_key}' not found")

        cur.execute("UPDATE feature_versions SET status='deprecated' WHERE version_key=%s", (version_key,))
        conn.commit()
        cur.close()

        return {"message": "marked as deprecated", "version_key": version_key}

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"feature-versions delete error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        conn.close()
