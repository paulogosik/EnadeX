from fastapi import APIRouter, Depends, HTTPException
from psycopg2.extensions import cursor as PgCursor

from enade_time.api.dependencies import get_db_cursor

router = APIRouter(tags=["health"])


@router.get("/health")
def health(cursor: PgCursor = Depends(get_db_cursor)) -> dict:
    try:
        cursor.execute("SELECT 1 AS ok")
        cursor.fetchone()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database unreachable: {exc}",
        ) from exc
    return {"status": "ok", "database": "ok", "version": "0.2.0"}
