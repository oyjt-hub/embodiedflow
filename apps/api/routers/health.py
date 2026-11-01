from __future__ import annotations

import platform
import sys

from fastapi import APIRouter

from core.config import settings
from persistence.db import engine
from persistence.migrate import current_version

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    db_ok = True
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except Exception:  # noqa: BLE001
        db_ok = False
    return {
        "ok": db_ok,
        "service": "embodiedflow-api",
        "version": "0.1.0",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": settings.host_label,
        "db": str(settings.db_path),
        "schema_version": current_version(),
        "data_root": str(settings.data_root),
    }
