"""轻量版本化迁移：当前用 SQLite + 顺序迁移，接口与 alembic 对齐思路。"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import text

from core.config import settings
from persistence.db import engine, session_scope
from persistence.models import Base, SchemaVersion

MIGRATIONS: list[tuple[int, str]] = [
    (1, "initial schema: datasets/jobs/models/devices/evaluations"),
    (2, "indexes for job queue and episode lookup"),
    (3, "quality reports + dataset split metadata"),
    (4, "job retry budget + attempt recovery"),
    (5, "attempt mlflow run linkage"),
    (6, "annotation tables"),
    (7, "flora auto-annotation runs"),
    (8, "dataset version episode indices"),
    (9, "workbench deployment records"),
    (10, "log index"),
    (11, "projects + project linkage for processing pipelines"),
    (12, "persistent agent confirmation records"),
    (13, "persistent agent execution events"),
    (14, "persistent agent workflow plans and step state"),
    (15, "username login sessions"),
]

INDEXES_V2 = [
    "CREATE INDEX IF NOT EXISTS ix_jobs_queue ON jobs(status, priority, id)",
    "CREATE INDEX IF NOT EXISTS ix_attempts_job ON job_attempts(job_id, attempt_no)",
    "CREATE INDEX IF NOT EXISTS ix_episodes_version ON episodes(dataset_version_id, episode_index)",
]


def current_version() -> int:
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
        ).fetchone()
        if not exists:
            return 0
        row = conn.execute(text("SELECT COALESCE(MAX(version),0) FROM schema_version")).fetchone()
        return int(row[0] or 0)


def upgrade(verbose: bool = True) -> int:
    """按 MIGRATIONS 列表顺序升级；每步幂等，可重复执行。"""
    Base.metadata.create_all(engine)
    ver = current_version()
    with session_scope() as s:
        for version, note in MIGRATIONS:
            if ver >= version:
                continue
            if version == 2:
                with engine.begin() as conn:
                    for ddl in INDEXES_V2:
                        conn.execute(text(ddl))
            elif version == 4:
                with engine.begin() as conn:
                    try:
                        conn.execute(text("ALTER TABLE jobs ADD COLUMN max_attempts INTEGER DEFAULT 2"))
                    except Exception:
                        pass
            elif version == 8:
                with engine.begin() as conn:
                    try:
                        conn.execute(text(
                            "ALTER TABLE dataset_versions ADD COLUMN episode_indices TEXT"))
                    except Exception:
                        pass
            elif version == 11:
                with engine.begin() as conn:
                    try:
                        conn.execute(text(
                            "ALTER TABLE pipeline_tasks ADD COLUMN project_id INTEGER "
                            "REFERENCES projects(id)"))
                    except Exception:
                        pass
            elif version == 12:
                with engine.begin() as conn:
                    conn.execute(text(
                        "CREATE TABLE IF NOT EXISTS agent_confirmations ("
                        "confirmation_id TEXT PRIMARY KEY, "
                        "project_id INTEGER NOT NULL, "
                        "action TEXT NOT NULL, "
                        "payload_digest TEXT NOT NULL, "
                        "payload_json TEXT NOT NULL, "
                        "status TEXT NOT NULL, "
                        "created_at REAL NOT NULL, "
                        "expires_at REAL NOT NULL, "
                        "updated_at REAL NOT NULL, "
                        "result_json TEXT, "
                        "error_json TEXT)"
                    ))
                    conn.execute(text(
                        "CREATE INDEX IF NOT EXISTS ix_agent_confirmations_status_expiry "
                        "ON agent_confirmations(status, expires_at)"
                    ))
            elif version == 13:
                with engine.begin() as conn:
                    conn.execute(text(
                        "CREATE TABLE IF NOT EXISTS agent_confirmation_events ("
                        "event_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "project_id INTEGER NOT NULL, "
                        "confirmation_id TEXT NOT NULL, "
                        "action TEXT NOT NULL, "
                        "event_type TEXT NOT NULL, "
                        "created_at REAL NOT NULL, "
                        "detail_json TEXT NOT NULL DEFAULT '{}')"
                    ))
                    conn.execute(text(
                        "CREATE INDEX IF NOT EXISTS ix_agent_confirmation_events_project_cursor "
                        "ON agent_confirmation_events(project_id, event_id)"
                    ))
            elif version == 14:
                with engine.begin() as conn:
                    conn.execute(text(
                        "CREATE TABLE IF NOT EXISTS agent_workflows ("
                        "workflow_id TEXT PRIMARY KEY, "
                        "project_id INTEGER NOT NULL, "
                        "kind TEXT NOT NULL, "
                        "name TEXT NOT NULL, "
                        "status TEXT NOT NULL, "
                        "payload_json TEXT NOT NULL, "
                        "steps_json TEXT NOT NULL, "
                        "result_json TEXT NOT NULL DEFAULT '{}', "
                        "error_json TEXT, "
                        "created_at REAL NOT NULL, "
                        "updated_at REAL NOT NULL)"
                    ))
                    conn.execute(text(
                        "CREATE INDEX IF NOT EXISTS ix_agent_workflows_project_status "
                        "ON agent_workflows(project_id, status, updated_at)"
                    ))
            elif version == 15:
                with engine.begin() as conn:
                    conn.execute(text(
                        "CREATE INDEX IF NOT EXISTS ix_user_sessions_expiry "
                        "ON user_sessions(expires_at)"
                    ))
            elif version == 5:
                with engine.begin() as conn:
                    try:
                        conn.execute(
                            text("ALTER TABLE job_attempts ADD COLUMN mlflow_run_id VARCHAR(64)")
                        )
                    except Exception:
                        pass
            s.add(SchemaVersion(version=version, note=note))
            ver = version
            if verbose:
                print(f"[migrate] applied v{version}")
    if verbose:
        print(f"[migrate] schema version = {current_version()}  db = {settings.db_path}")
    return current_version()


if __name__ == "__main__":
    upgrade()
