"""MCP surface over the existing EmbodiedFlow API and confirmation event store."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mcp.server import MCPServer

API_BASE = os.environ.get("EMBODIEDFLOW_API", "http://127.0.0.1:18080").rstrip("/")
DEFAULT_PROJECT_ID = int(os.environ.get("EMBODIEDFLOW_PROJECT_ID", "1"))
HTTP_TIMEOUT = float(os.environ.get("EMBODIEDFLOW_API_TIMEOUT", "30"))
APPROVAL_TTL_SECONDS = 900
DATA_ROOT = Path(os.environ.get(
    "EMBODIEDFLOW_DATA_ROOT",
    str(Path(__file__).resolve().parents[2].parent / "embodiedflow_data"),
))
CONFIRMATIONS_DB_PATH = Path(os.environ.get(
    "EMBODIEDFLOW_DB_PATH",
    str(DATA_ROOT / "metadata" / "platform.db"),
))
mcp = MCPServer("EmbodiedFlow")


@contextmanager
def _confirmation_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(CONFIRMATIONS_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_confirmation_table() -> None:
    CONFIRMATIONS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _confirmation_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_confirmations (
                confirmation_id TEXT PRIMARY KEY,
                project_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                result_json TEXT,
                error_json TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_agent_confirmations_status_expiry "
            "ON agent_confirmations(status, expires_at)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_confirmation_events ("
            "event_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "project_id INTEGER NOT NULL, "
            "confirmation_id TEXT NOT NULL, "
            "action TEXT NOT NULL, "
            "event_type TEXT NOT NULL, "
            "created_at REAL NOT NULL, "
            "detail_json TEXT NOT NULL DEFAULT '{}')"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_agent_confirmation_events_project_cursor "
            "ON agent_confirmation_events(project_id, event_id)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_workflows ("
            "workflow_id TEXT PRIMARY KEY, project_id INTEGER NOT NULL, "
            "kind TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL, "
            "payload_json TEXT NOT NULL, steps_json TEXT NOT NULL, "
            "result_json TEXT NOT NULL DEFAULT '{}', error_json TEXT, "
            "created_at REAL NOT NULL, updated_at REAL NOT NULL)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_agent_workflows_project_status "
            "ON agent_workflows(project_id, status, updated_at)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_pipeline_drafts ("
            "draft_id TEXT PRIMARY KEY, project_id INTEGER NOT NULL, goal TEXT NOT NULL, "
            "revision INTEGER NOT NULL, steps_json TEXT NOT NULL, created_at REAL NOT NULL, "
            "updated_at REAL NOT NULL)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_agent_pipeline_drafts_project "
            "ON agent_pipeline_drafts(project_id, updated_at)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_pipeline_idempotency ("
            "project_id INTEGER NOT NULL, idempotency_key TEXT NOT NULL, draft_id TEXT NOT NULL, "
            "confirmation_id TEXT NOT NULL, status TEXT NOT NULL, response_json TEXT NOT NULL, "
            "created_at REAL NOT NULL, updated_at REAL NOT NULL, "
            "PRIMARY KEY(project_id, idempotency_key))"
        )


def _append_confirmation_event(
    conn: sqlite3.Connection,
    project_id: int,
    confirmation_id: str,
    action: str,
    event_type: str,
    detail: dict[str, Any] | None = None,
    event_at: float | None = None,
) -> None:
    conn.execute(
        "INSERT INTO agent_confirmation_events "
        "(project_id, confirmation_id, action, event_type, created_at, detail_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            project_id, confirmation_id, action, event_type,
            event_at if event_at is not None else time.time(),
            json.dumps(detail or {}, ensure_ascii=False, default=str),
        ),
    )


_ensure_confirmation_table()


def _scope(project_id: int) -> None:
    if project_id != DEFAULT_PROJECT_ID:
        raise ValueError(f"当前服务绑定默认项目 {DEFAULT_PROJECT_ID}")


def _get(path: str, query: dict[str, Any] | None = None) -> Any:
    suffix = "?" + urlencode({k: v for k, v in (query or {}).items() if v is not None}) if query else ""
    req = Request(API_BASE + path + suffix, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=HTTP_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:1200]
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"API {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"EmbodiedFlow API 不可用: {exc.reason}") from exc


def _post(path: str, payload: dict[str, Any], timeout: float | None = None) -> Any:
    req = Request(
        API_BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout or HTTP_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:1200]
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"API {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"EmbodiedFlow API 不可用: {exc.reason}") from exc


def _training_capability(backend: str) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog = _get("/workbench/api/training/capabilities")
    entry = next(
        (item for item in catalog.get("backends", []) if item.get("id") == backend),
        None,
    )
    if not entry:
        raise ValueError(f"训练后端未登记：{backend}")
    if not entry.get("available"):
        missing = ", ".join(entry.get("unavailable_requirements", []))
        raise ValueError(f"训练后端不可用：{backend}" + (f"；缺少 {missing}" if missing else ""))
    return catalog, entry


def _submit_with_confirmation(
    project_id: int,
    action: str,
    payload: dict[str, Any],
    confirmation_id: str | None,
    submit: Callable[[], Any],
) -> dict[str, Any]:
    """Persist a user-visible preview and make confirmation retries idempotent."""
    payload_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    now = time.time()

    if not confirmation_id:
        confirmation_id = secrets.token_urlsafe(18)
        expires_at = now + APPROVAL_TTL_SECONDS
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        with _confirmation_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            expired = conn.execute(
                "SELECT project_id, confirmation_id, action FROM agent_confirmations "
                "WHERE status='pending' AND expires_at<=?",
                (now,),
            ).fetchall()
            conn.execute(
                "UPDATE agent_confirmations SET status='expired', updated_at=? "
                "WHERE status='pending' AND expires_at<=?",
                (now, now),
            )
            for row in expired:
                _append_confirmation_event(
                    conn, row["project_id"], row["confirmation_id"], row["action"],
                    "preview_expired", {"reason": "ttl_elapsed"}, now,
                )
            conn.execute(
                """
                INSERT INTO agent_confirmations
                    (confirmation_id, project_id, action, payload_digest, payload_json,
                     status, created_at, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    confirmation_id, project_id, action, payload_digest, payload_json,
                    now, expires_at, now,
                ),
            )
            _append_confirmation_event(
                conn, project_id, confirmation_id, action, "preview_created",
                {"parameters": payload, "expires_at": expires_at}, now,
            )
        return {
            "status": "awaiting_user_confirmation",
            "confirmation_id": confirmation_id,
            "expires_in_seconds": APPROVAL_TTL_SECONDS,
            "action": action,
            "parameters": payload,
            "instruction": "请先向用户展示操作和参数，并等待用户明确确认；不要在本轮继续提交。确认后用相同参数再次调用，并传入 confirmation_id。",
        }

    with _confirmation_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        pending = conn.execute(
            "SELECT * FROM agent_confirmations WHERE confirmation_id=?",
            (confirmation_id,),
        ).fetchone()
        if not pending:
            conn.rollback()
            raise ValueError("确认记录不存在或已使用；请重新调用工具生成预览，再等待用户确认")
        if (pending["project_id"] != project_id or pending["action"] != action
                or pending["payload_digest"] != payload_digest):
            _append_confirmation_event(
                conn, pending["project_id"], confirmation_id, pending["action"],
                "confirmation_rejected", {"reason": "binding_mismatch"}, now,
            )
            conn.commit()
            raise ValueError("提交参数与已确认预览不一致；请重新生成预览并等待用户确认")
        if pending["status"] == "completed":
            result = json.loads(pending["result_json"] or "null")
            _append_confirmation_event(
                conn, project_id, confirmation_id, action,
                "confirmation_replayed", {"status": "completed"}, now,
            )
            conn.commit()
            return {
                "status": "submitted",
                "action": action,
                "submitted": result,
                "replayed": True,
            }
        if pending["status"] == "executing":
            _append_confirmation_event(
                conn, project_id, confirmation_id, action,
                "confirmation_replayed", {"status": "executing"}, now,
            )
            conn.commit()
            return {
                "status": "submitting",
                "action": action,
                "confirmation_id": confirmation_id,
                "replayed": True,
            }
        if pending["status"] == "expired" or pending["expires_at"] <= now:
            conn.execute(
                "UPDATE agent_confirmations SET status='expired', updated_at=? "
                "WHERE confirmation_id=? AND status='pending'",
                (now, confirmation_id),
            )
            _append_confirmation_event(
                conn, project_id, confirmation_id, action,
                "preview_expired", {"reason": "ttl_elapsed"}, now,
            )
            conn.commit()
            raise ValueError("确认记录已过期；请重新生成预览并等待用户确认")
        if pending["status"] != "pending":
            conn.rollback()
            raise ValueError("该确认已结束；请重新生成预览")
        updated = conn.execute(
            "UPDATE agent_confirmations SET status='executing', updated_at=? "
            "WHERE confirmation_id=? AND status='pending'",
            (now, confirmation_id),
        )
        if updated.rowcount != 1:
            conn.rollback()
            raise ValueError("该确认正在处理；请查询任务状态，不要重复提交")
        _append_confirmation_event(
            conn, project_id, confirmation_id, action, "submission_started", {}, now,
        )
        conn.commit()

    try:
        result = submit()
    except Exception as exc:
        now = time.time()
        with _confirmation_connection() as conn:
            conn.execute(
                "UPDATE agent_confirmations SET status='failed', error_json=?, updated_at=? "
                "WHERE confirmation_id=? AND status='executing'",
                (
                    json.dumps({"message": str(exc)[:1000]}, ensure_ascii=False),
                    now,
                    confirmation_id,
                ),
            )
            _append_confirmation_event(
                conn, project_id, confirmation_id, action, "submission_failed",
                {"message": str(exc)[:1000]}, now,
            )
        raise

    now = time.time()
    result_json = json.dumps(result, ensure_ascii=False, default=str)
    with _confirmation_connection() as conn:
        conn.execute(
            "UPDATE agent_confirmations SET status='completed', result_json=?, "
            "error_json=NULL, updated_at=? WHERE confirmation_id=? AND status='executing'",
            (result_json, now, confirmation_id),
        )
        _append_confirmation_event(
            conn, project_id, confirmation_id, action, "submission_completed",
            {"result_available": True}, now,
        )
    return {"status": "submitted", "action": action, "submitted": result}


async def _call(project_id: int, operation: Callable[[], Any]) -> str:
    try:
        _scope(project_id)
        data = await asyncio.to_thread(operation)
        result = {"success": True, "project_id": project_id, "data": data, "error": None}
    except Exception as exc:
        result = {
            "success": False,
            "project_id": project_id,
            "data": None,
            "error": {"code": "QUERY_FAILED", "message": str(exc)},
        }
    return json.dumps(result, ensure_ascii=False, default=str)


def _offset(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        return max(0, int(cursor))
    except ValueError as exc:
        raise ValueError("cursor 需为非负整数") from exc


def _page(rows: list[Any], limit: int, cursor: str | None) -> dict[str, Any]:
    start = _offset(cursor)
    limit = max(1, min(int(limit), 200))
    selected = rows[start:start + limit]
    return {
        "items": selected,
        "count": len(selected),
        "next_cursor": str(start + limit) if start + limit < len(rows) else None,
    }


def _agent_event_page(
    project_id: int,
    cursor: str | None,
    limit: int,
    confirmation_id: str | None,
) -> dict[str, Any]:
    after_id = _offset(cursor)
    size = max(1, min(int(limit), 200))
    sql = (
        "SELECT event_id, confirmation_id, action, event_type, created_at, detail_json "
        "FROM agent_confirmation_events WHERE project_id=? AND event_id>?"
    )
    params: list[Any] = [project_id, after_id]
    if confirmation_id:
        sql += " AND confirmation_id=?"
        params.append(confirmation_id)
    sql += " ORDER BY event_id LIMIT ?"
    params.append(size + 1)
    with _confirmation_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    has_more = len(rows) > size
    items = []
    for row in rows[:size]:
        try:
            detail = json.loads(row["detail_json"] or "{}")
        except json.JSONDecodeError:
            detail = {}
        items.append({
            "event_id": row["event_id"],
            "confirmation_id": row["confirmation_id"],
            "action": row["action"],
            "event_type": row["event_type"],
            "created_at": row["created_at"],
            "detail": detail,
        })
    return {
        "items": items,
        "next_cursor": str(items[-1]["event_id"]) if items else str(after_id),
        "has_more": has_more,
    }


def _job_id(job_id: int | None, task_id: str | None) -> int:
    if job_id is not None:
        return int(job_id)
    match = re.search(r"(\d+)$", task_id or "")
    if not match:
        raise ValueError("请提供 job_id，或末尾包含数字的 task_id")
    return int(match.group(1))


@mcp.tool(name="agent_get_execution_events", description="按递增游标查询 Agent 训练、评测和部署确认事件。只读。")
async def agent_get_execution_events(
    project_id: int,
    cursor: str | None = None,
    limit: int = 50,
    confirmation_id: str | None = None,
) -> str:
    return await _call(
        project_id,
        lambda: _agent_event_page(project_id, cursor, limit, confirmation_id),
    )


@mcp.tool(name="dataset_list", description="查询数据集版本，可按名称搜索并分页。只读。")
async def dataset_list(
    project_id: int,
    limit: int = 50,
    cursor: str | None = None,
    q: str | None = None,
) -> str:
    """List dataset versions in the default project."""
    def query() -> dict[str, Any]:
        start = _offset(cursor)
        size = max(1, min(limit, 200))
        rows = _get("/workbench/api/datasets", {"limit": min(5000, start + size + 1), "q": q})
        return _page(rows, size, cursor)
    return await _call(project_id, query)


@mcp.tool(name="dataset_get", description="读取指定数据集版本详情。只读。")
async def dataset_get(project_id: int, dataset_version_id: int) -> str:
    return await _call(project_id, lambda: _get(f"/workbench/api/datasets/{dataset_version_id}"))


@mcp.tool(name="dataset_list_episodes", description="列出数据集版本中的 Episode，可分页。只读。")
async def dataset_list_episodes(
    project_id: int,
    dataset_version_id: int,
    limit: int = 50,
    cursor: str | None = None,
) -> str:
    def query() -> dict[str, Any]:
        start = _offset(cursor)
        size = max(1, min(limit, 200))
        rows = _get(f"/workbench/api/datasets/{dataset_version_id}/episodes", {"limit": min(10000, start + size + 1)})
        return _page(rows, size, cursor)
    return await _call(project_id, query)


@mcp.tool(name="dataset_get_schema", description="读取数据集的 LeRobot manifest 和特征 schema。只读。")
async def dataset_get_schema(project_id: int, dataset_version_id: int) -> str:
    return await _call(project_id, lambda: _get(f"/workbench/api/datasets/{dataset_version_id}/manifest"))


@mcp.tool(name="processing_quality_submit", description="创建数据集质检预览；用户确认后由 CPU Worker 检查并保存报告。")
async def processing_quality_submit(
    project_id: int,
    dataset_version_id: int,
    max_episodes: int = 5,
    confirmation_id: str | None = None,
) -> str:
    payload = {
        "dataset_version_id": dataset_version_id,
        "max_episodes": max_episodes,
    }
    return await _call(project_id, lambda: _submit_with_confirmation(
        project_id, "processing_quality_submit", payload, confirmation_id,
        lambda: _post("/workbench/api/processing/jobs", payload),
    ))


@mcp.tool(name="processing_quality_get", description="查询数据集质检 Worker Job 状态与摘要报告。只读。")
async def processing_quality_get(project_id: int, job_id: int) -> str:
    return await _call(project_id, lambda: _get(f"/workbench/api/processing/jobs/{job_id}"))


@mcp.tool(name="processing_clean_submit", description="预览 Episode 质量清洗方案；确认后由 CPU Worker 发布派生数据版本。")
async def processing_clean_submit(
    project_id: int,
    dataset_version_id: int,
    ops: list[str] | None = None,
    max_episodes: int = 200,
    processing_project_id: int | None = None,
    confirmation_id: str | None = None,
) -> str:
    payload = {
        "dataset_version_id": dataset_version_id,
        "ops": list(ops) if ops is not None else ["drop_any_finding"],
        "max_episodes": max_episodes,
        "apply": True,
        "project_id": processing_project_id,
    }
    return await _call(project_id, lambda: _submit_with_confirmation(
        project_id, "processing_clean_submit", payload, confirmation_id,
        lambda: _post("/workbench/api/processing/clean", payload),
    ))


@mcp.tool(name="processing_clean_get", description="查询质量清洗 Worker Job、生成的数据版本和处理报告。只读。")
async def processing_clean_get(project_id: int, job_id: int) -> str:
    return await _call(project_id, lambda: _get(f"/workbench/api/processing/jobs/{job_id}"))


@mcp.tool(name="dataset_materialize", description="生成 Episode 裁剪、重采样并物化为新 LeRobot 版本的预览；用户确认后由 Worker 执行。")
async def dataset_materialize(
    project_id: int,
    dataset_version_id: int,
    episode_indices: list[int] | None = None,
    crop_start_s: float = 0.0,
    crop_end_s: float | None = None,
    target_fps: int = 15,
    confirmation_id: str | None = None,
) -> str:
    payload = {
        "dataset_version_id": dataset_version_id,
        "episode_indices": episode_indices,
        "crop_start_s": crop_start_s,
        "crop_end_s": crop_end_s,
        "target_fps": target_fps,
        "project_id": project_id,
    }
    return await _call(project_id, lambda: _submit_with_confirmation(
        project_id, "dataset_materialize", payload, confirmation_id,
        lambda: _post("/workbench/api/processing/materialize", payload),
    ))


@mcp.tool(name="dataset_materialize_get", description="查询数据物化 Job、PipelineTask 和派生版本状态。只读。")
async def dataset_materialize_get(project_id: int, job_id: int) -> str:
    return await _call(project_id, lambda: _get(f"/workbench/api/processing/materialize/{job_id}"))


@mcp.tool(name="agent_cancel_confirmation", description="用户取消操作时关闭尚未执行的确认预览。不会取消已创建的任务。")
async def agent_cancel_confirmation(
    project_id: int,
    confirmation_id: str,
    action: str,
    payload: dict[str, Any],
) -> str:
    body = {"project_id": project_id, "action": action, "payload": payload}
    return await _call(project_id, lambda: _post(
        f"/workbench/api/agent/confirmations/{confirmation_id}/cancel", body,
    ))


@mcp.tool(name="training_get_job", description="查询训练或评测任务状态与配置。只读。")
async def training_get_job(project_id: int, job_id: int | None = None, task_id: str | None = None) -> str:
    return await _call(project_id, lambda: _get(f"/api/jobs/{_job_id(job_id, task_id)}"))


@mcp.tool(name="training_get_capabilities", description="查询当前可用训练后端、数据要求、资源需求、超参数范围及评测/部署支持。只读。")
async def training_get_capabilities(project_id: int) -> str:
    return await _call(project_id, lambda: _get("/workbench/api/training/capabilities"))


@mcp.tool(name="training_list_templates", description="按当前服务器可用能力列出 BC、ACT、OpenPI 训练模板、参数范围和资源需求。只读。")
async def training_list_templates(project_id: int, family: str | None = None) -> str:
    def list_templates() -> dict[str, Any]:
        catalog = _get("/workbench/api/training/capabilities")
        selected = family.strip().lower() if family else None
        items: list[dict[str, Any]] = []
        for backend in catalog.get("backends", []):
            backend_id = str(backend.get("id") or "").lower()
            if selected and selected not in {backend_id, str(backend.get("name") or "").lower()}:
                continue
            configs = backend.get("configs") or ["default"]
            for config_name in configs:
                items.append({
                    "template_id": f"{backend_id}/{config_name}",
                    "version": str(catalog.get("version") or "unknown"),
                    "family": backend_id,
                    "name": backend.get("name"),
                    "config": config_name,
                    "available": bool(backend.get("available")),
                    "status": backend.get("status"),
                    "supported_fields": [
                        item.get("name") for item in backend.get("parameters", [])
                        if item.get("name")
                    ],
                    "parameters": backend.get("parameters", []),
                    "resource": backend.get("resource", {}),
                    "dataset_requirements": backend.get("dataset_requirements", {}),
                    "evaluation": backend.get("evaluation", {}),
                    "deployment": backend.get("deployment", {}),
                    "unavailable_requirements": backend.get("unavailable_requirements", []),
                })
        return {
            "schema": catalog.get("schema"), "version": catalog.get("version"),
            "items": items, "total": len(items),
        }
    return await _call(project_id, list_templates)


def _config_field_error(field: str, message: str) -> dict[str, str]:
    return {"field": field, "message": message}


def _validate_parameter_value(value: Any, descriptor: dict[str, Any]) -> str | None:
    kind = descriptor.get("type")
    if kind == "integer":
        valid_type = type(value) is int
    elif kind == "number":
        valid_type = type(value) in {int, float}
    elif kind == "boolean":
        valid_type = type(value) is bool
    elif kind == "enum":
        valid_type = isinstance(value, str)
    elif kind == "integer|null":
        valid_type = value is None or type(value) is int
    else:
        valid_type = True
    if not valid_type:
        return f"类型应为 {kind}"
    if value is None:
        return None
    choices = descriptor.get("choices")
    if choices is not None and value not in choices:
        return "可选值为：" + ", ".join(map(str, choices))
    if type(value) in {int, float}:
        minimum = descriptor.get("minimum")
        maximum = descriptor.get("maximum")
        if minimum is not None and (value <= minimum if descriptor.get("minimum_exclusive") else value < minimum):
            relation = ">" if descriptor.get("minimum_exclusive") else ">="
            return f"数值需满足 {relation} {minimum}"
        if maximum is not None and value > maximum:
            return f"数值需 <= {maximum}"
    return None


@mcp.tool(name="training_validate_config", description="校验训练配置、数据集和资源估算；返回字段级错误及可审阅 JSON/YAML，不创建任务。")
async def training_validate_config(project_id: int, config: dict[str, Any]) -> str:
    def validate() -> dict[str, Any]:
        field_errors: list[dict[str, str]] = []
        warnings: list[str] = []
        backend_id = str(config.get("backend") or config.get("family") or "").strip().lower()
        if not backend_id:
            field_errors.append(_config_field_error("backend", "必填：bc、act 或 openpi"))
        catalog = _get("/workbench/api/training/capabilities")
        backend = next((item for item in catalog.get("backends", [])
                        if item.get("id") == backend_id), None)
        if backend_id and not backend:
            field_errors.append(_config_field_error("backend", f"未登记训练后端：{backend_id}"))
        elif backend and not backend.get("available"):
            missing = backend.get("unavailable_requirements", [])
            field_errors.append(_config_field_error(
                "backend", "当前不可用" + ("：" + ", ".join(missing) if missing else "")
            ))

        dataset_version_id = config.get("dataset_version_id")
        dataset_info = None
        if type(dataset_version_id) is not int or dataset_version_id < 1:
            field_errors.append(_config_field_error("dataset_version_id", "必填正整数"))
        else:
            try:
                dataset_info = _get(f"/workbench/api/datasets/{dataset_version_id}")
                if not dataset_info.get("has_manifest"):
                    field_errors.append(_config_field_error("dataset_version_id", "数据版本缺少 LeRobot manifest"))
                if int(dataset_info.get("episodes") or 0) < 1:
                    field_errors.append(_config_field_error("dataset_version_id", "数据版本没有 Episode"))
                if backend_id in {"openpi", "act"} and not dataset_info.get("cameras"):
                    field_errors.append(_config_field_error("dataset_version_id", f"{backend_id} 需要相机图像"))
                if backend_id == "bc" and not dataset_info.get("state_dim"):
                    warnings.append("数据详情未提供 observation.state 维度，提交前由训练 API 再校验")
            except Exception as exc:
                field_errors.append(_config_field_error("dataset_version_id", f"无法读取数据版本：{str(exc)[:240]}"))

        reserved = {
            "backend", "family", "dataset_version_id", "template_id", "parameters",
            "experiment_name", "exp_name",
        }
        parameters = config.get("parameters")
        if parameters is None:
            parameters = {key: value for key, value in config.items() if key not in reserved}
        if not isinstance(parameters, dict):
            field_errors.append(_config_field_error("parameters", "必须是对象"))
            parameters = {}

        normalized: dict[str, Any] = {}
        resource: dict[str, Any] = {}
        if backend:
            descriptors = {
                str(item.get("name")): item
                for item in backend.get("parameters", []) if item.get("name")
            }
            for name, descriptor in descriptors.items():
                if "default" in descriptor:
                    normalized[name] = descriptor["default"]
            if config.get("template_id") and backend_id == "openpi":
                template_config = str(config["template_id"]).rsplit("/", 1)[-1]
                parameters.setdefault("config", template_config)
            for name, value in parameters.items():
                descriptor = descriptors.get(str(name))
                field_path = f"parameters.{name}"
                if not descriptor:
                    field_errors.append(_config_field_error(field_path, f"{backend_id} 不支持此参数"))
                    continue
                message = _validate_parameter_value(value, descriptor)
                if message:
                    field_errors.append(_config_field_error(field_path, message))
                else:
                    normalized[name] = value
            if backend_id == "openpi" and "config" in normalized:
                if normalized["config"] not in (backend.get("configs") or []):
                    field_errors.append(_config_field_error("parameters.config", "该 OpenPI 配置不在当前能力目录中"))
            gpu_count = int(normalized.get("gpus", 1)) if backend_id == "openpi" else 1
            gpu_count = gpu_count if (backend_id != "bc" or normalized.get("device") == "cuda") else 0
            requested_per_gpu = int(normalized.get("gpu_memory_mb", 0)) if gpu_count else 0
            minimum_per_gpu = int((backend.get("resource") or {}).get("minimum_memory_mb") or 0)
            resource = {
                "gpu_count": gpu_count,
                "memory_per_gpu_mb": requested_per_gpu,
                "total_requested_gpu_memory_mb": requested_per_gpu * gpu_count,
                "minimum_memory_per_gpu_mb": minimum_per_gpu,
                "estimated_total_minimum_gpu_memory_mb": minimum_per_gpu * gpu_count,
                "cpu_supported": bool((backend.get("resource") or {}).get("cpu_supported")),
                "checked_at_submission": "提交时按实时 GPU 空闲显存再准入",
            }
            if gpu_count and requested_per_gpu < minimum_per_gpu:
                field_errors.append(_config_field_error(
                    "parameters.gpu_memory_mb", f"每卡至少需要 {minimum_per_gpu} MB"
                ))
            if backend_id == "openpi" and gpu_count and int(normalized.get("batch_size", 1)) % gpu_count:
                field_errors.append(_config_field_error("parameters.batch_size", "batch_size 必须能被 gpus 整除"))

        if backend_id == "openpi":
            normalized.setdefault("config", "4items_rtc_v2_lora")
        exp_name = str(config.get("exp_name") or config.get("experiment_name")
                       or f"agent-{backend_id or 'training'}-ds{dataset_version_id or 'unknown'}")
        if backend and dataset_version_id and not any(item["field"] in {"backend", "dataset_version_id"} for item in field_errors):
            if backend_id == "openpi":
                api_payload = {"dataset_version_id": dataset_version_id, "config": normalized.get("config"),
                               "exp_name": exp_name, **{k: normalized[k] for k in (
                                   "gpus", "gpu_memory_mb", "batch_size", "steps", "learning_rate",
                                   "lora_rank", "freeze_strategy", "num_workers") if k in normalized}}
            elif backend_id == "act":
                api_payload = {"dataset_version_id": dataset_version_id, "exp_name": exp_name,
                               **{k: normalized[k] for k in (
                                   "batch_size", "steps", "learning_rate", "chunk_size", "n_action_steps",
                                   "use_amp", "gpu_memory_mb", "gpu_index", "num_workers",
                                   "pretrained_backbone") if k in normalized}}
            else:
                api_payload = {"dataset_version_id": dataset_version_id, "epochs": normalized.get("epochs"),
                               "lr": normalized.get("learning_rate"),
                               **{k: normalized[k] for k in (
                                   "batch_size", "holdout_episodes", "gpu_memory_mb", "device",
                                   "optimizer", "weight_decay", "seed") if k in normalized}}
            final_config = {
                "capability_version": catalog.get("version"), "backend": backend_id,
                "template_id": config.get("template_id") or f"{backend_id}/{normalized.get('config', 'default')}",
                "dataset_version_id": dataset_version_id, "dataset": dataset_info,
                "experiment_name": exp_name, "parameters": normalized,
                "resource_estimate": resource, "api_route": backend.get("train_route"),
                "api_payload": api_payload,
            }
        else:
            final_config = None
        serialized = json.dumps(final_config, ensure_ascii=False, indent=2) if final_config else None
        return {
            "valid": not field_errors, "capability_version": catalog.get("version"),
            "backend": backend_id or None, "field_errors": field_errors,
            "warnings": warnings, "normalized_parameters": normalized,
            "resource_estimate": resource, "configuration": final_config,
            "config_json": serialized, "config_yaml": serialized,
            "config_yaml_note": "JSON 是 YAML 1.2 的有效子集；此配置可直接保存为 .json 或 .yaml",
        }
    return await _call(project_id, validate)


@mcp.tool(name="training_submit", description="创建训练任务预览。BC 可选 CPU/CUDA。必须先展示参数并等待用户明确确认；仅在后续调用带回 confirmation_id 时才提交。")
async def training_submit(
    project_id: int,
    dataset_version_id: int,
    backend: str = "openpi",
    config: str = "4items_rtc_v2_lora",
    exp_name: str = "",
    batch_size: int = 8,
    steps: int = 1000,
    learning_rate: float = 1e-5,
    gpus: int = 1,
    gpu_memory_mb: int = 22000,
    lora_rank: int = 0,
    freeze_strategy: str = "preset",
    num_workers: int = 4,
    epochs: int = 5,
    holdout_episodes: int = 3,
    optimizer: str = "adamw",
    weight_decay: float = 1e-4,
    seed: int = 42,
    chunk_size: int = 50,
    n_action_steps: int = 50,
    use_amp: bool = False,
    pretrained_backbone: bool = True,
    gpu_index: int | None = None,
    device: str = "cuda",
    confirmation_id: str | None = None,
) -> str:
    def submit() -> Any:
        selected = backend.strip().lower()
        catalog, capability = _training_capability(selected)
        run_name = exp_name.strip() or f"agent-{selected}-ds{dataset_version_id}"
        if selected == "openpi":
            payload = {
                "dataset_version_id": dataset_version_id, "config": config,
                "exp_name": run_name, "gpus": gpus,
                "gpu_memory_mb": gpu_memory_mb, "batch_size": batch_size,
                "steps": steps, "learning_rate": learning_rate,
                "lora_rank": lora_rank, "freeze_strategy": freeze_strategy,
                "num_workers": num_workers,
            }
        elif selected == "act":
            payload = {
                "dataset_version_id": dataset_version_id, "exp_name": run_name,
                "batch_size": batch_size, "steps": steps,
                "learning_rate": learning_rate, "chunk_size": chunk_size,
                "n_action_steps": n_action_steps, "use_amp": use_amp,
                "gpu_memory_mb": gpu_memory_mb, "gpu_index": gpu_index,
                "num_workers": num_workers, "pretrained_backbone": pretrained_backbone,
            }
        elif selected == "bc":
            payload = {
                "dataset_version_id": dataset_version_id, "epochs": epochs,
                "lr": learning_rate, "batch_size": batch_size,
                "holdout_episodes": holdout_episodes, "gpu_memory_mb": gpu_memory_mb,
                "device": device, "optimizer": optimizer,
                "weight_decay": weight_decay, "seed": seed,
            }
        else:
            raise ValueError(f"未实现训练参数映射：{selected}")
        preview_payload = {
            "capability_version": catalog.get("version"),
            "backend": selected,
            "request": payload,
        }
        return _submit_with_confirmation(
            project_id, f"training_submit:{selected}", preview_payload, confirmation_id,
            lambda: {
                "backend": selected,
                "capability_version": catalog.get("version"),
                "job": _post(capability["train_route"], payload),
            },
        )
    return await _call(project_id, submit)


@mcp.tool(name="workflow_training_submit", description="创建可恢复的数据质检→训练→MLflow 工作流，可选加入独立数据版本离线评测和部署（OpenPI、ACT、BC）。预览完整步骤后用户一次确认；后续由 workflow_advance 推进 Worker、评测与部署。")
async def workflow_training_submit(
    project_id: int,
    dataset_version_id: int,
    backend: str = "openpi",
    config: str = "4items_rtc_v2_lora",
    exp_name: str = "",
    batch_size: int = 8,
    steps: int = 1000,
    learning_rate: float = 1e-5,
    gpus: int = 1,
    gpu_memory_mb: int = 22000,
    lora_rank: int = 0,
    freeze_strategy: str = "preset",
    num_workers: int = 4,
    epochs: int = 5,
    holdout_episodes: int = 3,
    optimizer: str = "adamw",
    weight_decay: float = 1e-4,
    seed: int = 42,
    chunk_size: int = 50,
    n_action_steps: int = 50,
    use_amp: bool = False,
    pretrained_backbone: bool = True,
    gpu_index: int | None = None,
    device: str = "cuda",
    max_quality_episodes: int = 20,
    evaluation_dataset_version_id: int | None = None,
    evaluation_samples: int = 8,
    deploy_after_evaluation: bool = False,
    confirmation_id: str | None = None,
) -> str:
    """Persist a preflight -> approval -> quality -> training -> optional eval/deploy workflow."""
    selected = backend.strip().lower()
    action = "workflow_training_submit"
    requested_name = exp_name.strip()
    saved_payload: dict[str, Any] | None = None
    if confirmation_id and not requested_name:
        with _confirmation_connection() as conn:
            saved = conn.execute(
                "SELECT payload_json FROM agent_confirmations WHERE confirmation_id=? AND project_id=? AND action=?",
                (confirmation_id, project_id, action),
            ).fetchone()
        if saved:
            saved_payload = json.loads(saved["payload_json"])
            requested_name = str(saved_payload.get("training", {}).get("exp_name", ""))
    elif confirmation_id:
        with _confirmation_connection() as conn:
            saved = conn.execute(
                "SELECT payload_json FROM agent_confirmations WHERE confirmation_id=? AND project_id=? AND action=?",
                (confirmation_id, project_id, action),
            ).fetchone()
        if saved:
            saved_payload = json.loads(saved["payload_json"])
    if confirmation_id and saved_payload:
        saved_eval = saved_payload.get("evaluation") or {}
        if evaluation_dataset_version_id is None and saved_eval:
            evaluation_dataset_version_id = int(saved_eval["dataset_version_id"])
            evaluation_samples = int(saved_eval.get("samples", evaluation_samples))
        elif (saved_eval and evaluation_dataset_version_id == int(saved_eval.get("dataset_version_id", -1))
              and evaluation_samples == 8):
            evaluation_samples = int(saved_eval.get("samples", evaluation_samples))
        if saved_payload.get("deploy_after_evaluation") and not deploy_after_evaluation:
            deploy_after_evaluation = True
    run_name = requested_name or f"workflow-{selected}-ds{dataset_version_id}-{secrets.token_hex(4)}"
    if selected == "openpi":
        train_config = {
            "dataset_version_id": dataset_version_id, "config": config,
            "exp_name": run_name, "gpus": gpus, "gpu_memory_mb": gpu_memory_mb,
            "batch_size": batch_size, "steps": steps, "learning_rate": learning_rate,
            "lora_rank": lora_rank, "freeze_strategy": freeze_strategy,
            "num_workers": num_workers,
        }
    elif selected == "act":
        train_config = {
            "dataset_version_id": dataset_version_id, "exp_name": run_name,
            "batch_size": batch_size, "steps": steps, "learning_rate": learning_rate,
            "chunk_size": chunk_size, "n_action_steps": n_action_steps,
            "use_amp": use_amp, "gpu_memory_mb": gpu_memory_mb,
            "gpu_index": gpu_index, "num_workers": num_workers,
            "pretrained_backbone": pretrained_backbone,
        }
    elif selected == "bc":
        train_config = {
            "dataset_version_id": dataset_version_id, "epochs": epochs,
            "lr": learning_rate, "batch_size": batch_size,
            "holdout_episodes": holdout_episodes, "gpu_memory_mb": gpu_memory_mb,
            "device": device, "optimizer": optimizer,
            "weight_decay": weight_decay, "seed": seed,
        }
    else:
        train_config = {}

    def operation() -> dict[str, Any]:
        catalog, _capability = _training_capability(selected)
        if not 1 <= int(max_quality_episodes) <= 200:
            raise ValueError("max_quality_episodes 必须在 1–200 之间")
        if not 1 <= int(evaluation_samples) <= 32:
            raise ValueError("evaluation_samples 必须在 1–32 之间")
        if deploy_after_evaluation and not evaluation_dataset_version_id:
            raise ValueError("部署前需要先配置独立数据版本离线评测")
        evaluation = (
            {"dataset_version_id": int(evaluation_dataset_version_id), "samples": int(evaluation_samples)}
            if evaluation_dataset_version_id else None
        )
        if confirmation_id:
            with _confirmation_connection() as conn:
                row = conn.execute(
                    "SELECT project_id, action, payload_json FROM agent_confirmations WHERE confirmation_id=?",
                    (confirmation_id,),
                ).fetchone()
            if not row or row["project_id"] != project_id or row["action"] != action:
                raise ValueError("工作流确认不存在；请重新生成预览")
            payload = json.loads(row["payload_json"])
            if (payload.get("capability_version") != catalog.get("version")
                    or payload.get("backend") != selected
                    or payload.get("training") != train_config
                    or payload.get("quality", {}).get("max_episodes") != max_quality_episodes
                    or payload.get("evaluation") != evaluation
                    or bool(payload.get("deploy_after_evaluation")) != bool(deploy_after_evaluation)):
                raise ValueError("训练参数与工作流预览不一致；请使用原参数确认")
        else:
            dataset = _get(f"/workbench/api/datasets/{dataset_version_id}")
            if not dataset.get("has_manifest"):
                raise ValueError("数据版本缺少 LeRobot manifest，无法进入训练工作流")
            evaluation_dataset = None
            if evaluation:
                if int(evaluation["dataset_version_id"]) == int(dataset_version_id):
                    raise ValueError("离线评测请指定独立的数据版本，不要用训练版本自评")
                evaluation_dataset = _get(
                    f"/workbench/api/datasets/{evaluation['dataset_version_id']}"
                )
                if not evaluation_dataset.get("has_manifest"):
                    raise ValueError("离线评测数据版本缺少 LeRobot manifest")
            cards = _get("/api/resources/gpus", {"refresh": "true"})
            if not isinstance(cards, list):
                cards = []
            training_gpu_required = selected != "bc" or device == "cuda"
            eval_gpu_memory_mb = 18000 if evaluation and selected == "openpi" else 0
            preflight_gpu_memory_mb = max(
                int(gpu_memory_mb) if training_gpu_required else 0,
                eval_gpu_memory_mb,
            )
            usable = [
                row for row in cards
                if str(row.get("state", "")).lower() in {"free", "partial"}
                and int(row.get("memory_free_mb") or 0) >= preflight_gpu_memory_mb
            ]
            payload = {
                "workflow": "training_and_experiment_report",
                "workflow_version": 2,
                "capability_version": catalog.get("version"),
                "backend": selected,
                "training": train_config,
                "evaluation": evaluation,
                "deploy_after_evaluation": bool(deploy_after_evaluation),
                "quality": {
                    "dataset_version_id": dataset_version_id,
                    "max_episodes": max_quality_episodes,
                },
                "preflight": {
                    "dataset": {
                        "dataset_version_id": dataset.get("id"),
                        "name": dataset.get("dataset"), "version": dataset.get("version"),
                        "format": dataset.get("format"), "status": dataset.get("status"),
                        "episodes": dataset.get("episodes"), "fps": dataset.get("fps"),
                        "state_dim": dataset.get("state_dim"),
                        "action_dim": dataset.get("action_dim"),
                        "cameras": dataset.get("cameras", []),
                        "manifest_ready": True,
                    },
                    "evaluation_dataset": ({
                        "dataset_version_id": evaluation_dataset.get("id"),
                        "name": evaluation_dataset.get("dataset"),
                        "version": evaluation_dataset.get("version"),
                        "episodes": evaluation_dataset.get("episodes"),
                        "manifest_ready": True,
                    } if evaluation_dataset else None),
                    "resources": {
                        "gpu_count": len(cards), "usable_gpu_count": len(usable),
                        "usable_gpu_indexes": [row.get("index") for row in usable],
                        "required_gpu_memory_mb": preflight_gpu_memory_mb,
                        "training_gpu_required": training_gpu_required,
                        "evaluation_gpu_memory_mb": eval_gpu_memory_mb,
                        "queue_if_busy": True,
                    },
                },
                "steps": [
                    {"id": "preflight", "status": "completed"},
                    {"id": "approval", "status": "awaiting_user_confirmation"},
                    {"id": "data_quality", "status": "blocked"},
                    {"id": "training", "status": "blocked"},
                    {"id": "evaluation", "status": "blocked" if evaluation else "skipped"},
                    {"id": "deployment", "status": "blocked" if deploy_after_evaluation else "skipped"},
                    {"id": "experiment_report", "status": "blocked"},
                ],
            }

        def start_workflow() -> dict[str, Any]:
            if not confirmation_id:
                return {}
            with _confirmation_connection() as conn:
                existing = conn.execute(
                    "SELECT status, result_json FROM agent_workflows WHERE workflow_id=?",
                    (confirmation_id,),
                ).fetchone()
            if existing:
                return json.loads(existing["result_json"] or "{}")
            now = time.time()
            steps = [
                {"id": "preflight", "status": "completed", "result": payload.get("preflight", {})},
                {"id": "approval", "status": "completed"},
                {"id": "data_quality", "status": "submitting"},
                {"id": "training", "status": "blocked"},
                {"id": "evaluation", "status": "blocked" if payload.get("evaluation") else "skipped"},
                {"id": "deployment", "status": "blocked" if payload.get("deploy_after_evaluation") else "skipped"},
                {"id": "experiment_report", "status": "blocked"},
            ]
            result = {
                "workflow_id": confirmation_id, "status": "quality_submitting",
            }
            with _confirmation_connection() as conn:
                conn.execute(
                    "INSERT INTO agent_workflows "
                    "(workflow_id, project_id, kind, name, status, payload_json, steps_json, "
                    "result_json, error_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                    (
                        confirmation_id, project_id, payload["workflow"], run_name,
                        "quality_submitting",
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        json.dumps(steps, ensure_ascii=False),
                        json.dumps(result, ensure_ascii=False),
                        now, now,
                    ),
                )
            try:
                quality_job = _post("/workbench/api/processing/jobs", {
                    **payload["quality"], "workflow_id": confirmation_id,
                })
                quality_job_id = int(quality_job["job_id"])
            except Exception as exc:
                recovered = _workflow_find_job(confirmation_id, ["quality"])
                if not recovered:
                    _workflow_store(
                        confirmation_id, "failed", steps,
                        {**result, "error": str(exc)[:1000]},
                        {"message": str(exc)[:1000]},
                    )
                    raise
                quality_job = recovered
                quality_job_id = int(recovered["job_id"])
            steps[2] = {"id": "data_quality", "status": "queued", "job_id": quality_job_id}
            result = {
                "workflow_id": confirmation_id, "status": "quality_running",
                "quality_job_id": quality_job_id, "quality_job": quality_job,
                "next_action": "workflow_get",
            }
            _workflow_store(confirmation_id, "quality_running", steps, result)
            return result

        return _submit_with_confirmation(
            project_id, action, payload, confirmation_id,
            start_workflow,
        )

    result = json.loads(await _call(project_id, operation))
    data = result.get("data")
    if isinstance(data, dict):
        preview = data.get("confirmation_id")
        if preview:
            data["workflow_id"] = preview
        elif confirmation_id:
            data["workflow_id"] = confirmation_id
    return json.dumps(result, ensure_ascii=False)


def _workflow_store(
    workflow_id: str,
    status: str,
    steps: list[dict[str, Any]],
    result: dict[str, Any],
    error: dict[str, Any] | None = None,
) -> None:
    with _confirmation_connection() as conn:
        conn.execute(
            "UPDATE agent_workflows SET status=?, steps_json=?, result_json=?, error_json=?, updated_at=? "
            "WHERE workflow_id=?",
            (
                status, json.dumps(steps, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False, default=str),
                json.dumps(error, ensure_ascii=False) if error else None,
                time.time(), workflow_id,
            ),
        )


def _workflow_claim_stale(workflow_id: str, expected_status: str, minimum_age: float = 90.0) -> bool:
    now = time.time()
    with _confirmation_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status, updated_at FROM agent_workflows WHERE workflow_id=?",
            (workflow_id,),
        ).fetchone()
        if (not row or row["status"] != expected_status
                or now - float(row["updated_at"]) < minimum_age):
            conn.rollback()
            return False
        conn.execute(
            "UPDATE agent_workflows SET updated_at=? WHERE workflow_id=? AND status=?",
            (now, workflow_id, expected_status),
        )
        conn.commit()
        return True


def _workflow_read(project_id: int, workflow_id: str) -> sqlite3.Row:
    with _confirmation_connection() as conn:
        row = conn.execute(
            "SELECT * FROM agent_workflows WHERE workflow_id=? AND project_id=?",
            (workflow_id, project_id),
        ).fetchone()
    if not row:
        raise ValueError("工作流不存在")
    return row


def _workflow_find_job(workflow_id: str, kinds: list[str]) -> dict[str, Any] | None:
    placeholders = ",".join("?" for _ in kinds)
    with _confirmation_connection() as conn:
        rows = conn.execute(
            f"SELECT id, kind, status, spec FROM jobs WHERE kind IN ({placeholders}) "
            "ORDER BY id DESC LIMIT 500",
            kinds,
        ).fetchall()
    for row in rows:
        try:
            spec = json.loads(row["spec"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if spec.get("workflow_id") == workflow_id:
            return {"job_id": row["id"], "kind": row["kind"], "status": row["status"]}
    return None


def _workflow_training_request(payload: dict[str, Any], workflow_id: str) -> dict[str, Any]:
    backend = payload["backend"]
    catalog, capability = _training_capability(backend)
    if payload.get("capability_version") != catalog.get("version"):
        raise ValueError("训练能力版本已更新；请重新生成工作流预览")
    training = {**payload["training"], "workflow_id": workflow_id}
    return _post(capability["train_route"], training)


def _workflow_model_dir(training_job_id: int) -> str:
    job = _get(f"/workbench/api/jobs/{training_job_id}")
    spec = job.get("spec") or {}
    model_dir = spec.get("exp_dir") or spec.get("model_dir") or spec.get("checkpoint_path")
    if not model_dir:
        raise ValueError(f"训练 Job {training_job_id} 未登记可评测模型目录")
    return str(model_dir)


def _workflow_evaluation_request(
    payload: dict[str, Any], workflow_id: str, model_dir: str,
) -> dict[str, Any]:
    evaluation = payload.get("evaluation") or {}
    if not evaluation:
        raise ValueError("工作流未配置离线评测数据版本")
    return _post("/workbench/api/evaluations", {
        **evaluation, "model_dir": model_dir, "workflow_id": workflow_id,
    })


def _workflow_find_deployment(model_dir: str, after_id: int = 0) -> dict[str, Any] | None:
    target = os.path.normpath(str(model_dir))
    rows = _get("/workbench/api/deployments")
    for row in rows if isinstance(rows, list) else []:
        if int(row.get("id") or 0) <= int(after_id):
            continue
        deployed = os.path.normpath(str(row.get("model_dir") or ""))
        try:
            same_model = os.path.commonpath((target, deployed)) == target
        except ValueError:
            same_model = False
        if (os.path.normpath(str(row.get("model_dir") or "")) == target
                or same_model) and str(row.get("status", "")).lower() == "running":
            return row
    return None


def _workflow_start_deployment(model_dir: str, after_id: int = 0) -> dict[str, Any]:
    found = _workflow_find_deployment(model_dir, after_id)
    if found:
        return found
    try:
        return _post("/workbench/api/deployments", {"model_dir": model_dir}, timeout=360)
    except Exception:
        recovered = _workflow_find_deployment(model_dir, after_id)
        if recovered:
            return recovered
        raise


def _workflow_experiment_report(training_job_id: int) -> dict[str, Any]:
    data = _get("/workbench/api/experiments", {"limit": 20, "job_id": training_job_id})
    return {
        "mlflow_available": data.get("mlflow_available", False),
        "count": data.get("count", 0),
        "runs": [
            {
                "run_id": run.get("run_id"), "status": run.get("status"),
                "params": run.get("params", {}), "metrics": run.get("metrics", {}),
                "history": {
                    key: points[-20:] if isinstance(points, list) else points
                    for key, points in (run.get("history") or {}).items()
                },
            }
            for run in data.get("runs", [])
        ],
    }


def _workflow_job_status(job_id: int, *, quality: bool = False) -> dict[str, Any]:
    path = f"/workbench/api/processing/jobs/{job_id}" if quality else f"/api/jobs/{job_id}"
    return _get(path)


def _workflow_snapshot(project_id: int, workflow_id: str) -> dict[str, Any]:
    row = _workflow_read(project_id, workflow_id)
    payload = json.loads(row["payload_json"] or "{}")
    result = json.loads(row["result_json"] or "{}")
    steps = json.loads(row["steps_json"] or "[]")
    state = row["status"]
    quality_job_id = result.get("quality_job_id")
    training_job_id = result.get("training_job_id")
    evaluation_job_id = result.get("evaluation_job_id")
    deployment_id = result.get("deployment_id")
    quality = _workflow_job_status(int(quality_job_id), quality=True) if quality_job_id else None
    training = _workflow_job_status(int(training_job_id)) if training_job_id else None
    evaluation = _workflow_job_status(int(evaluation_job_id)) if evaluation_job_id else None
    quality_report = (quality or {}).get("report") or {}
    findings = quality_report.get("findings") or []
    quality_summary = None
    if quality:
        quality_summary = {
            "job_id": quality_job_id, "status": quality.get("status"),
            "dataset_version_id": quality.get("dataset_version_id"),
            "episodes_scanned": quality_report.get("episodes_scanned", quality_report.get("episodes")),
            "findings_count": quality_report.get("findings_count", len(findings)),
            "findings": findings[:10], "error": quality.get("error"),
        }
    training_status = str((training or {}).get("status") or "blocked").lower()
    evaluation_status = str((evaluation or {}).get("status") or "blocked").lower()
    evaluation_report = None
    if evaluation and evaluation_status == "completed":
        report = _get(f"/workbench/api/evaluations/{int(evaluation_job_id)}/report")
        evaluation_report = {
            key: value for key, value in report.items()
            if isinstance(value, (str, int, float, bool)) or key in {"metrics", "baseline", "summary"}
        }
    deployments = _get("/workbench/api/deployments") if deployment_id else []
    deployment = next((
        item for item in deployments
        if isinstance(item, dict) and int(item.get("id") or 0) == int(deployment_id)
    ), None)
    deployment_lineage = None
    if deployment:
        deployment_lineage = _get(f"/workbench/api/deployments/{int(deployment_id)}/lineage")
    experiments = None
    if training and training_status in {"completed", "failed", "cancelled"}:
        data = _get("/workbench/api/experiments", {"limit": 20, "job_id": int(training_job_id)})
        experiments = {
            "mlflow_available": data.get("mlflow_available", False),
            "count": data.get("count", 0),
            "runs": [
                {
                    "run_id": run.get("run_id"), "status": run.get("status"),
                    "params": run.get("params", {}), "metrics": run.get("metrics", {}),
                    "history": {
                        key: points[-20:] if isinstance(points, list) else points
                        for key, points in (run.get("history") or {}).items()
                    },
                }
                for run in data.get("runs", [])
            ],
        }
    view_status = state
    next_action = None
    if state == "quality_running" and quality and quality.get("status") == "completed":
        view_status, next_action = "ready_to_train", "workflow_advance"
    elif state == "quality_submitting":
        next_action = "workflow_advance"
    elif state == "quality_review_required":
        next_action = "review_findings_then_workflow_advance"
    elif state == "training_running" and training_status == "completed":
        view_status = "ready_for_evaluation" if payload.get("evaluation") else "ready_for_report"
        next_action = "workflow_advance"
    elif state == "training_running" and training_status in {"failed", "cancelled"}:
        view_status, next_action = training_status, "workflow_advance"
    elif state == "evaluation_running" and evaluation_status == "completed":
        view_status = "ready_for_deployment" if payload.get("deploy_after_evaluation") else "ready_for_report"
        next_action = "workflow_advance"
    elif state == "evaluation_running" and evaluation_status in {"failed", "cancelled"}:
        view_status, next_action = evaluation_status, "workflow_advance"
    elif state in {"training_submitting", "evaluation_submitting", "deployment_submitting", "reporting"}:
        next_action = "workflow_advance"
    elif state in {"quality_running", "training_running", "evaluation_running"}:
        next_action = "workflow_advance"

    step_map = {item.get("id"): item for item in steps}
    if quality_summary:
        step_map["data_quality"] = {
            "id": "data_quality", "status": quality.get("status"),
            "job_id": quality_job_id, "result": quality_summary,
        }
    if training:
        step_map["training"] = {
            "id": "training", "status": training_status,
            "job_id": training_job_id, "result": training,
        }
    if evaluation:
        step_map["evaluation"] = {
            "id": "evaluation", "status": evaluation_status,
            "job_id": evaluation_job_id, "dataset_version_id": (payload.get("evaluation") or {}).get("dataset_version_id"),
            "report": evaluation_report,
        }
    if deployment_id:
        step_map["deployment"] = {
            "id": "deployment", "status": (deployment or {}).get("status", "unknown"),
            "deployment_id": deployment_id, "endpoint": (deployment or {}).get("endpoint"),
            "lineage": deployment_lineage,
        }
    if experiments is not None:
        step_map["experiment_report"] = {
            "id": "experiment_report",
            "status": "completed" if experiments.get("runs") else "unavailable",
            "result": experiments,
        }
    ordered = [step_map[key] for key in (
        "preflight", "approval", "data_quality", "training", "evaluation",
        "deployment", "experiment_report"
    ) if key in step_map]
    return {
        "workflow_id": workflow_id, "kind": row["kind"], "name": row["name"],
        "status": view_status, "stored_status": state, "steps": ordered,
        "training_config": payload.get("training"), "quality": quality_summary,
        "training": training, "evaluation": evaluation, "evaluation_report": evaluation_report,
        "deployment": deployment, "deployment_lineage": deployment_lineage,
        "experiments": experiments, "result": result,
        "error": json.loads(row["error_json"]) if row["error_json"] else None,
        "next_action": next_action,
    }


@mcp.tool(name="workflow_create_draft", description="保存可审阅的多步研发 Pipeline 草稿并生成 JSON/YAML；仅保存草稿，不创建 Job、不启动训练。")
async def workflow_create_draft(project_id: int, goal: str, steps: list[dict[str, Any]]) -> str:
    def create_draft() -> dict[str, Any]:
        clean_goal = goal.strip()
        if not clean_goal or len(clean_goal) > 500:
            raise ValueError("goal 必须为 1–500 个字符")
        if not isinstance(steps, list) or not 1 <= len(steps) <= 50:
            raise ValueError("steps 需要包含 1–50 个步骤")
        normalized: list[dict[str, Any]] = []
        ids: set[str] = set()
        for index, item in enumerate(steps, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"steps[{index - 1}] 必须是对象")
            step_id = str(item.get("step_id") or item.get("id") or f"step_{index}").strip()
            tool_name = str(item.get("tool") or "").strip()
            if not step_id or len(step_id) > 128 or step_id in ids:
                raise ValueError(f"steps[{index - 1}].step_id 为空、过长或重复")
            if not tool_name or len(tool_name) > 128:
                raise ValueError(f"steps[{index - 1}].tool 必填，最多 128 个字符")
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError(f"steps[{index - 1}].arguments 必须是对象")
            depends_on = item.get("depends_on", [])
            if not isinstance(depends_on, list) or any(not isinstance(dep, str) for dep in depends_on):
                raise ValueError(f"steps[{index - 1}].depends_on 必须是步骤 ID 数组")
            digest = hashlib.sha256(
                json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
                .encode("utf-8")
            ).hexdigest()
            normalized.append({
                "step_id": step_id, "tool": tool_name,
                "arguments": arguments,
                "arguments_digest": str(item.get("arguments_digest") or digest),
                "depends_on": list(dict.fromkeys(depends_on)),
            })
            ids.add(step_id)
        for item in normalized:
            missing = [dep for dep in item["depends_on"] if dep not in ids]
            if missing:
                raise ValueError(f"{item['step_id']}.depends_on 引用了不存在的步骤：{', '.join(missing)}")
            if item["step_id"] in item["depends_on"]:
                raise ValueError(f"{item['step_id']} 不能依赖自身")
        # Reject dependency cycles so the saved draft always has a runnable order.
        unresolved = {item["step_id"]: set(item["depends_on"]) for item in normalized}
        ready: list[str] = []
        visited: set[str] = set()
        while True:
            ready = [step_id for step_id, deps in unresolved.items()
                     if step_id not in visited and deps.issubset(visited)]
            if not ready:
                break
            visited.update(ready)
        if len(visited) != len(normalized):
            raise ValueError("Pipeline 步骤存在依赖循环")

        draft_id = secrets.token_urlsafe(18)
        now = time.time()
        document = {"draft_id": draft_id, "revision": 1, "status": "draft",
                    "project_id": project_id, "goal": clean_goal, "steps": normalized}
        with _confirmation_connection() as conn:
            conn.execute(
                "INSERT INTO agent_pipeline_drafts "
                "(draft_id, project_id, goal, revision, steps_json, created_at, updated_at) "
                "VALUES (?, ?, ?, 1, ?, ?, ?)",
                (draft_id, project_id, clean_goal,
                 json.dumps(normalized, ensure_ascii=False, default=str), now, now),
            )
        serialized = json.dumps(document, ensure_ascii=False, indent=2)
        return {
            **document, "task_ids": [], "job_created": False,
            "config_json": serialized, "config_yaml": serialized,
            "config_yaml_note": "JSON 是 YAML 1.2 的有效子集；草稿可保存为 .json 或 .yaml",
        }
    return await _call(project_id, create_draft)


def _pipeline_draft_training_payload(project_id: int, draft_id: str) -> dict[str, Any]:
    with _confirmation_connection() as conn:
        row = conn.execute(
            "SELECT goal, revision, steps_json FROM agent_pipeline_drafts WHERE draft_id=? AND project_id=?",
            (draft_id, project_id),
        ).fetchone()
    if not row:
        raise ValueError("Pipeline 草稿不存在")
    steps = json.loads(row["steps_json"] or "[]")
    training_steps = [step for step in steps if step.get("tool") in {"training_submit", "workflow_training_submit"}]
    if len(training_steps) != 1:
        raise ValueError("可执行草稿需要且只能包含一个 training_submit 或 workflow_training_submit 步骤")
    if any(step.get("tool") == "workflow_training_submit" for step in training_steps) and len(steps) != 1:
        raise ValueError("workflow_training_submit 已包含质检、训练、评测和部署步骤，不能再与其他阶段重复组合")
    stage_for_tool = {
        "processing_quality_submit": 1, "training_submit": 2,
        "workflow_training_submit": 2, "evaluation_submit": 3, "deployment_start": 4,
    }
    for step in steps:
        tool_name = step.get("tool")
        if tool_name not in stage_for_tool:
            raise ValueError(f"暂不支持执行 Pipeline 工具：{tool_name}")
        for dependency in step.get("depends_on", []):
            parent = next((item for item in steps if item.get("step_id") == dependency), None)
            if parent and stage_for_tool[parent.get("tool")] > stage_for_tool[tool_name]:
                raise ValueError(f"步骤 {step.get('step_id')} 不能依赖后续阶段 {dependency}")

    training_step = training_steps[0]
    args = dict(training_step.get("arguments") or {})
    if isinstance(args.get("configuration"), dict):
        validated_config = args.pop("configuration")
        api_payload = validated_config.get("api_payload")
        if isinstance(api_payload, dict):
            args = {**args, **api_payload}
            args.setdefault("backend", validated_config.get("backend"))
    parameters = args.pop("parameters", None)
    if isinstance(parameters, dict):
        args = {**parameters, **args}
    if "lr" in args and "learning_rate" not in args:
        args["learning_rate"] = args.pop("lr")
    if "max_steps" in args and "steps" not in args:
        args["steps"] = args.pop("max_steps")
    allowed = {
        "dataset_version_id", "backend", "config", "exp_name", "batch_size", "steps",
        "learning_rate", "gpus", "gpu_memory_mb", "lora_rank", "freeze_strategy",
        "num_workers", "epochs", "holdout_episodes", "optimizer", "weight_decay", "seed",
        "chunk_size", "n_action_steps", "use_amp", "pretrained_backbone", "gpu_index", "device",
        "max_quality_episodes", "evaluation_dataset_version_id", "evaluation_samples",
        "deploy_after_evaluation",
    }
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise ValueError("training_submit 步骤存在不支持字段：" + ", ".join(unknown))
    training = {"project_id": project_id, **args}

    quality_steps = [step for step in steps if step.get("tool") == "processing_quality_submit"]
    if len(quality_steps) > 1:
        raise ValueError("每个训练 Pipeline 只支持一个数据质检步骤")
    if quality_steps:
        quality_args = quality_steps[0].get("arguments") or {}
        if (quality_args.get("dataset_version_id") is not None
                and quality_args.get("dataset_version_id") != training.get("dataset_version_id")):
            raise ValueError("质检与训练必须使用同一数据版本")
        max_episodes = quality_args.get("max_episodes")
        if max_episodes is not None:
            training["max_quality_episodes"] = int(max_episodes)

    eval_steps = [step for step in steps if step.get("tool") == "evaluation_submit"]
    if len(eval_steps) > 1:
        raise ValueError("每个训练 Pipeline 只支持一个独立评测步骤")
    if eval_steps:
        evaluation = eval_steps[0].get("arguments") or {}
        evaluation_id = evaluation.get("dataset_version_id")
        if type(evaluation_id) is not int:
            raise ValueError("evaluation_submit 步骤需要 dataset_version_id")
        training["evaluation_dataset_version_id"] = evaluation_id
        if evaluation.get("samples") is not None:
            training["evaluation_samples"] = int(evaluation["samples"])

    deploy_steps = [step for step in steps if step.get("tool") == "deployment_start"]
    if len(deploy_steps) > 1:
        raise ValueError("每个训练 Pipeline 只支持一个部署步骤")
    if deploy_steps:
        if not (training.get("evaluation_dataset_version_id") or training.get("deploy_after_evaluation")):
            raise ValueError("部署步骤需要先配置独立 evaluation_submit 步骤")
        training["deploy_after_evaluation"] = True
    return {"goal": row["goal"], "revision": int(row["revision"]), "training": training}


@mcp.tool(name="workflow_start", description="启动已保存的 Pipeline 草稿：首次调用生成整条训练/评测/部署链的单次确认预览；用户确认后以 approval_id 继续。相同幂等键与确认号复用同一 Workflow/Job。")
async def workflow_start(
    project_id: int,
    workflow_id: str,
    revision: int = 1,
    approval_id: str | None = None,
    idempotency_key: str | None = None,
) -> str:
    try:
        draft_id = workflow_id
        key = (idempotency_key or "").strip()
        if key and (len(key) < 8 or len(key) > 128):
            raise ValueError("idempotency_key 长度需为 8–128 个字符")
        cached: dict[str, Any] | None = None
        if key:
            with _confirmation_connection() as conn:
                saved = conn.execute(
                    "SELECT * FROM agent_pipeline_idempotency WHERE project_id=? AND idempotency_key=?",
                    (project_id, key),
                ).fetchone()
            if saved:
                if saved["draft_id"] != draft_id:
                    raise ValueError("同一 idempotency_key 已绑定其他 Pipeline 草稿")
                if approval_id and saved["confirmation_id"] != approval_id:
                    raise ValueError("approval_id 与该 idempotency_key 的预览不匹配")
                cached = json.loads(saved["response_json"] or "{}")
                if saved["status"] == "submitted":
                    if isinstance(cached.get("data"), dict):
                        cached["data"]["replayed"] = True
                    return json.dumps(cached, ensure_ascii=False, default=str)
                if not approval_id:
                    with _confirmation_connection() as conn:
                        state = conn.execute(
                            "SELECT status FROM agent_confirmations WHERE confirmation_id=? AND project_id=?",
                            (saved["confirmation_id"], project_id),
                        ).fetchone()
                    if state and state["status"] == "pending":
                        return json.dumps(cached, ensure_ascii=False, default=str)
                    raise ValueError("该草稿预览已取消或过期；请使用新的 idempotency_key 重新预览")
                if saved["status"] != "pending":
                    raise ValueError("该 Pipeline 草稿当前不能继续执行")
        compiled = await asyncio.to_thread(_pipeline_draft_training_payload, project_id, draft_id)
        if int(revision) != compiled["revision"]:
            raise ValueError(f"Pipeline 草稿版本已变化：当前 revision={compiled['revision']}")
        result = json.loads(await workflow_training_submit(
            **compiled["training"], confirmation_id=approval_id,
        ))
        data = result.get("data")
        if isinstance(data, dict):
            data["source_draft_id"] = draft_id
            data["draft_id"] = draft_id
            data["workflow_id"] = data.get("workflow_id") or data.get("confirmation_id") or approval_id
            data["revision"] = compiled["revision"]
            data["goal"] = compiled["goal"]
            preview_id = data.get("confirmation_id")
            if preview_id:
                data["approval_id"] = preview_id
            confirmation_ref = preview_id or approval_id
            if confirmation_ref:
                with _confirmation_connection() as conn:
                    row = conn.execute(
                        "SELECT action FROM agent_confirmations WHERE confirmation_id=? AND project_id=?",
                        (confirmation_ref, project_id),
                    ).fetchone()
                    if row:
                        _append_confirmation_event(
                            conn, project_id, confirmation_ref, row["action"],
                            "pipeline_draft_linked" if preview_id else "pipeline_draft_confirmed",
                            {"draft_id": draft_id},
                        )
            if key and confirmation_ref:
                now = time.time()
                state = "submitted" if data.get("status") == "submitted" else "pending"
                with _confirmation_connection() as conn:
                    conn.execute(
                        "INSERT INTO agent_pipeline_idempotency "
                        "(project_id,idempotency_key,draft_id,confirmation_id,status,response_json,created_at,updated_at) "
                        "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(project_id,idempotency_key) DO UPDATE SET "
                        "status=excluded.status,response_json=excluded.response_json,updated_at=excluded.updated_at",
                        (project_id, key, draft_id, confirmation_ref, state,
                         json.dumps(result, ensure_ascii=False, default=str), now, now),
                    )
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"success": False, "project_id": project_id, "data": None,
                           "error": {"code": "DRAFT_EXECUTION_FAILED", "message": str(exc)}},
                          ensure_ascii=False)


@mcp.tool(name="workflow_get", description="按 workflow_id 恢复质检、训练、可选离线评测/部署与 MLflow 状态。返回各步结果和下一步操作；只读。")
async def workflow_get(project_id: int, workflow_id: str) -> str:
    def query() -> dict[str, Any]:
        with _confirmation_connection() as conn:
            workflow = conn.execute(
                "SELECT workflow_id FROM agent_workflows WHERE workflow_id=? AND project_id=?",
                (workflow_id, project_id),
            ).fetchone()
            draft = conn.execute(
                "SELECT * FROM agent_pipeline_drafts WHERE draft_id=? AND project_id=?",
                (workflow_id, project_id),
            ).fetchone()
            confirmation = conn.execute(
                "SELECT * FROM agent_confirmations WHERE confirmation_id=? AND project_id=?",
                (workflow_id, project_id),
            ).fetchone()
        if workflow:
            return _workflow_snapshot(project_id, workflow_id)
        if draft:
            steps = json.loads(draft["steps_json"] or "[]")
            document = {
                "draft_id": draft["draft_id"], "revision": draft["revision"],
                "status": "draft", "goal": draft["goal"], "steps": steps,
                "task_ids": [], "job_created": False,
            }
            serialized = json.dumps(document, ensure_ascii=False, indent=2)
            return {**document, "config_json": serialized, "config_yaml": serialized}
        if not confirmation or confirmation["action"] != "workflow_training_submit":
            raise ValueError("工作流不存在")
        payload = json.loads(confirmation["payload_json"] or "{}")
        return {
            "workflow_id": workflow_id,
            "status": "awaiting_user_confirmation" if confirmation["status"] == "pending" else confirmation["status"],
            "steps": payload.get("steps", []), "preflight": payload.get("preflight", {}),
            "training_config": payload.get("training"),
        }
    return await _call(project_id, query)


@mcp.tool(name="workflow_list", description="列出项目内可恢复的 Agent Workflow、保存的 Pipeline 草稿和待确认预览；用于跨设备查找 workflow_id。只读。")
async def workflow_list(
    project_id: int,
    limit: int = 50,
    cursor: str | None = None,
    status: str | None = None,
) -> str:
    def query() -> dict[str, Any]:
        with _confirmation_connection() as conn:
            workflows = conn.execute(
                "SELECT workflow_id,kind,name,status,steps_json,result_json,created_at,updated_at "
                "FROM agent_workflows WHERE project_id=? ORDER BY updated_at DESC",
                (project_id,),
            ).fetchall()
            drafts = conn.execute(
                "SELECT draft_id,goal,revision,steps_json,created_at,updated_at "
                "FROM agent_pipeline_drafts WHERE project_id=? ORDER BY updated_at DESC",
                (project_id,),
            ).fetchall()
            previews = conn.execute(
                "SELECT confirmation_id,payload_json,status,created_at,updated_at,expires_at "
                "FROM agent_confirmations WHERE project_id=? AND action='workflow_training_submit' "
                "AND status IN ('pending','executing') ORDER BY updated_at DESC",
                (project_id,),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in workflows:
            try:
                steps = json.loads(row["steps_json"] or "[]")
                result = json.loads(row["result_json"] or "{}")
            except json.JSONDecodeError:
                steps, result = [], {}
            job_ids = sorted({
                int(value) for key, value in result.items()
                if key.endswith("_job_id") and isinstance(value, (int, str)) and str(value).isdigit()
            })
            items.append({
                "workflow_id": row["workflow_id"], "kind": row["kind"],
                "name": row["name"], "status": row["status"],
                "steps": [{"id": step.get("id"), "status": step.get("status")}
                          for step in steps if isinstance(step, dict)],
                "job_ids": job_ids, "created_at": row["created_at"], "updated_at": row["updated_at"],
                "source": "workflow",
            })
        for row in drafts:
            try:
                steps = json.loads(row["steps_json"] or "[]")
            except json.JSONDecodeError:
                steps = []
            items.append({
                "workflow_id": row["draft_id"], "kind": "pipeline_draft",
                "name": row["goal"], "status": "draft", "revision": row["revision"],
                "steps": [{"id": step.get("step_id"), "tool": step.get("tool")}
                          for step in steps if isinstance(step, dict)],
                "job_ids": [], "created_at": row["created_at"], "updated_at": row["updated_at"],
                "source": "draft",
            })
        for row in previews:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            training = payload.get("training") or {}
            items.append({
                "workflow_id": row["confirmation_id"], "kind": "workflow_preview",
                "name": training.get("exp_name") or f"{payload.get('backend', 'training')} workflow",
                "status": "awaiting_user_confirmation" if row["status"] == "pending" else row["status"],
                "backend": payload.get("backend"),
                "dataset_version_id": training.get("dataset_version_id"),
                "evaluation_dataset_version_id": (payload.get("evaluation") or {}).get("dataset_version_id"),
                "deploy_after_evaluation": payload.get("deploy_after_evaluation", False),
                "steps": [{"id": step.get("id"), "status": step.get("status")}
                          for step in payload.get("steps", []) if isinstance(step, dict)],
                "job_ids": [], "expires_at": row["expires_at"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
                "source": "confirmation",
            })
        items.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
        if status:
            items = [item for item in items if str(item.get("status", "")).lower() == status.lower()]
        return _page(items, limit, cursor)
    return await _call(project_id, query)


@mcp.tool(name="workflow_advance", description="推进已确认的可恢复工作流：质检后提交训练，再收集 MLflow；若预览含独立评测或部署步骤则继续执行。质检有发现时，须用户明确允许后传 continue_with_findings=true。")
async def workflow_advance(
    project_id: int,
    workflow_id: str,
    continue_with_findings: bool = False,
) -> str:
    def advance() -> dict[str, Any]:
        row = _workflow_read(project_id, workflow_id)
        payload = json.loads(row["payload_json"] or "{}")
        result = json.loads(row["result_json"] or "{}")
        steps = json.loads(row["steps_json"] or "[]")
        saved_steps = {item.get("id"): item for item in steps if isinstance(item, dict)}
        step_defaults = [
            {"id": "preflight", "status": "completed"},
            {"id": "approval", "status": "completed"},
            {"id": "data_quality", "status": "blocked"},
            {"id": "training", "status": "blocked"},
            {"id": "evaluation", "status": "blocked" if payload.get("evaluation") else "skipped"},
            {"id": "deployment", "status": "blocked" if payload.get("deploy_after_evaluation") else "skipped"},
            {"id": "experiment_report", "status": "blocked"},
        ]
        steps = [saved_steps.get(item["id"], item) for item in step_defaults]
        state = row["status"]

        if state in {"quality_submitting", "training_submitting", "evaluation_submitting"}:
            stage = {
                "quality_submitting": ("quality", "quality_job_id", 2, "quality_running"),
                "training_submitting": ("training", "training_job_id", 3, "training_running"),
                "evaluation_submitting": ("evaluation", "evaluation_job_id", 4, "evaluation_running"),
            }[state]
            stage_name, id_key, step_index, running_state = stage
            kinds = {
                "quality": ["quality"],
                "training": ["train", "train_openpi", "train_act"],
                "evaluation": ["eval"],
            }[stage_name]
            found = _workflow_find_job(workflow_id, kinds)
            if found:
                result[id_key] = int(found["job_id"])
                step_id = "data_quality" if stage_name == "quality" else stage_name
                steps[step_index] = {
                    "id": step_id, "status": found["status"], "job_id": int(found["job_id"]),
                }
                _workflow_store(workflow_id, running_state, steps, result)
                return _workflow_snapshot(project_id, workflow_id)
            if not _workflow_claim_stale(workflow_id, state):
                return _workflow_snapshot(project_id, workflow_id)
            try:
                if stage_name == "quality":
                    job = _post("/workbench/api/processing/jobs", {
                        **payload["quality"], "workflow_id": workflow_id,
                    })
                    job_id = int(job["job_id"])
                elif stage_name == "training":
                    job = _workflow_training_request(payload, workflow_id)
                    job_id = int(job.get("job_id") or job.get("id"))
                else:
                    model_dir = result.get("model_dir") or _workflow_model_dir(int(result["training_job_id"]))
                    result["model_dir"] = model_dir
                    job = _workflow_evaluation_request(payload, workflow_id, model_dir)
                    job_id = int(job.get("job_id") or job.get("id"))
            except Exception as exc:
                recovered = _workflow_find_job(workflow_id, kinds)
                if not recovered:
                    _workflow_store(workflow_id, "failed", steps, result, {"message": str(exc)[:1000]})
                    raise
                job = recovered
                job_id = int(recovered["job_id"])
            result[id_key] = job_id
            step_id = "data_quality" if stage_name == "quality" else stage_name
            steps[step_index] = {"id": step_id, "status": job.get("status", "queued"), "job_id": job_id}
            _workflow_store(workflow_id, running_state, steps, result)
            return _workflow_snapshot(project_id, workflow_id)

        if state == "deployment_submitting":
            model_dir = str(result.get("model_dir") or _workflow_model_dir(int(result["training_job_id"])))
            after_id = int(result.get("deployment_id_after") or 0)
            found = _workflow_find_deployment(model_dir, after_id)
            if not found and not _workflow_claim_stale(workflow_id, state, minimum_age=180):
                return _workflow_snapshot(project_id, workflow_id)
            if found:
                deployment = found
            else:
                try:
                    deployment = _workflow_start_deployment(model_dir, after_id)
                except Exception as exc:
                    _workflow_store(workflow_id, "failed", steps, result, {"message": str(exc)[:1000]})
                    raise
            result["model_dir"] = model_dir
            result["deployment_id"] = int(deployment.get("deployment_id") or deployment.get("id"))
            result["deployment"] = deployment
            steps[5] = {"id": "deployment", "status": "completed", "result": deployment}
            steps[6] = {"id": "experiment_report", "status": "completed"}
            _workflow_store(workflow_id, "completed", steps, result)
            return _workflow_snapshot(project_id, workflow_id)

        if state in {"quality_running", "quality_review_required"}:
            quality_job_id = int(result["quality_job_id"])
            quality = _workflow_job_status(quality_job_id, quality=True)
            qstatus = str(quality.get("status", "")).lower()
            if qstatus in {"queued", "running"}:
                return _workflow_snapshot(project_id, workflow_id)
            if qstatus != "completed":
                error = {"message": quality.get("error") or f"质检任务状态：{qstatus}"}
                _workflow_store(workflow_id, "failed", steps, {**result, "quality": quality}, error)
                return _workflow_snapshot(project_id, workflow_id)
            report = quality.get("report") or {}
            findings = report.get("findings") or []
            finding_count = int(report.get("findings_count", len(findings)) or 0)
            steps[2] = {
                "id": "data_quality", "status": "completed", "job_id": quality_job_id,
                "result": {"findings_count": finding_count, "findings": findings[:10]},
            }
            result["quality_result"] = steps[2]["result"]
            if finding_count and not continue_with_findings:
                _workflow_store(workflow_id, "quality_review_required", steps, result)
                return _workflow_snapshot(project_id, workflow_id)
            with _confirmation_connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                updated = conn.execute(
                    "UPDATE agent_workflows SET status='training_submitting', steps_json=?, updated_at=? "
                    "WHERE workflow_id=? AND status IN ('quality_running','quality_review_required')",
                    (json.dumps(steps, ensure_ascii=False), time.time(), workflow_id),
                )
                if updated.rowcount != 1:
                    conn.rollback()
                    return _workflow_snapshot(project_id, workflow_id)
                conn.commit()
            try:
                job = _workflow_training_request(payload, workflow_id)
            except Exception as exc:
                recovered = _workflow_find_job(workflow_id, ["train", "train_openpi", "train_act"])
                if not recovered:
                    _workflow_store(workflow_id, "failed", steps, result, {"message": str(exc)[:1000]})
                    raise
                job = recovered
            training_job_id = int(job.get("job_id") or job.get("id"))
            result["training_job_id"] = training_job_id
            steps[3] = {"id": "training", "status": job.get("status", "queued"), "job_id": training_job_id}
            _workflow_store(workflow_id, "training_running", steps, result)
            return _workflow_snapshot(project_id, workflow_id)

        if state in {"training_running", "reporting"}:
            training_job_id = int(result["training_job_id"])
            job = _workflow_job_status(training_job_id)
            status = str(job.get("status", "")).lower()
            if status in {"queued", "running"}:
                return _workflow_snapshot(project_id, workflow_id)
            if status != "completed":
                steps[3] = {"id": "training", "status": status, "job_id": training_job_id, "error": job.get("error")}
                _workflow_store(workflow_id, "failed", steps, {**result, "training_job": job}, {"message": job.get("error") or status})
                return _workflow_snapshot(project_id, workflow_id)
            if state == "training_running":
                with _confirmation_connection() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    updated = conn.execute(
                        "UPDATE agent_workflows SET status='reporting', updated_at=? "
                        "WHERE workflow_id=? AND status='training_running'",
                        (time.time(), workflow_id),
                    )
                    if updated.rowcount != 1:
                        conn.rollback()
                        return _workflow_snapshot(project_id, workflow_id)
                    conn.commit()
            experiments = _workflow_experiment_report(training_job_id)
            steps[3] = {"id": "training", "status": "completed", "job_id": training_job_id}
            result.update({"training_job": job, "experiments": experiments})
            if payload.get("evaluation"):
                model_dir = result.get("model_dir") or _workflow_model_dir(training_job_id)
                result["model_dir"] = model_dir
                steps[4] = {"id": "evaluation", "status": "submitting"}
                _workflow_store(workflow_id, "evaluation_submitting", steps, result)
                try:
                    evaluation_job = _workflow_evaluation_request(payload, workflow_id, model_dir)
                except Exception as exc:
                    recovered = _workflow_find_job(workflow_id, ["eval"])
                    if not recovered:
                        _workflow_store(workflow_id, "failed", steps, result, {"message": str(exc)[:1000]})
                        raise
                    evaluation_job = recovered
                evaluation_job_id = int(evaluation_job.get("job_id") or evaluation_job.get("id"))
                result["evaluation_job_id"] = evaluation_job_id
                steps[4] = {
                    "id": "evaluation", "status": evaluation_job.get("status", "queued"),
                    "job_id": evaluation_job_id,
                    "dataset_version_id": payload["evaluation"]["dataset_version_id"],
                }
                _workflow_store(workflow_id, "evaluation_running", steps, result)
                return _workflow_snapshot(project_id, workflow_id)

            steps[4] = {"id": "evaluation", "status": "skipped"}
            steps[5] = {"id": "deployment", "status": "skipped"}
            steps[6] = {
                "id": "experiment_report",
                "status": "completed" if experiments["runs"] else "unavailable",
                "result": experiments,
            }
            _workflow_store(workflow_id, "completed", steps, result)
            return _workflow_snapshot(project_id, workflow_id)

        if state == "evaluation_running":
            evaluation_job_id = int(result["evaluation_job_id"])
            evaluation_job = _workflow_job_status(evaluation_job_id)
            evaluation_status = str(evaluation_job.get("status", "")).lower()
            if evaluation_status in {"queued", "running"}:
                return _workflow_snapshot(project_id, workflow_id)
            if evaluation_status != "completed":
                steps[4] = {
                    "id": "evaluation", "status": evaluation_status,
                    "job_id": evaluation_job_id, "error": evaluation_job.get("error"),
                }
                _workflow_store(
                    workflow_id, "failed", steps, {**result, "evaluation_job": evaluation_job},
                    {"message": evaluation_job.get("error") or evaluation_status},
                )
                return _workflow_snapshot(project_id, workflow_id)
            report = _get(f"/workbench/api/evaluations/{evaluation_job_id}/report")
            report_summary = {
                key: value for key, value in report.items()
                if isinstance(value, (str, int, float, bool)) or key in {"metrics", "baseline", "summary"}
            }
            result["evaluation_report"] = report_summary
            steps[4] = {
                "id": "evaluation", "status": "completed", "job_id": evaluation_job_id,
                "dataset_version_id": (payload.get("evaluation") or {}).get("dataset_version_id"),
                "report": report_summary,
            }
            if payload.get("deploy_after_evaluation"):
                model_dir = str(result.get("model_dir") or _workflow_model_dir(int(result["training_job_id"])))
                result["model_dir"] = model_dir
                existing_deployments = _get("/workbench/api/deployments")
                result["deployment_id_after"] = max(
                    (int(item.get("id") or 0) for item in existing_deployments if isinstance(item, dict)),
                    default=0,
                )
                steps[5] = {"id": "deployment", "status": "starting"}
                with _confirmation_connection() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    updated = conn.execute(
                        "UPDATE agent_workflows SET status='deployment_submitting', steps_json=?, "
                        "result_json=?, updated_at=? WHERE workflow_id=? AND status='evaluation_running'",
                        (
                            json.dumps(steps, ensure_ascii=False),
                            json.dumps(result, ensure_ascii=False, default=str),
                            time.time(), workflow_id,
                        ),
                    )
                    if updated.rowcount != 1:
                        conn.rollback()
                        return _workflow_snapshot(project_id, workflow_id)
                    conn.commit()
                try:
                    deployment = _workflow_start_deployment(
                        model_dir, int(result["deployment_id_after"]),
                    )
                except Exception as exc:
                    _workflow_store(workflow_id, "failed", steps, result, {"message": str(exc)[:1000]})
                    raise
                result["deployment_id"] = int(deployment.get("deployment_id") or deployment.get("id"))
                result["deployment"] = deployment
                steps[5] = {"id": "deployment", "status": "completed", "result": deployment}
                steps[6] = {
                    "id": "experiment_report",
                    "status": "completed" if (result.get("experiments") or {}).get("runs") else "unavailable",
                    "result": result.get("experiments", {}),
                }
                _workflow_store(workflow_id, "completed", steps, result)
                return _workflow_snapshot(project_id, workflow_id)

            steps[5] = {"id": "deployment", "status": "skipped"}
            steps[6] = {
                "id": "experiment_report",
                "status": "completed" if (result.get("experiments") or {}).get("runs") else "unavailable",
                "result": result.get("experiments", {}),
            }
            _workflow_store(workflow_id, "completed", steps, result)
            return _workflow_snapshot(project_id, workflow_id)

        return _workflow_snapshot(project_id, workflow_id)

    return await _call(project_id, advance)


@mcp.tool(name="evaluation_submit", description="创建离线评测预览。必须先展示参数并等待用户明确确认；仅在后续调用带回 confirmation_id 时才提交。")
async def evaluation_submit(
    project_id: int,
    model_dir: str,
    dataset_version_id: int,
    samples: int = 8,
    confirmation_id: str | None = None,
) -> str:
    payload = {"model_dir": model_dir, "dataset_version_id": dataset_version_id, "samples": samples}
    return await _call(project_id, lambda: _submit_with_confirmation(
        project_id, "evaluation_submit", payload, confirmation_id,
        lambda: _post("/workbench/api/evaluations", payload),
    ))


@mcp.tool(name="deployment_start", description="创建推理部署预览。必须先展示模型与参数并等待用户明确确认；仅在后续调用带回 confirmation_id 时才启动。")
async def deployment_start(project_id: int, model_dir: str, confirmation_id: str | None = None) -> str:
    payload = {"model_dir": model_dir}
    return await _call(project_id, lambda: _submit_with_confirmation(
        project_id, "deployment_start", payload, confirmation_id,
        lambda: _post("/workbench/api/deployments", payload, timeout=360),
    ))


@mcp.tool(name="training_cancel", description="取消排队中或运行中的训练/评测任务。")
async def training_cancel(project_id: int, job_id: int) -> str:
    return await _call(project_id, lambda: _post(f"/workbench/api/jobs/{job_id}/cancel", {}))


@mcp.tool(name="training_retry", description="重试失败或已取消的训练/评测任务。")
async def training_retry(project_id: int, job_id: int) -> str:
    return await _call(project_id, lambda: _post(f"/workbench/api/jobs/{job_id}/retry", {}))


@mcp.tool(name="training_get_logs", description="读取任务日志，可用 offset/max_bytes 取日志片段。只读。")
async def training_get_logs(
    project_id: int,
    job_id: int | None = None,
    task_id: str | None = None,
    offset: int = 0,
    max_bytes: int = 12000,
) -> str:
    def query() -> dict[str, Any]:
        jid = _job_id(job_id, task_id)
        payload = _get(f"/api/jobs/{jid}/logs", {"tail": 1000})
        lines = payload.get("lines", []) if isinstance(payload, dict) else []
        raw = "\n".join(str(line) for line in lines)
        start = max(0, int(offset))
        size = max(1, min(int(max_bytes), 100000))
        chunk = raw.encode("utf-8")[start:start + size].decode("utf-8", "replace")
        return {**payload, "text": chunk, "offset": start, "next_offset": start + len(chunk.encode("utf-8"))}
    return await _call(project_id, query)


@mcp.tool(name="compute_list_gpus", description="查询 GPU 型号、显存和利用率。只读。")
async def compute_list_gpus(project_id: int) -> str:
    return await _call(project_id, lambda: _get("/api/resources/gpus", {"refresh": "true"}))


@mcp.tool(name="compute_get_queue", description="查询当前任务队列，可按状态筛选并分页。只读。")
async def compute_get_queue(
    project_id: int,
    limit: int = 50,
    cursor: str | None = None,
    status: str | None = None,
) -> str:
    def query() -> dict[str, Any]:
        rows = _get("/api/resources/jobs")
        if status:
            rows = [row for row in rows if str(row.get("status", "")).lower() == status.lower()]
        return _page(rows, limit, cursor)
    return await _call(project_id, query)


@mcp.tool(name="experiment_list_runs", description="查询 MLflow 实验 Runs，可按模型、数据版本或名称筛选。只读。")
async def experiment_list_runs(
    project_id: int,
    limit: int = 50,
    cursor: str | None = None,
    dataset_version_id: int | None = None,
    family: str | None = None,
    q: str | None = None,
) -> str:
    def query() -> dict[str, Any]:
        start = _offset(cursor)
        size = max(1, min(limit, 200))
        payload = _get("/workbench/api/experiments", {"limit": min(1000, start + size + 1)})
        rows = payload.get("runs", []) if isinstance(payload, dict) else payload
        if family:
            rows = [r for r in rows if str(r.get("family", r.get("params", {}).get("model_family", ""))).lower() == family.lower()]
        if dataset_version_id is not None:
            rows = [r for r in rows if str(r.get("params", {}).get("dataset_version_id", "")) == str(dataset_version_id)]
        if q:
            needle = q.lower()
            rows = [r for r in rows if needle in json.dumps(r, ensure_ascii=False).lower()]
        return {"mlflow_available": payload.get("mlflow_available", False), **_page(rows, size, cursor)}
    return await _call(project_id, query)


@mcp.tool(name="experiment_get_metrics", description="读取一个 MLflow Run 的参数、指标与曲线。只读。")
async def experiment_get_metrics(
    project_id: int,
    run_id: str,
    metric_keys: list[str] | None = None,
    max_points: int = 500,
) -> str:
    def query() -> dict[str, Any]:
        payload = _get("/workbench/api/experiments", {"limit": 1000})
        rows = payload.get("runs", []) if isinstance(payload, dict) else payload
        run = next((r for r in rows if str(r.get("run_id")) == run_id), None)
        if run is None:
            raise ValueError(f"Run {run_id} 未找到")
        if metric_keys:
            run = dict(run)
            run["metrics"] = {k: v for k, v in (run.get("metrics") or {}).items() if k in metric_keys}
            run["history"] = {k: v for k, v in (run.get("history") or {}).items() if k in metric_keys}
        max_points = max(1, min(int(max_points), 5000))
        run = dict(run)
        run["history"] = {k: (v[-max_points:] if isinstance(v, list) else v) for k, v in (run.get("history") or {}).items()}
        return run
    return await _call(project_id, query)


@mcp.tool(name="experiment_compare_runs", description="对比 2–20 个 MLflow Run 的参数、指标与曲线，输出带 Run/Job/指标引用的事实、推断、建议及模型/部署草稿。只读，不修改 Run 或创建部署。")
async def experiment_compare_runs(project_id: int, run_ids: list[str]) -> str:
    def compare() -> dict[str, Any]:
        selected_ids = list(dict.fromkeys(str(value).strip() for value in run_ids if str(value).strip()))
        if not 2 <= len(selected_ids) <= 20:
            raise ValueError("run_ids 需要包含 2–20 个不同 Run ID")
        catalog = _get("/workbench/api/experiments", {"limit": 1000})
        all_runs = catalog.get("runs", []) if isinstance(catalog, dict) else catalog
        by_id = {str(row.get("run_id")): row for row in all_runs if row.get("run_id")}
        missing = [run_id for run_id in selected_ids if run_id not in by_id]
        if missing:
            raise ValueError("找不到 Run：" + ", ".join(missing))
        runs: list[dict[str, Any]] = []
        metric_load_warnings: list[str] = []
        for run_id in selected_ids:
            row = dict(by_id[run_id])
            if not isinstance(row.get("metrics"), dict) or not isinstance(row.get("history"), dict):
                job_id = row.get("job_id")
                if job_id is not None:
                    try:
                        detail = _get("/workbench/api/experiments", {"limit": 5, "job_id": int(job_id)})
                        candidate = next((item for item in detail.get("runs", [])
                                          if str(item.get("run_id")) == run_id), None)
                        if candidate:
                            row = {**row, **candidate}
                    except Exception as exc:
                        metric_load_warnings.append(f"{run_id}: {str(exc)[:180]}")
            row["params"] = row.get("params") if isinstance(row.get("params"), dict) else {}
            row["metrics"] = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            row["history"] = row.get("history") if isinstance(row.get("history"), dict) else {}
            runs.append(row)

        def stable(value: Any) -> str:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))

        parameter_keys = sorted({key for row in runs for key in row["params"]})
        parameter_differences: list[dict[str, Any]] = []
        for key in parameter_keys:
            entries = [{"run_id": row["run_id"], "job_id": row.get("job_id"), "value": row["params"].get(key)}
                       for row in runs if key in row["params"]]
            if len({stable(item["value"]) for item in entries}) > 1:
                parameter_differences.append({
                    "parameter": key, "values": entries,
                    "evidence": [{"run_id": item["run_id"], "job_id": item["job_id"],
                                  "source": "platform experiment record / MLflow params"} for item in entries],
                })

        protocol_fields = ("kind", "dataset_name", "dataset_version_id", "holdout_episodes", "seed")
        protocol_differences: list[dict[str, Any]] = []
        for key in protocol_fields:
            values = []
            for row in runs:
                value = row.get(key)
                if value is None:
                    value = row["params"].get(key)
                values.append(value)
            if any(value is None for value in values):
                protocol_differences.append({"field": key, "values": values, "reason": "missing from one or more runs"})
            elif len({stable(value) for value in values}) > 1:
                protocol_differences.append({"field": key, "values": values, "reason": "values differ"})
        protocol_comparable = not protocol_differences

        metric_keys = sorted({key for row in runs for key, value in row["metrics"].items()
                              if type(value) in {int, float}})
        metric_comparisons: list[dict[str, Any]] = []
        facts: list[dict[str, Any]] = []
        for row in runs:
            facts.append({
                "claim": f"Run {row['run_id']} (Job {row.get('job_id')}) 状态为 {row.get('status')}; 模型版本 {row.get('model_version') or '未登记'}。",
                "evidence": [{"run_id": row["run_id"], "job_id": row.get("job_id"),
                              "model_version": row.get("model_version"), "source": "platform experiment record"}],
                "claim_type": "fact",
            })
        for key in metric_keys:
            entries = [{"run_id": row["run_id"], "job_id": row.get("job_id"), "value": float(row["metrics"][key]),
                        "source": "MLflow final summary metric", "unit": None,
                        "sample_count": row["metrics"].get("val_samples") if key.startswith("val_") else None}
                       for row in runs if type(row["metrics"].get(key)) in {int, float}]
            if len(entries) >= 2:
                numbers = [item["value"] for item in entries]
                metric_comparisons.append({
                    "metric": key, "values": entries, "minimum": min(numbers), "maximum": max(numbers),
                    "absolute_spread": max(numbers) - min(numbers),
                    "protocol_comparable": protocol_comparable,
                    "metric_definition": "由 MLflow 指标名和值返回；当前接口未提供单位/聚合口径元数据。",
                })
            for entry in entries:
                facts.append({
                    "claim": f"Run {entry['run_id']} 的 MLflow 指标 {key}={entry['value']}; 单位/聚合定义未由接口提供。",
                    "evidence": [{**entry, "metric_key": key}], "claim_type": "fact",
                })

        curve_analysis: list[dict[str, Any]] = []
        for row in runs:
            for metric, raw_points in row["history"].items():
                if not isinstance(raw_points, list):
                    continue
                points = [point for point in raw_points
                          if isinstance(point, dict) and type(point.get("value")) in {int, float}]
                if not points:
                    continue
                first, last = points[0], points[-1]
                delta = float(last["value"]) - float(first["value"])
                tolerance = max(abs(float(first["value"])), abs(float(last["value"])), 1.0) * 1e-9
                trend = ("decreasing" if delta < -tolerance else
                         "increasing" if delta > tolerance else "flat") if len(points) >= 2 else "insufficient_points"
                curve_analysis.append({
                    "run_id": row["run_id"], "job_id": row.get("job_id"), "metric": metric,
                    "point_count": len(points), "first": first, "last": last,
                    "delta": delta if len(points) >= 2 else None, "trend": trend,
                    "evidence": {"run_id": row["run_id"], "job_id": row.get("job_id"),
                                 "metric_key": metric, "source": "MLflow metric history"},
                })

        facts.extend({
            "claim": f"参数 {item['parameter']} 的记录值在所选 Runs 间不同。",
            "evidence": item["evidence"], "claim_type": "fact",
        } for item in parameter_differences)
        inferences: list[dict[str, Any]] = []
        recommendations: list[dict[str, Any]] = []
        if protocol_differences:
            fields = ", ".join(item["field"] for item in protocol_differences)
            inferences.append({
                "claim": f"这些 Run 的评估/训练协议在 {fields} 上存在差异或缺失，因此当前指标差异不能归因于单一超参数。",
                "confidence": "high for protocol mismatch; no causal claim",
                "limitations": "基于已记录字段；数据接口未提供完整任务定义、随机性和指标单位。",
                "evidence": [{"run_id": row["run_id"], "job_id": row.get("job_id"),
                              "protocol_fields": protocol_differences} for row in runs],
            })
            recommendations.append({
                "action": "固定数据版本、留出 Episode、随机种子和评测口径后再比较超参数。",
                "rationale": "当前选定 Runs 的协议字段不完全一致。",
                "evidence": [{"run_id": row["run_id"], "job_id": row.get("job_id")} for row in runs],
            })
        else:
            recommendations.append({
                "action": "将参数差异与同一留出评测报告一起复核，再决定是否登记或部署候选模型。",
                "rationale": "协议字段一致，但仍需使用同一评测指标定义和样本范围。",
                "evidence": [{"run_id": row["run_id"], "job_id": row.get("job_id")} for row in runs],
            })
        if not metric_comparisons:
            inferences.append({
                "claim": "所选 Runs 没有足够的同名数值指标形成数值对比。",
                "confidence": "high", "limitations": "Run 可能尚未记录 MLflow 指标或后端暂不可用。",
                "evidence": [{"run_id": row["run_id"], "job_id": row.get("job_id"),
                              "metric_keys": sorted(row["metrics"])} for row in runs],
            })

        candidates = [{
            "run_id": row["run_id"], "job_id": row.get("job_id"),
            "model_version": row.get("model_version"), "dataset_version_id": row["params"].get("dataset_version_id"),
            "status": row.get("status"),
        } for row in runs if row.get("model_version")]
        model_draft = {"status": "draft_requires_user_selection" if candidates else "no_model_candidate",
                       "candidates": candidates, "created": False}
        deployment_draft = {"status": "draft_requires_user_selection" if candidates else "no_deployment_candidate",
                            "candidates": candidates, "created": False,
                            "note": "需选择已评测候选后再调用部署预览；本工具不创建部署。"}
        lines = ["# 实验对比报告", "", f"- Runs：{len(runs)}", f"- 协议可比：{'是' if protocol_comparable else '否'}"]
        for item in parameter_differences:
            values = "; ".join(
                f"Job {value['job_id']}=`{stable(value['value'])}`" for value in item["values"]
            )
            lines.append(f"- 参数差异 `{item['parameter']}`：{values}")
        for item in metric_comparisons:
            lines.append(f"- `{item['metric']}` 范围：{item['minimum']}–{item['maximum']}（单位/聚合口径未提供）。")
            lines.append("  证据：" + ", ".join(f"Job {v['job_id']} / Run `{v['run_id']}`" for v in item["values"]))
        for item in inferences:
            lines.append(f"- 推断：{item['claim']}")
        for item in recommendations:
            lines.append(f"- 建议：{item['action']}")
        return {
            "runs": [{key: row.get(key) for key in (
                "run_id", "job_id", "kind", "name", "status", "model_version", "started_at", "params", "metrics"
            )} for row in runs],
            "protocol_comparable": protocol_comparable,
            "protocol_differences": protocol_differences,
            "parameter_differences": parameter_differences,
            "metric_comparisons": metric_comparisons,
            "curve_analysis": curve_analysis,
            "facts": facts, "inferences": inferences, "recommendations": recommendations,
            "model_draft": model_draft, "deployment_draft": deployment_draft,
            "report_markdown": "\n".join(lines),
            "metric_load_warnings": metric_load_warnings,
        }
    return await _call(project_id, compare)


@mcp.tool(name="evaluation_get_report", description="读取已完成评测任务的详细报告。只读。")
async def evaluation_get_report(project_id: int, evaluation_id: int | None = None, job_id: int | None = None) -> str:
    eval_id = evaluation_id if evaluation_id is not None else job_id
    if eval_id is None:
        return json.dumps({"success": False, "project_id": project_id, "data": None, "error": {"code": "VALIDATION_ERROR", "message": "请提供 evaluation_id 或 job_id"}}, ensure_ascii=False)
    return await _call(project_id, lambda: _get(f"/workbench/api/evaluations/{eval_id}/report"))


@mcp.tool(name="model_list_versions", description="查询模型仓库及其版本、权重路径和训练血缘。只读。")
async def model_list_versions(
    project_id: int,
    limit: int = 50,
    cursor: str | None = None,
    family: str | None = None,
) -> str:
    def query() -> dict[str, Any]:
        start = _offset(cursor)
        size = max(1, min(limit, 200))
        payload = _get("/workbench/api/registry", {"limit": min(1000, start + size + 1)})
        models = payload.get("models", []) if isinstance(payload, dict) else payload
        if family:
            models = [m for m in models if str(m.get("family", "")).lower() == family.lower()]
        return {"available_models": payload.get("available_models", []), **_page(models, size, cursor)}
    return await _call(project_id, query)


@mcp.tool(name="benchmark_submit", description="生成 MuJoCo 基准对比预览；确认后创建 CPU Worker Job 并对比任务内置策略。")
async def benchmark_submit(
    project_id: int,
    task: str = "cartpole_balance",
    seeds: list[int] | None = None,
    confirmation_id: str | None = None,
) -> str:
    selected = list(seeds) if seeds is not None else [0, 1, 2, 3, 4]
    payload = {"task": task, "seeds": selected}
    return await _call(project_id, lambda: _submit_with_confirmation(
        project_id, "benchmark_submit", payload, confirmation_id,
        lambda: _post("/workbench/api/benchmarks/submit", payload),
    ))


@mcp.tool(name="benchmark_get_job", description="查询 MuJoCo 基准 Worker Job 状态与对比报告。只读。")
async def benchmark_get_job(project_id: int, job_id: int) -> str:
    return await _call(project_id, lambda: _get(f"/workbench/api/benchmarks/jobs/{job_id}"))


@mcp.tool(name="model_register_candidate", description="把已有 checkpoint 登记为模型候选。先预览名称、版本、权重路径和血缘；用户确认后写入模型仓库。")
async def model_register_candidate(
    project_id: int,
    name: str,
    family: str,
    version: str,
    weights_path: str,
    source: str = "agent_candidate",
    attempt_id: int | None = None,
    mlflow_run_id: str | None = None,
    confirmation_id: str | None = None,
) -> str:
    payload = {
        "name": name.strip(),
        "family": family.strip(),
        "version": version.strip(),
        "weights_path": weights_path.strip(),
        "source": source.strip() or "agent_candidate",
        "attempt_id": attempt_id,
        "mlflow_run_id": mlflow_run_id,
    }

    def submit() -> dict[str, Any]:
        model = _post("/api/models", {"name": payload["name"], "family": payload["family"]})
        model_version = _post(
            f"/api/models/{model['id']}/versions",
            {key: value for key, value in payload.items() if key != "name" and key != "family"},
        )
        return {"model": model, "model_version": model_version}

    return await _call(project_id, lambda: _submit_with_confirmation(
        project_id, "model_register_candidate", payload, confirmation_id, submit,
    ))


@mcp.tool(name="vision_flora_get_result", description="读取真实 FLORA 机器人视频任务、动作片段、三视角来源与模型修订。只读，不启动标注。")
async def vision_flora_get_result(project_id: int, job_id: int) -> str:
    def query() -> dict[str, Any]:
        job = _get(f"/api/jobs/{job_id}")
        if job.get("kind") != "vision_flora":
            raise ValueError("任务不是 FLORA 标注任务")
        result = _get(f"/workbench/api/vision/flora/jobs/{job_id}/result") if job.get("status") == "completed" else None
        return {"job": job, "result": result, "result_ready": result is not None}
    return await _call(project_id, query)


@mcp.tool(name="vision_flora_get_review", description="列出 FLORA 人工审核修订，并读取最新或指定修订的片段与 Mask 引用。只读。")
async def vision_flora_get_review(project_id: int, job_id: int, revision: int | None = None) -> str:
    def query() -> dict[str, Any]:
        versions = _get(f"/workbench/api/vision/flora/jobs/{job_id}/review-versions")
        review = None
        if revision is not None or versions.get("current_revision"):
            review = _get(f"/workbench/api/vision/flora/jobs/{job_id}/review", {"revision": revision} if revision is not None else None)
        return {"job_id": job_id, "versions": versions, "review": review, "reviewed": review is not None}
    return await _call(project_id, query)


@mcp.tool(name="vision_get_capabilities", description="读取视觉模型资产、可用任务、参数、运行方式与导出能力。只读。")
async def vision_get_capabilities(project_id: int) -> str:
    return await _call(project_id, lambda: _get("/workbench/api/vision/capabilities"))


@mcp.tool(name="vision_list_masks", description="按数据版本查询已保存的 SAM2 单图 Mask。只读。")
async def vision_list_masks(project_id: int, dataset_version_id: int | None = None, limit: int = 30) -> str:
    return await _call(project_id, lambda: _get("/workbench/api/vision/runs",
        {"dataset_version_id": dataset_version_id, "limit": max(1, min(limit, 100))}))


@mcp.tool(name="vision_get_mask", description="读取 SAM2 单图结果、原帧来源、提示点、父修订与导出地址。预测分数不是人工真值 IoU。只读。")
async def vision_get_mask(project_id: int, run_id: str) -> str:
    def query() -> dict[str, Any]:
        metadata = _get(f"/workbench/api/vision/runs/{run_id}")
        return {"metadata": metadata, "image_url": f"/workbench/api/vision/runs/{run_id}/image",
                "mask_url": f"/workbench/api/vision/masks/{run_id}",
                "coco_url": f"/workbench/api/vision/runs/{run_id}/coco",
                "score_semantics": "SAM2 predicted IoU; not measured ground-truth IoU"}
    return await _call(project_id, query)


if __name__ == "__main__":
    # STDIO 传输：供目标 Codex/Claude 客户端以子进程方式启动
    # 用法：EMBODIEDFLOW_MCP_TRANSPORT=stdio python -m apps.mcp.server
    #      或 python -m apps.mcp.server --stdio
    import sys as _sys

    if os.environ.get("EMBODIEDFLOW_MCP_TRANSPORT", "").lower() == "stdio"             or "--stdio" in _sys.argv:
        mcp.run(transport="stdio")
        raise SystemExit(0)

    mcp.run(
        transport="streamable-http",
        host=os.environ.get("EMBODIEDFLOW_MCP_HOST", "0.0.0.0"),
        port=int(os.environ.get("EMBODIEDFLOW_MCP_PORT", "18081")),
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
    )
