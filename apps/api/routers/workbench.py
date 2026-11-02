"""当日上线工作台：数据 → 训练 → 评测 → 部署。

页面只提交结构化参数，后端复用现有 Worker 与模型仓库。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import re
import secrets
import sqlite3
import time

import json as _json
import os
import socket
import subprocess
import sys
import time
import uuid
import urllib.request

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from core.config import settings
from persistence.db import engine, session_scope
from persistence.models import Dataset, DatasetVersion, Job, User, UserSession
from services import jobs as jobsvc
from services.processing import CLEAN_OPS, execute_quality_clean

router = APIRouter(prefix="/workbench", tags=["workbench"])

TORCH_PY = "/home/oyjt/workspace/openpi_v2/.venv/bin/python"
TRAINER = str(settings.root / "scripts" / "bc_train.py")
ROOT_HTML = settings.root / "apps" / "web" / "workbench.html"
SERVER = str(settings.root / "scripts" / "bc_server.py")
OPENPI_SERVER = str(settings.root / "scripts" / "openpi_workbench_server.py")


class AgentConfirmationRequest(BaseModel):
    project_id: int = 1
    action: str
    payload: dict = Field(default_factory=dict)


def _confirmation_db():
    path = settings.data_root / "metadata" / "platform.db"
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _confirmation_event(conn, project_id: int, confirmation_id: str, action: str,
                        event_type: str, detail: dict | None = None, created_at: float | None = None):
    conn.execute(
        "INSERT INTO agent_confirmation_events "
        "(project_id, confirmation_id, action, event_type, created_at, detail_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (project_id, confirmation_id, action, event_type, created_at or time.time(),
         json.dumps(detail or {}, ensure_ascii=False, default=str)),
    )


def _confirmation_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


AGENT_CONFIRMATION_ACTIONS = {
    "training_submit:bc", "training_submit:openpi", "training_submit:act",
    "evaluation_submit", "deployment_start", "dataset_materialize",
    "model_register_candidate", "benchmark_submit", "processing_quality_submit",
    "processing_clean_submit",
}


class BenchmarkCompareIn(BaseModel):
    task: str = Field(default="cartpole_balance", pattern="^(cartpole_balance|arm_reach)$")
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4], min_length=1, max_length=32)


class QualityJobIn(BaseModel):
    dataset_version_id: int = Field(ge=1)
    max_episodes: int = Field(default=5, ge=1, le=200)
    workflow_id: str | None = Field(default=None, min_length=1, max_length=128)


def submit_benchmark_compare(payload: BenchmarkCompareIn):
    """Queue a deterministic CPU benchmark comparison through the shared Worker."""
    script = settings.root / "scripts" / "benchmark_worker.py"
    if not script.is_file():
        raise HTTPException(500, f"基准 Runner 不存在：{script}")
    if len(set(payload.seeds)) != len(payload.seeds):
        raise HTTPException(422, "种子不能重复")
    policies = ( ["passive", "pd_balance"] if payload.task == "cartpole_balance"
                 else ["proportional", "random"] )
    with session_scope() as session:
        job = jobsvc.create_job(
            session, kind="benchmark", name=f"benchmark-{payload.task}", spec={},
            requested_gpus=0, max_attempts=1, created_by="agent",
        )
        session.flush()
        report_path = settings.artifacts_dir / "benchmarks" / "jobs" / f"{job.id}.json"
        command = [sys.executable, "-u", str(script), "--task", payload.task,
                   "--seeds-json", json.dumps(payload.seeds, separators=(",", ":")),
                   "--output", str(report_path)]
        job.spec = {
            "command": command,
            "cwd": str(settings.root),
            "metrics_path": str(report_path),
            "report_path": str(report_path),
            "task": payload.task,
            "seeds": payload.seeds,
            "policies": policies,
        }
        return {"job_id": job.id, "status": job.status, "kind": job.kind,
                "task": payload.task, "seeds": payload.seeds, "policies": policies,
                "report_path": str(report_path)}


@router.post("/api/benchmarks/submit", status_code=202)
def create_benchmark_job(payload: BenchmarkCompareIn):
    return submit_benchmark_compare(payload)


@router.get("/api/benchmarks/jobs/{job_id}")
def get_benchmark_job(job_id: int):
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None or job.kind != "benchmark":
            raise HTTPException(404, "基准任务不存在")
        spec = dict(job.spec or {})
        report_path = Path(spec.get("report_path", ""))
        report = None
        if job.status == "completed" and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        return {"job_id": job.id, "status": job.status, "error": job.error,
                "task": spec.get("task"), "seeds": spec.get("seeds"),
                "policies": spec.get("policies"), "report": report,
                "report_path": str(report_path) if report_path else None}


@router.post("/api/agent/confirmations/preview", status_code=201)
def create_agent_confirmation_preview(payload: AgentConfirmationRequest):
    if payload.project_id != 1:
        raise HTTPException(400, "当前服务绑定项目 1")
    if payload.action not in AGENT_CONFIRMATION_ACTIONS:
        raise HTTPException(400, "不支持该确认操作")
    now = time.time()
    confirmation_id = secrets.token_urlsafe(18)
    expires_at = now + 900
    payload_json = json.dumps(payload.payload, ensure_ascii=False, sort_keys=True, default=str)
    conn = _confirmation_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        expired = conn.execute(
            "SELECT project_id, confirmation_id, action FROM agent_confirmations "
            "WHERE status='pending' AND expires_at<=?", (now,),
        ).fetchall()
        conn.execute(
            "UPDATE agent_confirmations SET status='expired', updated_at=? "
            "WHERE status='pending' AND expires_at<=?", (now, now),
        )
        for row in expired:
            _confirmation_event(conn, row["project_id"], row["confirmation_id"], row["action"],
                                "preview_expired", {"reason": "ttl_elapsed"}, now)
        conn.execute(
            "INSERT INTO agent_confirmations "
            "(confirmation_id, project_id, action, payload_digest, payload_json, status, "
            "created_at, expires_at, updated_at) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
            (confirmation_id, payload.project_id, payload.action,
             _confirmation_digest(payload.payload), payload_json, now, expires_at, now),
        )
        _confirmation_event(conn, payload.project_id, confirmation_id, payload.action,
                            "preview_created", {"parameters": payload.payload,
                                                "expires_at": expires_at}, now)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"status": "awaiting_user_confirmation", "confirmation_id": confirmation_id,
            "action": payload.action, "parameters": payload.payload,
            "expires_at": expires_at}


@router.post("/api/agent/confirmations/{confirmation_id}/cancel")
def cancel_agent_confirmation(confirmation_id: str, payload: AgentConfirmationRequest):
    conn = _confirmation_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT project_id, action, status FROM agent_confirmations WHERE confirmation_id=?",
            (confirmation_id,),
        ).fetchone()
        if not row or row["project_id"] != payload.project_id or row["action"] != payload.action:
            conn.rollback()
            raise HTTPException(404, "确认记录不存在")
        if row["status"] == "pending":
            conn.execute(
                "UPDATE agent_confirmations SET status='cancelled', updated_at=? "
                "WHERE confirmation_id=? AND status='pending'", (time.time(), confirmation_id),
            )
            _confirmation_event(conn, row["project_id"], confirmation_id, row["action"],
                                "preview_cancelled")
        conn.commit()
        return {"status": row["status"] if row["status"] != "pending" else "cancelled"}
    finally:
        conn.close()


def _execute_agent_confirmation(action: str, payload: dict):
    if action == "training_submit:bc":
        return create_training(TrainIn(**payload))
    if action == "training_submit:openpi":
        return train_openpi(OpenPIIn(**payload))
    if action == "training_submit:act":
        return train_act(ACTIn(**payload))
    if action == "evaluation_submit":
        return evaluate(EvalIn(**payload))
    if action == "deployment_start":
        return deploy(DeployIn(**payload))
    if action == "dataset_materialize":
        return materialize_dataset(MaterializeIn(**payload))
    if action == "processing_quality_submit":
        return processing_job(QualityJobIn(**payload))
    if action == "processing_clean_submit":
        clean_payload = {**payload, "apply": True}
        return clean_dataset(CleanIn(**clean_payload))
    if action == "model_register_candidate":
        from apps.api.routers.models import ModelIn, VersionIn, create_model, create_version

        model = create_model(ModelIn(name=payload["name"], family=payload["family"]))
        version = create_version(
            model["id"],
            VersionIn(
                version=payload["version"], weights_path=payload["weights_path"],
                source=payload.get("source", "agent_candidate"),
                attempt_id=payload.get("attempt_id"),
                mlflow_run_id=payload.get("mlflow_run_id"),
            ),
        )
        return {"model": model, "model_version": version}
    if action == "benchmark_submit":
        return submit_benchmark_compare(BenchmarkCompareIn(**payload))
    raise HTTPException(400, "不支持该确认操作")


@router.post("/api/agent/confirmations/{confirmation_id}/confirm")
def confirm_agent_submission(confirmation_id: str, payload: AgentConfirmationRequest):
    if payload.project_id != 1 or payload.action not in AGENT_CONFIRMATION_ACTIONS:
        raise HTTPException(400, "项目或确认操作无效")
    now = time.time()
    digest = _confirmation_digest(payload.payload)
    conn = _confirmation_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM agent_confirmations WHERE confirmation_id=?",
                           (confirmation_id,)).fetchone()
        if not row:
            conn.rollback()
            raise HTTPException(404, "确认记录不存在或已使用")
        if (row["project_id"] != payload.project_id or row["action"] != payload.action
                or row["payload_digest"] != digest):
            _confirmation_event(conn, row["project_id"], confirmation_id, row["action"],
                                "confirmation_rejected", {"reason": "binding_mismatch"}, now)
            conn.commit()
            raise HTTPException(409, "提交参数与确认预览不一致，请重新预览")
        if row["status"] == "completed":
            result = json.loads(row["result_json"] or "null")
            _confirmation_event(conn, payload.project_id, confirmation_id, payload.action,
                                "confirmation_replayed", {"status": "completed"}, now)
            conn.commit()
            return result
        if row["status"] == "executing":
            _confirmation_event(conn, payload.project_id, confirmation_id, payload.action,
                                "confirmation_replayed", {"status": "executing"}, now)
            conn.commit()
            raise HTTPException(409, "提交正在处理中，请稍后查看任务")
        if row["status"] != "pending" or row["expires_at"] <= now:
            conn.execute("UPDATE agent_confirmations SET status='expired', updated_at=? "
                         "WHERE confirmation_id=? AND status='pending'", (now, confirmation_id))
            _confirmation_event(conn, payload.project_id, confirmation_id, payload.action,
                                "preview_expired", {"reason": "ttl_elapsed"}, now)
            conn.commit()
            raise HTTPException(410, "确认预览已结束，请重新提交")
        updated = conn.execute(
            "UPDATE agent_confirmations SET status='executing', updated_at=? "
            "WHERE confirmation_id=? AND status='pending'", (now, confirmation_id),
        )
        if updated.rowcount != 1:
            conn.rollback()
            raise HTTPException(409, "提交正在处理中，请稍后查看任务")
        _confirmation_event(conn, payload.project_id, confirmation_id, payload.action,
                            "submission_started", {}, now)
        conn.commit()
    finally:
        conn.close()

    try:
        result = _execute_agent_confirmation(payload.action, payload.payload)
    except Exception as exc:
        conn = _confirmation_db()
        try:
            conn.execute("UPDATE agent_confirmations SET status='failed', error_json=?, updated_at=? "
                         "WHERE confirmation_id=? AND status='executing'",
                         (json.dumps({"message": str(exc)[:1000]}, ensure_ascii=False),
                          time.time(), confirmation_id))
            _confirmation_event(conn, payload.project_id, confirmation_id, payload.action,
                                "submission_failed", {"message": str(exc)[:1000]})
            conn.commit()
        finally:
            conn.close()
        raise

    conn = _confirmation_db()
    try:
        result_json = json.dumps(result, ensure_ascii=False, default=str)
        conn.execute("UPDATE agent_confirmations SET status='completed', result_json=?, "
                     "error_json=NULL, updated_at=? WHERE confirmation_id=? AND status='executing'",
                     (result_json, time.time(), confirmation_id))
        _confirmation_event(conn, payload.project_id, confirmation_id, payload.action,
                            "submission_completed", {"result_available": True})
        conn.commit()
    finally:
        conn.close()
    return result


def _execution_events(project_id: int, cursor: int, limit: int = 100) -> list[dict]:
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT event_id, confirmation_id, action, event_type, created_at, detail_json "
                "FROM agent_confirmation_events "
                "WHERE project_id=:project_id AND event_id>:cursor "
                "ORDER BY event_id LIMIT :limit"
            ),
            {"project_id": project_id, "cursor": cursor, "limit": limit},
        ).mappings().all()
    events = []
    for row in rows:
        try:
            detail = json.loads(row["detail_json"] or "{}")
        except json.JSONDecodeError:
            detail = {}
        events.append({
            "event_id": row["event_id"],
            "confirmation_id": row["confirmation_id"],
            "action": row["action"],
            "event_type": row["event_type"],
            "created_at": row["created_at"],
            "detail": detail,
        })
    return events


@router.get("/api/agent/events/stream")
async def stream_agent_execution_events(
    request: Request,
    project_id: int = 1,
    cursor: int = 0,
):
    if project_id < 1 or cursor < 0:
        raise HTTPException(400, "project_id 和 cursor 必须为非负值")
    header_cursor = request.headers.get("last-event-id")
    if header_cursor:
        try:
            cursor = max(cursor, int(header_cursor))
        except ValueError as exc:
            raise HTTPException(400, "Last-Event-ID 必须为整数") from exc

    async def stream():
        position = cursor
        while not await request.is_disconnected():
            events = await asyncio.to_thread(_execution_events, project_id, position)
            if events:
                for event in events:
                    position = event["event_id"]
                    yield f"id: {position}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            else:
                yield ": keepalive\n\n"
                await asyncio.sleep(1)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class TrainIn(BaseModel):
    dataset_version_id: int
    workflow_id: str | None = Field(default=None, min_length=1, max_length=128)
    epochs: int = Field(default=5, ge=1, le=10000)
    lr: float = Field(default=1e-3, gt=0, le=1)
    batch_size: int = Field(default=256, ge=1, le=65536)
    optimizer: str = Field(default="adamw", pattern="^(adam|adamw)$")
    weight_decay: float = Field(default=1e-4, ge=0, le=1)
    holdout_episodes: int = Field(default=3, ge=1, le=1000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    device: str = Field(default="cuda", pattern="^(cpu|cuda)$")
    gpu_memory_mb: int = Field(default=8192, ge=1024, le=49140)


def _dataset_path(version_id: int) -> tuple[Dataset, DatasetVersion, Path]:
    with session_scope() as s:
        v = s.get(DatasetVersion, version_id)
        if v is None:
            raise HTTPException(404, f"数据版本 {version_id} 不存在")
        d = s.get(Dataset, v.dataset_id)
        mf = v.manifest_uri or ""
        root = Path(mf).parent.parent if mf.endswith("meta/info.json") else Path(mf)
        return d, v, root


def _episode_indices(version: DatasetVersion) -> list[int] | None:
    """Return a derived version's selected episode IDs, or None for all episodes."""
    if not version.episode_indices:
        return None
    try:
        values = json.loads(version.episode_indices)
        selected = sorted({int(value) for value in values if int(value) >= 0})
    except (TypeError, ValueError, json.JSONDecodeError):
        raise HTTPException(500, f"数据版本 {version.id} 的 Episode 清单无效")
    return selected


@router.get("/api/datasets/{version_id}")
def dataset_detail(version_id: int):
    """模块二：数据集详情（基本信息）。"""
    d, v, path = _dataset_path(version_id)
    info = {}
    mf = path / "meta" / "info.json"
    if mf.exists():
        info = json.loads(mf.read_text(encoding="utf-8"))
    feats = info.get("features", {})
    st = (feats.get("observation.state") or {}).get("shape") or [None]
    ac = (feats.get("action") or {}).get("shape") or [None]
    cameras = [
        {"key": key.removeprefix("observation.images."),
         "shape": value.get("shape")}
        for key, value in feats.items()
        if key.startswith("observation.images.") and isinstance(value, dict)
    ]
    return {
        "id": v.id, "dataset": d.name if d else "?",
        "format": d.format if d else None,
        "version": v.version, "status": v.status,
        "episodes": v.episode_count, "bytes": v.total_bytes,
        "sha256": v.sha256, "path": str(path),
        "fps": info.get("fps"), "robot_type": info.get("robot_type"),
        "state_dim": st[0], "action_dim": ac[0],
        "has_manifest": mf.exists(), "cameras": cameras,
    }


@router.get("/api/datasets/{version_id}/episodes")
def dataset_episodes(version_id: int, limit: int = 50):
    """模块二：Episode 浏览器。"""
    from sqlalchemy import select

    from persistence.models import Episode

    from persistence.models import PipelineTask

    with session_scope() as s:
        version = s.get(DatasetVersion, version_id)
        if version is None:
            raise HTTPException(404, f"数据版本 {version_id} 不存在")
        allowed = _episode_indices(version)
        wanted = allowed[:limit] if allowed is not None else None
        source_id = version_id
        visited = set()
        rows = []
        while source_id and source_id not in visited:
            visited.add(source_id)
            stmt = select(Episode).where(Episode.dataset_version_id == source_id)
            if wanted is not None:
                stmt = stmt.where(Episode.episode_index.in_(wanted))
            rows = list(s.scalars(stmt.order_by(Episode.episode_index).limit(limit)))
            if rows:
                break
            parent = s.scalar(select(PipelineTask.input_version_id)
                              .where(PipelineTask.output_version_id == source_id)
                              .order_by(PipelineTask.id.desc()).limit(1))
            source_id = int(parent) if parent else None
        return [{"index": e.episode_index, "frames": e.n_frames,
                 "duration_s": round(e.duration_s, 2) if e.duration_s else None,
                 "task": e.task, "success": e.success} for e in rows]


@router.get("/api/datasets/{version_id}/manifest")
def dataset_manifest(version_id: int):
    """模块二：Dataset Manifest。"""
    _d, _v, path = _dataset_path(version_id)
    mf = path / "meta" / "info.json"
    if not mf.exists():
        raise HTTPException(404, "该版本没有 meta/info.json")
    return json.loads(mf.read_text(encoding="utf-8"))


def _dataset_video_ref(version_id: int, episode: int, camera: str):
    """兼容 LeRobot v2 按 Episode 与 v3 按视频分片存储的映射。"""
    import glob as _glob
    import re as _re

    if episode < 0 or episode > 1_000_000:
        raise HTTPException(422, "episode 超出范围")
    if not _re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", camera):
        raise HTTPException(422, "camera 参数无效")

    _d, version, path = _dataset_path(version_id)
    allowed = _episode_indices(version)
    if allowed is not None and episode not in set(allowed):
        raise HTTPException(404, f"Episode {episode} 不属于派生数据版本 {version_id}")
    mf = path / "meta" / "info.json"
    if not mf.exists():
        raise HTTPException(404, "该版本没有 LeRobot manifest")
    info = json.loads(mf.read_text(encoding="utf-8"))
    camera_key = "observation.images." + camera
    features = info.get("features", {})
    if camera_key not in features:
        raise HTTPException(404, f"数据版本没有相机 {camera}")

    matches = sorted(_glob.glob(str(
        path / "videos" / "chunk-*" / camera_key / f"episode_{episode:06d}.mp4"
    )))
    if matches:
        return Path(matches[0]), 0.0, None

    # LeRobot v3 的 episodes 元数据将 episode 时间段映射到共享 video shard。
    import pyarrow.parquet as _pq

    prefix = f"videos/{camera_key}/"
    columns = ["episode_index", prefix + "chunk_index", prefix + "file_index",
               prefix + "from_timestamp", prefix + "to_timestamp"]
    for meta_file in sorted((path / "meta" / "episodes").glob("**/*.parquet")):
        try:
            table = _pq.read_table(meta_file, columns=columns).to_pylist()
        except Exception:
            continue
        for row in table:
            if int(row.get("episode_index", -1)) != episode:
                continue
            chunk = int(row[prefix + "chunk_index"])
            file_index = int(row[prefix + "file_index"])
            video = (path / "videos" / camera_key / f"chunk-{chunk:03d}"
                     / f"file-{file_index:03d}.mp4")
            if video.is_file():
                start = float(row.get(prefix + "from_timestamp") or 0.0)
                end = float(row.get(prefix + "to_timestamp") or 0.0)
                return video, start, end
            break
    raise HTTPException(404, f"未找到 Episode {episode} 的 {camera} 视频")


@router.get("/api/datasets/{version_id}/video-info")
def dataset_video_info(version_id: int, episode: int = 0, camera: str = "front"):
    video, start, end = _dataset_video_ref(version_id, episode, camera)
    return {"camera": camera, "episode": episode, "from_timestamp": start,
            "to_timestamp": end, "segmented": end is not None,
            "size_bytes": video.stat().st_size}


@router.get("/api/datasets/{version_id}/video")
def dataset_video(version_id: int, episode: int = 0, camera: str = "front"):
    """在浏览器中播放 LeRobot 已录制的视频，不复制或转码原始数据。"""
    video, _start, _end = _dataset_video_ref(version_id, episode, camera)
    return FileResponse(video, media_type="video/mp4",
                        headers={"Content-Disposition": "inline"})


@router.get("/api/preview/{version_id}")
def preview(version_id: int, episode: int = 0, frames: int = 120):
    """返回某个 Episode 的前 N 帧状态/动作，供页面画曲线。"""
    import glob as _glob
    import pyarrow.parquet as _pq

    _d, version, path = _dataset_path(version_id)
    allowed = _episode_indices(version)
    if allowed is not None:
        if not allowed:
            raise HTTPException(404, f"派生数据版本 {version_id} 没有 Episode")
        if episode not in set(allowed):
            episode = allowed[0]
    files = sorted(_glob.glob(str(path / "data" / "**" / "*.parquet"), recursive=True)) or \
            sorted(_glob.glob(str(path / "**" / "*.parquet"), recursive=True))
    for f in files:
        t = _pq.read_table(f, columns=["episode_index", "observation.state", "action"]).to_pydict()
        idx = [i for i, e in enumerate(t["episode_index"]) if int(e) == episode]
        if not idx:
            continue
        sel = idx[:frames]
        st = [[round(float(x), 4) for x in t["observation.state"][i]] for i in sel]
        ac = [[round(float(x), 4) for x in t["action"][i]] for i in sel]
        return {"episode": episode, "frames": len(sel),
                "dims": len(st[0]) if st else 0, "state": st, "action": ac}
    raise HTTPException(404, f"未找到 Episode {episode}")


@router.get("/api/processing/options")
def processing_options():
    """模块三：可选的数据处理项（规则清单）。"""
    rules = [
        {"key": "timestamp", "label": "时间戳完整性"},
        {"key": "gap", "label": "采样间隔不规则"},
        {"key": "frames", "label": "帧数缺失"},
        {"key": "video", "label": "视频损坏"},
        {"key": "align", "label": "相机-动作对齐"},
        {"key": "joint_limit", "label": "关节越限"},
        {"key": "jump", "label": "动作跳变"},
        {"key": "gripper", "label": "夹爪异常"},
        {"key": "incomplete", "label": "Episode 不完整"},
        {"key": "static", "label": "静止帧"},
        {"key": "break", "label": "轨迹断裂"},
    ]
    return {"rules": rules, "rules_version": "1.0"}


@router.post("/api/processing/jobs", status_code=202)
def processing_job(payload: QualityJobIn):
    """Queue dataset quality inspection as a CPU Worker job."""
    script = settings.root / "scripts" / "processing_quality_worker.py"
    if not script.is_file():
        raise HTTPException(500, f"质检 Runner 不存在：{script}")
    _dataset, version, dataset_path = _dataset_path(payload.dataset_version_id)
    if not dataset_path.exists():
        raise HTTPException(400, f"数据目录不存在：{dataset_path}")
    with session_scope() as session:
        job = jobsvc.create_job(
            session, kind="quality", name=f"quality-ds{version.id}", spec={},
            requested_gpus=0, max_attempts=1, created_by="workbench",
            dataset_version_id=version.id,
        )
        session.flush()
        report_path = settings.artifacts_dir / "processing" / "quality_jobs" / f"{job.id}.json"
        command = [
            sys.executable, "-u", str(script),
            "--dataset-version-id", str(version.id),
            "--dataset-path", str(dataset_path),
            "--max-episodes", str(payload.max_episodes),
            "--output", str(report_path),
        ]
        job.spec = {
            "command": command,
            "cwd": str(settings.root),
            "dataset_version_id": version.id,
            "max_episodes": payload.max_episodes,
            "report_path": str(report_path),
            "metrics_path": str(report_path),
            **({"workflow_id": payload.workflow_id} if payload.workflow_id else {}),
        }
        return {"job_id": job.id, "kind": job.kind, "status": job.status,
                "dataset_version_id": version.id,
                "max_episodes": payload.max_episodes}


@router.get("/api/processing/jobs/{job_id}")
def processing_job_status(job_id: int):
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None or job.kind not in {"quality", "clean"}:
            raise HTTPException(404, "数据处理任务不存在")
        spec = dict(job.spec or {})
        report_path = Path(spec.get("report_path", ""))
        report = None
        if job.status == "completed" and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        return {"job_id": job.id, "kind": job.kind, "status": job.status,
                "error": job.error, "dataset_version_id": job.dataset_version_id,
                "max_episodes": spec.get("max_episodes"), "report": report}


@router.get("/api/experiments")
def experiments(limit: int = 20, job_id: int | None = None):
    """模块五：实验列表。

    数据源为平台自有表（job_attempts.mlflow_run_id），不依赖 MLflow 在线；
    若 MLflow 可达，附加实时指标，不可达则跳过（不报错）。
    """
    import urllib.request
    from urllib.parse import urlencode
    from sqlalchemy import desc, select

    from persistence.models import Job, JobAttempt, ModelVersion

    rows = []
    with session_scope() as s:
        stmt = select(JobAttempt).where(JobAttempt.mlflow_run_id.isnot(None))
        if job_id is not None:
            stmt = stmt.where(JobAttempt.job_id == job_id)
        attempts = list(s.scalars(stmt.order_by(desc(JobAttempt.id)).limit(limit)))
        for a in attempts:
            j = s.get(Job, a.job_id)
            mv = s.scalars(select(ModelVersion)
                           .where(ModelVersion.attempt_id == a.id)).first()
            spec = (j.spec or {}) if j else {}
            params = {k: spec.get(k) for k in
                      ("epochs", "lr", "device", "config", "exp_name") if k in spec}
            params.update(spec.get("mlflow_params") or {})
            rows.append({
                "attempt_id": a.id, "job_id": a.job_id,
                "kind": j.kind if j else None, "name": j.name if j else None,
                "status": a.status, "run_id": a.mlflow_run_id,
                "started_at": str(a.started_at)[:19] if a.started_at else None,
                "model_version": mv.version if mv else None,
                "sha256": (mv.sha256 or "")[:12] if mv else None,
                "params": params,
            })

    # 可选：从 MLflow 拉指标
    mlflow_ok = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:15000/health", timeout=2) as r:
            mlflow_ok = r.status == 200
    except Exception:
        pass

    if mlflow_ok:
        for row in rows[:5]:
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:15000/api/2.0/mlflow/runs/get?run_id={row['run_id']}")
                with urllib.request.urlopen(req, timeout=3) as r:
                    d = _json.loads(r.read())
                m = d.get("run", {}).get("data", {}).get("metrics", [])
                row["metrics"] = {
                    x["key"]: round(float(x["value"]), 6)
                    for x in m if x.get("key") and x.get("value") is not None
                }
                row["history"] = {}
                for key in ("train_loss", "val_loss"):
                    url = "http://127.0.0.1:15000/api/2.0/mlflow/metrics/get-history?" + urlencode(
                        {"run_id": row["run_id"], "metric_key": key})
                    try:
                        with urllib.request.urlopen(url, timeout=3) as r:
                            points = _json.loads(r.read()).get("metrics", [])
                        # MLflow can contain duplicate values at a step when older
                        # runs logged both the scalar and the epoch history.
                        by_step = {int(p["step"]): float(p["value"]) for p in points}
                        row["history"][key] = [
                            {"step": step, "value": round(value, 6)}
                            for step, value in sorted(by_step.items())
                        ]
                    except Exception:
                        row["history"][key] = []
            except Exception:
                row["metrics"] = None
                row["history"] = {}

    return {"mlflow_available": mlflow_ok, "count": len(rows), "runs": rows}


@router.get("/api/gpu")
def gpu_cluster():
    """模块六：GPU 算力管理（实时卡态 + 活跃租约 + 调度记录）。"""
    from sqlalchemy import desc, select

    from persistence.models import GPULease, GPUResource, Job, JobAttempt

    with session_scope() as s:
        cards = list(s.scalars(select(GPUResource).order_by(GPUResource.index)))
        leases = list(s.scalars(select(GPULease).where(GPULease.released_at.is_(None))))
        lease_map = {}
        for l in leases:
            a = s.get(JobAttempt, l.attempt_id)
            j = s.get(Job, a.job_id) if a else None
            lease_map[l.gpu_uuid] = {
                "attempt_id": l.attempt_id,
                "job_id": j.id if j else None,
                "job_name": j.name if j else None,
                "since": str(l.acquired_at)[:19],
            }
        recent = []
        for j in s.scalars(select(Job).where(Job.requested_gpus > 0)
                           .order_by(desc(Job.id)).limit(10)):
            recent.append({"id": j.id, "name": j.name, "status": j.status,
                           "gpus": j.requested_gpus,
                           "created_at": str(j.created_at)[:19]})

    out = []
    for c in cards:
        out.append({
            "index": c.index, "name": c.name,
            "memory_used_mb": c.memory_used_mb, "memory_total_mb": c.memory_total_mb,
            "utilization": c.utilization, "state": c.state,
            "lease": lease_map.get(c.uuid),
        })
    return {
        "cards": out,
        "total": len(out),
        "leased": len(lease_map),
        "free_mb_total": sum(c["memory_total_mb"] - c["memory_used_mb"] for c in out),
        "recent_jobs": recent,
    }


def _local_model_catalog() -> list[dict]:
    """Model sources/environments found on the shared host; no framework imports."""
    ws = Path.home() / "workspace"
    home = Path.home()
    sonic_cache = home / ".cache/huggingface/hub/models--nvidia--GEAR-SONIC"
    try:
        sonic_revision = (sonic_cache / "refs/main").read_text(encoding="utf-8").strip()
    except OSError:
        sonic_revision = ""
    sonic_snapshot = sonic_cache / "snapshots" / sonic_revision
    sonic_partials = sorted((sonic_cache / "blobs").glob("*.incomplete")) if (sonic_cache / "blobs").is_dir() else []
    supermimic_repo = Path("/mnt/20e266f0-6466-4ef3-a875-5959cd1015d7/oyjt/supermimic_vibed")
    specs = [
        {
            "name": "OpenPI π0.5", "family": "Vision-Language-Action",
            "repo": ws / "openpi_v2", "python": ws / "openpi_v2/.venv/bin/python",
            "weights": [ws / "checkpoints/pi05_base"], "adapter": "workbench-train",
        },
        {
            "name": "LeRobot ACT", "family": "Action Chunking Transformer",
            "repo": ws / "envs/lingbotvla/lib/python3.12/site-packages/lerobot",
            "python": ws / "envs/lingbotvla/bin/python",
            "weights": [], "adapter": "workbench-train",
            "upstream_url": "https://github.com/huggingface/lerobot",
        },
        {
            "name": "LingBot-VLA V2 6B", "family": "Vision-Language-Action",
            "repo": ws / "lingbot-vla-v2-6b", "python": ws / "envs/lingbotvla/bin/python",
            "weights": [ws / "lingbot-vla-v2-6b/model-00001-of-00006.safetensors",
                        ws / "lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"],
            "adapter": "repository-and-environment-found",
        },
        {
            "name": "NVIDIA GR00T N1.7 3B", "family": "Vision-Language-Action",
            "repo": ws / "GR00T-N1.7-3B", "python": ws / "Isaac-GR00T/.venv/bin/python",
            "weights": [ws / "hf_home/hub/models--nvidia--GR00T-N1.7-3B",
                        ws / "gr00t_finetune_g1_dex3_real"],
            "adapter": "repository-and-environment-found",
        },
        {
            "name": "OpenWAM", "family": "World-Action-Model",
            "repo": Path.home() / "OpenWAM", "python": ws / "envs/openwam/bin/python",
            "weights": [ws / "openwam_assets/openwam_ckpt/openwam_alpha",
                        ws / "openwam_checkpoints"],
            "adapter": "repository-and-environment-found",
        },
        {
            "name": "DreamZero", "family": "Video-Action-Model",
            "repo": ws / "dreamzero", "python": ws / "envs/dreamzero/bin/python",
            "weights": [ws / "dreamzero_assets/checkpoints/DreamZero-DROID",
                        ws / "dreamzero_benchmark_4000"],
            "adapter": "repository-and-environment-found",
        },
        {
            "name": "NVIDIA GEAR-SONIC", "family": "Humanoid Whole-Body Control",
            "repo": ws / "GR00T-WholeBodyControl", "python": ws / "envs/sonic/bin/python",
            "weights": [sonic_snapshot / "model_encoder.onnx",
                        sonic_snapshot / "model_decoder.onnx",
                        sonic_snapshot / "planner_sonic.onnx"],
            "partial_weights": sonic_partials,
            "adapter": "assets-incomplete", "status_key": "sonic",
            "upstream_url": "https://github.com/NVlabs/GR00T-WholeBodyControl",
            "model_url": "https://huggingface.co/nvidia/GEAR-SONIC",
        },
        {
            "name": "SuperMimic", "family": "Humanoid Model-Based RL",
            "repo": supermimic_repo, "python": ws / "envs/supermimic/bin/python",
            "weights": [supermimic_repo / "logs/lafan1_lc_20261023/last.pt",
                        supermimic_repo / "logs/lafan1_lc_20261023/ckpt_014000.pt"],
            "data": [supermimic_repo / "data/lafan1_g1"],
            "adapter": "repository-and-environment-found", "status_key": "supermimic",
            "upstream_url": "https://supermimic.github.io/",
        },
    ]
    found = []
    for spec in specs:
        repo, python = spec["repo"], spec["python"]
        candidates = spec["weights"]
        entry = {
            "name": spec["name"], "family": spec["family"],
            "repo_path": str(repo), "repo_exists": repo.exists(),
            "python_path": str(python), "environment_exists": python.is_file(),
            "weight_paths": [{"path": str(path), "exists": path.exists()} for path in candidates],
            "training_adapter": spec["adapter"],
        }
        if spec.get("upstream_url"):
            entry["upstream_url"] = spec["upstream_url"]
        if spec.get("model_url"):
            entry["model_url"] = spec["model_url"]
        if spec.get("data"):
            entry["data_paths"] = [{"path": str(path), "exists": path.exists()} for path in spec["data"]]
        partials = spec.get("partial_weights", [])
        if partials:
            entry["partial_weight_paths"] = [
                {"path": str(path), "size_bytes": path.stat().st_size}
                for path in partials if path.is_file()
            ]
        if spec.get("status_key") == "sonic":
            found_weights = sum(path.exists() for path in candidates)
            missing_weights = [path.name for path in candidates if not path.exists()]
            notes = []
            if not repo.exists():
                notes.append("本机没有 SONIC 代码仓库")
            if missing_weights:
                notes.append("缺少 " + "、".join(missing_weights))
            if entry.get("partial_weight_paths"):
                partial_mb = sum(item["size_bytes"] for item in entry["partial_weight_paths"]) / (1024 * 1024)
                notes.append(f"检测到未完成下载的权重片段（{partial_mb:.0f} MiB）")
            if not python.is_file():
                notes.append("没有专用运行环境")
            notes.append("工作台训练/评测/部署适配待接入")
            entry["status_note"] = "；".join(notes) if notes else f"已发现 {found_weights} 个模型文件"
        elif spec.get("status_key") == "supermimic":
            weight_count = sum(path.exists() for path in candidates)
            has_data = any(path.exists() for path in spec.get("data", []))
            entry["status_note"] = (
                "本机有 SuperMimic 非官方复现、MuJoCo-Warp 环境和训练 checkpoint"
                + ("、LAFAN1 数据" if has_data else "")
                + "；工作台训练/评测/部署适配待接入"
                if repo.exists() and python.is_file() and weight_count
                else "本机 SuperMimic 代码、环境或 checkpoint 不完整"
            )
        found.append(entry)
    return found


@router.get("/api/registry")
def registry(limit: int = 30):
    """模块八：模型仓库（版本 + 血缘 + 权重路径）。"""
    from sqlalchemy import desc, select

    from persistence.models import (Dataset, DatasetVersion, Job, JobAttempt,
                                    Model, ModelVersion)

    out = []
    with session_scope() as s:
        models = list(s.scalars(select(Model).order_by(Model.name)))
        for m in models:
            vers = list(s.scalars(
                select(ModelVersion).where(ModelVersion.model_id == m.id)
                .order_by(desc(ModelVersion.id)).limit(limit)))
            vlist = []
            for v in vers:
                lineage = None
                if v.attempt_id:
                    a = s.get(JobAttempt, v.attempt_id)
                    j = s.get(Job, a.job_id) if a else None
                    dv = (s.get(DatasetVersion, j.dataset_version_id)
                          if j and j.dataset_version_id else None)
                    ds = s.get(Dataset, dv.dataset_id) if dv else None
                    lineage = {
                        "job_id": j.id if j else None,
                        "job_name": j.name if j else None,
                        "dataset_version": f"{ds.name}/{dv.version}" if ds and dv else None,
                    }
                weights = v.weights_uri or ""
                vlist.append({
                    "version": v.version, "sha256": (v.sha256 or "")[:16],
                    "weights_uri": weights,
                    "weights_exists": os.path.exists(weights) if weights else None,
                    "source": v.source, "mlflow_run_id": (v.mlflow_run_id or "")[:16],
                    "lineage": lineage,
                })
            out.append({"model": m.name, "family": m.family, "versions": vlist})
    return {"count": len(out), "models": out,
            "available_models": _local_model_catalog()}


@router.get("/api/robots")
def robots():
    """模块十：机器人设备管理（设备台账 + 执行契约状态 + 安全约束）。"""
    from sqlalchemy import select

    from persistence.models import Device
    from services import edge

    with session_scope() as s:
        rows = list(s.scalars(select(Device).order_by(Device.id)))
        devices = [{
            "id": d.id, "name": d.name, "robot_spec": d.robot_spec,
            "source": d.source, "state": d.state, "dof": d.dof,
            "camera_url": (d.config or {}).get("camera_url"),
            "last_seen": str(d.last_seen)[:19] if getattr(d, "last_seen", None) else None,
        } for d in rows]

    st = edge.BRIDGE.status()
    return {
        "devices": devices,
        "count": len(devices),
        "adapter": {
            "online": st["online"], "estopped": st["estopped"],
            "action_count": st["action_count"],
            "recent_rejected": st.get("rejected") or [],
        },
    }


class CameraConfigIn(BaseModel):
    camera_url: str = Field(default="", max_length=512)


@router.post("/api/robots/{device_id}/camera")
def save_robot_camera(device_id: int, payload: CameraConfigIn):
    """Save the browser-viewable HTTP MJPEG URL for a robot camera."""
    from persistence.models import Device

    url = payload.camera_url.strip()
    if url and not url.startswith(("http://", "https://")):
        raise HTTPException(422, "浏览器直连仅支持 HTTP/HTTPS 相机流（MJPEG）")
    with session_scope() as s:
        device = s.get(Device, device_id)
        if device is None:
            raise HTTPException(404, "设备不存在")
        config = dict(device.config or {})
        if url:
            config["camera_url"] = url
        else:
            config.pop("camera_url", None)
        device.config = config
        return {"device_id": device_id, "camera_url": config.get("camera_url")}


@router.get("/api/logs")
def logs(kind: str = "", limit: int = 30):
    """日志索引：按类型统计 + 最近记录。"""
    from sqlalchemy import desc, func, select

    from persistence.models import LogRecord

    with session_scope() as s:
        stats = {k: n for k, n in s.execute(
            select(LogRecord.kind, func.count()).group_by(LogRecord.kind)).all()}
        stmt = select(LogRecord).order_by(desc(LogRecord.id)).limit(limit)
        if kind:
            stmt = (select(LogRecord).where(LogRecord.kind == kind)
                    .order_by(desc(LogRecord.id)).limit(limit))
        rows = list(s.scalars(stmt))
        items = [{"id": r.id, "kind": r.kind, "path": r.source_path,
                  "size": r.size_bytes, "lines": r.line_count,
                  "job_id": r.job_id} for r in rows]
        total = sum(stats.values())
    return {"total": total, "by_kind": stats, "items": items}


def _server_users() -> dict[str, dict]:
    """从 /etc/passwd 读取真实存在的普通用户（uid >= 1000）。"""
    users: dict[str, dict] = {}
    try:
        with open("/etc/passwd", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split(":")
                if len(parts) < 7:
                    continue
                name, _pw, uid, gid, _gecos, home, shell = parts[:7]
                try:
                    uid_i = int(uid)
                except ValueError:
                    continue
                if uid_i < 1000 or uid_i >= 65534:
                    continue
                if shell.endswith(("nologin", "false")):
                    continue
                users[name] = {"name": name, "uid": uid_i, "home": home}
    except OSError:
        pass
    return users


@router.get("/api/users")
def list_users():
    """可供登录的服务器用户名（不返回任何敏感信息）。"""
    return sorted(_server_users().values(), key=lambda u: u["uid"])


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)


LOGIN_COOKIE = "ef_session"
LOGIN_TTL_SECONDS = 30 * 24 * 60 * 60


def _session_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _session_user(request: Request) -> str | None:
    token = request.cookies.get(LOGIN_COOKIE, "")
    if not token:
        return None
    with session_scope() as s:
        rec = s.get(UserSession, _session_hash(token))
        if rec is None or rec.expires_at <= time.time():
            return None
        user = s.get(User, rec.user_id)
        if user:
            now = time.time()
            if now - rec.last_seen_at > 300:
                rec.last_seen_at = now
            return user.name
    return None


@router.post("/api/login")
def login(payload: LoginIn, request: Request, response: Response):
    """输入服务器用户名，签发可续用的内网会话 Cookie。"""
    name = (payload.username or "").strip()
    if not name:
        raise HTTPException(400, "请输入用户名")
    users = _server_users()
    if name not in users:
        raise HTTPException(404, f"用户 {name} 不存在于本服务器")
    token = secrets.token_urlsafe(32)
    now = time.time()
    with session_scope() as s:
        old_token = request.cookies.get(LOGIN_COOKIE, "")
        if old_token:
            old_session = s.get(UserSession, _session_hash(old_token))
            if old_session:
                s.delete(old_session)
        user = s.scalars(select(User).where(User.name == name)).first()
        if user is None:
            user = User(name=name, role="engineer")
            s.add(user)
            s.flush()
        s.add(UserSession(
            token_hash=_session_hash(token), user_id=user.id,
            created_at=now, expires_at=now + LOGIN_TTL_SECONDS, last_seen_at=now,
        ))
    response.set_cookie(
        LOGIN_COOKIE, token, max_age=LOGIN_TTL_SECONDS, httponly=True,
        secure=False, samesite="lax", path="/workbench",
    )
    return {"ok": True, "authenticated": True, "username": name,
            "expires_in": LOGIN_TTL_SECONDS}


@router.get("/api/session")
def current_session(request: Request):
    username = _session_user(request)
    return {"authenticated": bool(username), "username": username}


@router.post("/api/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(LOGIN_COOKIE, "")
    if token:
        with session_scope() as s:
            rec = s.get(UserSession, _session_hash(token))
            if rec:
                s.delete(rec)
    response.delete_cookie(LOGIN_COOKIE, path="/workbench", httponly=True, samesite="lax")
    return {"ok": True}


@router.get("/api/running")
def running_jobs():
    """当前运行中的作业（总览页专用，只返回必要字段）。"""
    from sqlalchemy import desc, select

    from persistence.models import Job

    with session_scope() as s:
        rows = list(s.scalars(
            select(Job).where(Job.status == "running")
            .order_by(desc(Job.id)).limit(20)))
        return [{"id": j.id, "kind": j.kind, "name": j.name,
                 "gpus": j.requested_gpus,
                 "started_at": str(j.started_at)[:19] if j.started_at else None}
                for j in rows]


TS_RULES = ("missing_timestamp", "irregular_gap", "time_reversal")


@router.get("/api/processing/timestamps/{version_id}")
def check_timestamps(version_id: int, max_episodes: int = 20):
    """时间戳检查：空值、间隔不齐、时间倒流，逐 Episode 汇总。"""
    from adapters.datasets.lerobot import LeRobotAdapter
    from services import quality as qsvc

    _d, v, path = _dataset_path(version_id)
    if not path.exists():
        raise HTTPException(400, f"数据目录不存在：{path}")
    report = qsvc.build_report(LeRobotAdapter(path), max_episodes=max_episodes)
    findings = [f for f in (report.get("findings") or [])
                if f.get("rule") in TS_RULES]

    per_ep: dict[int, dict] = {}
    for f in findings:
        ep = f.get("episode")
        if ep is None:
            continue
        row = per_ep.setdefault(int(ep), {"episode": int(ep), "missing_timestamp": 0,
                                          "irregular_gap": 0, "time_reversal": 0})
        row[f["rule"]] += int(f.get("count") or 1)

    bad = sorted(per_ep.values(), key=lambda r: -(r["missing_timestamp"] +
                   r["irregular_gap"] + r["time_reversal"]))
    return {
        "dataset_version_id": v.id,
        "episodes_checked": report.get("episodes"),
        "total_findings": len(findings),
        "bad_episodes": len(per_ep),
        "episodes": bad[:50],
        "metrics": {k: report.get("metrics", {}).get(k) for k in
                    ("timestamp_completeness",)},
    }


class AlignIn(BaseModel):
    dataset_version_id: int
    target_fps: float = 30.0
    max_episodes: int = 20


@router.post("/api/processing/align")
def align_plan(payload: AlignIn):
    """对齐计划：算出每个 Episode 按目标帧率重采样后的帧数。

    注意：本接口只出计划，不重写数据（重写需要生成新 parquet，另行实施）。
    """
    import glob as _glob

    import pyarrow.parquet as _pq

    _d, v, path = _dataset_path(payload.dataset_version_id)
    if not path.exists():
        raise HTTPException(400, f"数据目录不存在：{path}")
    if payload.target_fps <= 0:
        raise HTTPException(400, "target_fps 必须为正数")

    files = sorted(_glob.glob(str(path / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        files = sorted(_glob.glob(str(path / "**" / "*.parquet"), recursive=True))

    src_fps = None
    import json as _json
    mf = path / "meta" / "info.json"
    if mf.exists():
        try:
            src_fps = float(_json.loads(mf.read_text(encoding="utf-8")).get("fps") or 0) or None
        except Exception:
            src_fps = None

    counts: dict[int, int] = {}
    for f in files:
        tbl = _pq.read_table(f, columns=["episode_index"]).to_pydict()
        for e in tbl["episode_index"]:
            counts[int(e)] = counts.get(int(e), 0) + 1

    plan = []
    for ep, n in sorted(counts.items())[: payload.max_episodes]:
        dur = n / src_fps if src_fps else None
        target_n = int(round(dur * payload.target_fps)) if dur else None
        plan.append({"episode": ep, "frames": n, "duration_s": round(dur, 3) if dur else None,
                     "target_frames": target_n})
    return {"dataset_version_id": v.id, "source_fps": src_fps,
            "target_fps": payload.target_fps, "episodes": plan,
            "note": "仅计划，不重写数据"}


class MaterializeIn(BaseModel):
    dataset_version_id: int
    episode_indices: list[int] | None = None
    crop_start_s: float = Field(default=0.0, ge=0, le=3600)
    crop_end_s: float | None = Field(default=None, gt=0, le=3600)
    target_fps: int = Field(default=15, ge=1, le=120)
    project_id: int | None = Field(default=None, ge=1)


@router.post("/api/processing/materialize", status_code=202)
def materialize_dataset(payload: MaterializeIn):
    """在 Worker 中实写新的 LeRobot 版本；成功后再登记到数据集仓库。"""
    from persistence.models import PipelineTask, Project

    dataset, version, source_path = _dataset_path(payload.dataset_version_id)
    if not source_path.exists() or not (source_path / "meta" / "info.json").is_file():
        raise HTTPException(400, f"LeRobot 数据目录不存在：{source_path}")
    try:
        info = json.loads((source_path / "meta" / "info.json").read_text(encoding="utf-8"))
        source_fps = float(info.get("fps") or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"无法读取源数据版本：{exc}")
    if source_fps <= 0:
        raise HTTPException(400, "源数据版本缺少有效 fps")
    if payload.crop_end_s is not None and payload.crop_end_s <= payload.crop_start_s:
        raise HTTPException(422, "裁剪结束时间必须大于开始时间")

    try:
        declared = json.loads(version.episode_indices) if version.episode_indices else list(range(version.episode_count or 0))
        declared = sorted({int(ep) for ep in declared})
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"数据版本 Episode 清单无效：{exc}")
    selected = sorted(set(payload.episode_indices)) if payload.episode_indices is not None else declared
    if not selected:
        raise HTTPException(400, "没有可物化的 Episode")
    missing = sorted(set(selected) - set(declared))
    if missing:
        raise HTTPException(400, f"Episode 不属于当前数据版本：{missing}")

    script = settings.root / "scripts" / "lerobot_materialize.py"
    if not script.is_file():
        raise HTTPException(500, f"物化脚本不存在：{script}")
    with session_scope() as session:
        project = session.get(Project, payload.project_id) if payload.project_id else None
        if payload.project_id and project is None:
            raise HTTPException(404, "项目不存在")
        project_name = project.name if project else None
        job = jobsvc.create_job(
            session,
            kind="materialize",
            name=f"数据物化 · {dataset.name} · v{version.id}",
            requested_gpus=0,
            max_attempts=1,
            created_by="workbench",
            dataset_version_id=version.id,
            spec={},
        )
        output_path = settings.data_root / "datasets" / "derived" / "materialized" / f"job_{job.id}"
        report_path = settings.data_root / "artifacts" / "processing" / f"job_{job.id}.json"
        command = [
            TORCH_PY, "-u", str(script),
            "--source", str(source_path), "--output", str(output_path),
            "--repo-id", f"embodiedflow/materialized_{job.id}",
            "--target-fps", str(payload.target_fps),
            "--episodes-json", json.dumps(selected, separators=(",", ":")),
            "--crop-start-s", str(payload.crop_start_s),
            "--source-version-id", str(version.id),
            "--report", str(report_path),
        ]
        if payload.crop_end_s is not None:
            command.extend(["--crop-end-s", str(payload.crop_end_s)])
        spec = {
            "command": command,
            "cwd": str(settings.root),
            "output_path": str(output_path),
            "report_path": str(report_path),
            "source_dataset_id": dataset.id,
            "source_version_id": version.id,
            "selected_episodes": selected,
            "project_name": project_name,
            "target_fps": payload.target_fps,
            "crop_start_s": payload.crop_start_s,
            "crop_end_s": payload.crop_end_s,
        }
        job.spec = spec
        task = PipelineTask(
            job_id=job.id,
            input_version_id=version.id,
            project_id=payload.project_id,
            steps={"operator": "materialize_lerobot", **{k: v for k, v in spec.items() if k not in {"command", "cwd", "output_path", "report_path"}}},
            status="pending",
        )
        session.add(task)
        session.flush()
        return {"job_id": job.id, "status": job.status, "pipeline_task_id": task.id,
                "selected_episodes": selected, "target_fps": payload.target_fps}


@router.get("/api/processing/materialize/{job_id}")
def materialize_status(job_id: int):
    from persistence.models import PipelineTask

    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None or job.kind != "materialize":
            raise HTTPException(404, "物化任务不存在")
        task = session.scalar(select(PipelineTask).where(PipelineTask.job_id == job_id))
        spec = job.spec or {}
        return {
            "job_id": job.id,
            "status": job.status,
            "error": job.error,
            "pipeline_task_id": task.id if task else None,
            "pipeline_status": task.status if task else None,
            "output_version_id": spec.get("output_dataset_version_id"),
            "report": task.steps if task and task.status == "completed" else None,
        }


class InterpIn(BaseModel):
    dataset_version_id: int
    episode: int = 0
    target_fps: float = 30.0
    apply: bool = False
    output_name: str | None = None


def _resample(y: "list[list[float]]", src_fps: float, dst_fps: float) -> list[list[float]]:
    """沿时间轴做线性插值重采样（numpy.interp 逐维）。"""
    import numpy as np

    arr = np.asarray(y, dtype=np.float64)
    n_src = arr.shape[0]
    if n_src == 0 or src_fps <= 0 or dst_fps <= 0:
        return []
    dur = (n_src - 1) / src_fps
    n_dst = int(round(dur * dst_fps)) + 1
    if n_dst < 2:
        return arr.tolist()
    t_src = np.arange(n_src) / src_fps
    t_dst = np.arange(n_dst) / dst_fps
    out = np.empty((n_dst, arr.shape[1]), dtype=np.float64)
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(t_dst, t_src, arr[:, d])
    return out.tolist()


@router.post("/api/processing/interpolate")
def interpolate_episode(payload: InterpIn):
    """按目标帧率对单条 Episode 的 state/action 做线性插值重采样。

    当前返回插值结果（不落盘）；物化成新的数据版本是下一步。
    """
    import glob as _glob
    import json as _json

    import pyarrow.parquet as _pq

    _d, v, path = _dataset_path(payload.dataset_version_id)
    if not path.exists():
        raise HTTPException(400, f"数据目录不存在：{path}")
    if payload.target_fps <= 0:
        raise HTTPException(400, "target_fps 必须为正数")

    src_fps = None
    mf = path / "meta" / "info.json"
    if mf.exists():
        try:
            src_fps = float(_json.loads(mf.read_text(encoding="utf-8")).get("fps") or 0) or None
        except Exception:
            src_fps = None
    if not src_fps:
        raise HTTPException(400, "数据缺少 fps 元信息，无法重采样")

    files = sorted(_glob.glob(str(path / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        files = sorted(_glob.glob(str(path / "**" / "*.parquet"), recursive=True))

    state, action = [], []
    for f in files:
        tbl = _pq.read_table(f, columns=["episode_index", "observation.state", "action"]).to_pydict()
        idx = [i for i, e in enumerate(tbl["episode_index"]) if int(e) == payload.episode]
        if not idx:
            continue
        for i in idx:
            state.append([float(x) for x in tbl["observation.state"][i]])
            action.append([float(x) for x in tbl["action"][i]])
    if not state:
        raise HTTPException(404, f"未找到 Episode {payload.episode}")

    s2 = _resample(state, src_fps, payload.target_fps)
    a2 = _resample(action, src_fps, payload.target_fps)

    materialized = None
    if payload.apply:
        import pyarrow as _pa
        import pyarrow.parquet as _pq

        name = (payload.output_name or f"interp{int(payload.target_fps)}fps").strip("/")
        out_root = settings.data_root / "derived" / "interpolated" / name
        (out_root / "meta").mkdir(parents=True, exist_ok=True)
        (out_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

        n = len(s2)
        table = _pa.table({
            "episode_index": [0] * n,
            "observation.state": s2,
            "action": a2,
            "timestamp": [i / payload.target_fps for i in range(n)],
            "frame_index": list(range(n)),
        })
        pq_path = out_root / "data" / "chunk-000" / "episode_000000.parquet"
        _pq.write_table(table, pq_path)

        info = {
            "codebase_version": "v2.1",
            "robot_type": v.manifest_uri and "interpolated" or "interpolated",
            "total_episodes": 1,
            "total_frames": n,
            "total_tasks": 1,
            "fps": payload.target_fps,
            "source_version_id": v.id,
            "source_fps": src_fps,
            "note": "仅状态与动作插值；未重编码视频，不可用于视觉策略",
        }
        (out_root / "meta" / "info.json").write_text(
            _json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_root / "meta" / "episodes.jsonl").write_text(
            _json.dumps({"episode_index": 0, "length": n}) + chr(10), encoding="utf-8")

        from persistence.models import DatasetVersion

        with session_scope() as s:
            dv = DatasetVersion(
                dataset_id=v.dataset_id,
                version=f"{v.version}-interp{int(payload.target_fps)}",
                status="derived", episode_count=1,
                total_bytes=pq_path.stat().st_size,
                manifest_uri=str(out_root / "meta" / "info.json"),
                episode_indices=_json.dumps([0]),
            )
            s.add(dv)
            s.flush()
            materialized = {"version_id": dv.id, "version": dv.version,
                            "path": str(out_root), "frames": n}

    return {
        "dataset_version_id": v.id,
        "episode": payload.episode,
        "source_fps": src_fps,
        "target_fps": payload.target_fps,
        "source_frames": len(state),
        "target_frames": len(s2),
        "state_dim": len(state[0]),
        "state_head": [[round(x, 4) for x in row] for row in s2[:2]],
        "state_tail": [[round(x, 4) for x in row] for row in s2[-1:]],
        "action_head": [[round(x, 4) for x in row] for row in a2[:2]],
        "materialized": materialized,
        "note": "apply=true 时落成派生版本（仅状态/动作，未重编码视频）",
    }


AGENT_ENV = Path("/home/oyjt/workspace/embodiedflow_data/agent/agent.env")
_AGENT_KEYS = ("DEEPSEEK_BASE_URL", "DSH_MODEL", "DEEPSEEK_API_KEY")


def _read_agent_env() -> dict[str, str]:
    cfg: dict[str, str] = {}
    if not AGENT_ENV.exists():
        return cfg
    for line in AGENT_ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()
    return cfg


def _mask(secret: str | None) -> str:
    if not secret:
        return "未设置"
    tail = secret[-4:] if len(secret) >= 4 else ""
    return f"已设置 …{tail}（长度 {len(secret)}）"


@router.get("/api/admin/agent-config")
def get_agent_config():
    """Agent/MCP 的 Provider 配置。密钥只返回掩码，永不回显明文。"""
    cfg = _read_agent_env()
    return {
        "path": str(AGENT_ENV),
        "exists": AGENT_ENV.exists(),
        "base_url": cfg.get("DEEPSEEK_BASE_URL") or None,
        "model": cfg.get("DSH_MODEL") or None,
        "api_key": _mask(cfg.get("DEEPSEEK_API_KEY")),
        "api_key_set": bool(cfg.get("DEEPSEEK_API_KEY")),
    }


class AgentCfgIn(BaseModel):
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None


@router.post("/api/admin/agent-config")
def set_agent_config(payload: AgentCfgIn):
    """更新 agent.env；空字段保持不变，api_key 传 "clear" 可清除。"""
    cfg = _read_agent_env()
    if payload.base_url is not None:
        cfg["DEEPSEEK_BASE_URL"] = payload.base_url.strip()
    if payload.model is not None:
        cfg["DSH_MODEL"] = payload.model.strip()
    if payload.api_key is not None:
        key = payload.api_key.strip()
        if key.lower() == "clear":
            cfg.pop("DEEPSEEK_API_KEY", None)
        elif key:
            cfg["DEEPSEEK_API_KEY"] = key

    AGENT_ENV.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={cfg[k]}" for k in _AGENT_KEYS if cfg.get(k)]
    _nl = chr(10)
    AGENT_ENV.write_text(_nl.join(lines) + _nl, encoding="utf-8")
    try:
        AGENT_ENV.chmod(0o600)
    except OSError:
        pass
    return {"ok": True, "changed": [k for k in _AGENT_KEYS if cfg.get(k)],
            "api_key": _mask(cfg.get("DEEPSEEK_API_KEY")),
            "restart_hint": "bash ops/restart_agent.sh"}


@router.get("/api/processing/clean-options")
def clean_options():
    return [{"key": k, "label": v} for k, v in CLEAN_OPS.items()]


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=500)


@router.get("/api/projects")
def list_projects():
    from persistence.models import Project

    with session_scope() as s:
        rows = list(s.scalars(select(Project).order_by(Project.name)))
        return [{"id": p.id, "name": p.name, "description": p.description}
                for p in rows]


@router.post("/api/projects", status_code=201)
def create_project(payload: ProjectIn):
    from persistence.models import Project

    name = payload.name.strip()
    if not name:
        raise HTTPException(422, "项目名称不能为空")
    with session_scope() as s:
        existing = s.scalar(select(Project).where(Project.name == name))
        if existing:
            return {"id": existing.id, "name": existing.name,
                    "description": existing.description, "created": False}
        project = Project(name=name, description=payload.description.strip())
        s.add(project)
        s.flush()
        return {"id": project.id, "name": project.name,
                "description": project.description, "created": True}


class CleanIn(BaseModel):
    dataset_version_id: int
    ops: list[str] = ["drop_any_finding"]
    max_episodes: int = Field(default=200, ge=1, le=200)
    apply: bool = False
    project_id: int | None = Field(default=None, ge=1)


@router.post("/api/processing/clean")
def clean_dataset(payload: CleanIn):
    """Preview synchronously; run version-producing cleanup in the CPU Worker."""
    if not payload.apply:
        return _clean_dataset_sync(payload)
    if not payload.ops:
        raise HTTPException(400, "至少选择一种清洗方式")
    unknown = [op for op in payload.ops if op not in CLEAN_OPS]
    if unknown:
        raise HTTPException(400, f"未知清洗方式: {unknown}")
    _dataset, version, dataset_path = _dataset_path(payload.dataset_version_id)
    if not dataset_path.exists():
        raise HTTPException(400, f"数据目录不存在：{dataset_path}")
    from persistence.models import Project
    with session_scope() as session:
        if payload.project_id and session.get(Project, payload.project_id) is None:
            raise HTTPException(404, "项目不存在")
    script = settings.root / "scripts" / "processing_clean_worker.py"
    if not script.is_file():
        raise HTTPException(500, f"清洗 Runner 不存在：{script}")
    payload_data = {
        "dataset_version_id": version.id,
        "ops": payload.ops,
        "max_episodes": payload.max_episodes,
        "apply": True,
        "project_id": payload.project_id,
    }
    with session_scope() as session:
        job = jobsvc.create_job(
            session, kind="clean", name=f"clean-ds{version.id}", spec={},
            requested_gpus=0, max_attempts=1, created_by="workbench",
            dataset_version_id=version.id,
        )
        session.flush()
        report_path = settings.artifacts_dir / "processing" / "clean_jobs" / f"{job.id}.json"
        command = [
            sys.executable, "-u", str(script),
            "--payload-json", json.dumps(payload_data, ensure_ascii=False, separators=(",", ":")),
            "--output", str(report_path),
        ]
        job.spec = {
            "command": command,
            "cwd": str(settings.root),
            "dataset_version_id": version.id,
            "max_episodes": payload.max_episodes,
            "ops": payload.ops,
            "project_id": payload.project_id,
            "report_path": str(report_path),
            "metrics_path": str(report_path),
        }
        return {"job_id": job.id, "kind": job.kind, "status": job.status,
                "dataset_version_id": version.id}


def _clean_dataset_sync(payload: CleanIn, job_id: int | None = None):
    """Run the shared data-processing Service for previews and Worker jobs."""
    try:
        return execute_quality_clean(
            dataset_version_id=payload.dataset_version_id,
            ops=payload.ops,
            max_episodes=payload.max_episodes,
            apply=payload.apply,
            project_id=payload.project_id,
            job_id=job_id,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/api/datasets/{version_id}/lineage")
def dataset_lineage(version_id: int):
    """Show the processing path from this dataset version back to its source."""
    from persistence.models import PipelineTask
    from persistence.models import Project

    chain = []
    visited = set()
    current_id = version_id
    with session_scope() as s:
        while current_id and current_id not in visited:
            visited.add(current_id)
            version = s.get(DatasetVersion, current_id)
            if version is None:
                break
            dataset = s.get(Dataset, version.dataset_id)
            step = s.scalar(select(PipelineTask)
                            .where(PipelineTask.output_version_id == current_id)
                            .order_by(PipelineTask.id.desc()).limit(1))
            project = s.get(Project, step.project_id) if step and step.project_id else None
            chain.append({
                "dataset_version_id": version.id,
                "dataset": dataset.name if dataset else "?",
                "version": version.version,
                "status": version.status,
                "episodes": version.episode_count,
                "operation": (step.steps or {}).get("operator") if step else "source",
                "project_id": step.project_id if step else None,
                "project": project.name if project else None,
                "steps": step.steps if step else None,
            })
            current_id = step.input_version_id if step else None
    if not chain:
        raise HTTPException(404, f"数据版本 {version_id} 不存在")
    return {"dataset_version_id": version_id, "chain": chain}


@router.get("/api/overview")
def overview():
    """模块一：系统总览。数据/训练/GPU/模型四类统计 + 最近活动。"""
    from sqlalchemy import desc, func, select
    from persistence.models import (Dataset, DatasetVersion, Episode, GPUResource,
                                    Job, Model, ModelVersion)

    with session_scope() as s:
        n_ds = s.scalar(select(func.count()).select_from(Dataset)) or 0
        n_ver = s.scalar(select(func.count()).select_from(DatasetVersion)) or 0
        n_ep = s.scalar(select(func.count()).select_from(Episode)) or 0
        total_bytes = s.scalar(select(func.coalesce(func.sum(DatasetVersion.total_bytes), 0))) or 0

        n_job = s.scalar(select(func.count()).select_from(Job)) or 0
        n_running = s.scalar(select(func.count()).select_from(Job)
                             .where(Job.status == "running")) or 0
        n_queued = s.scalar(select(func.count()).select_from(Job)
                            .where(Job.status == "queued")) or 0
        n_failed = s.scalar(select(func.count()).select_from(Job)
                            .where(Job.status == "failed")) or 0
        n_done = s.scalar(select(func.count()).select_from(Job)
                          .where(Job.status == "completed")) or 0

        gpus = list(s.scalars(select(GPUResource).order_by(GPUResource.uuid)))
        gpu_rows = [{"index": g.index, "name": g.name,
                     "memory_used_mb": g.memory_used_mb,
                     "memory_total_mb": g.memory_total_mb,
                     "utilization": g.utilization, "state": g.state} for g in gpus]

        n_model = s.scalar(select(func.count()).select_from(Model)) or 0
        n_mver = s.scalar(select(func.count()).select_from(ModelVersion)) or 0

        recent = []
        for j in s.scalars(select(Job).order_by(desc(Job.id)).limit(8)):
            recent.append({"id": j.id, "kind": j.kind, "name": j.name,
                           "status": j.status, "created_at": str(j.created_at)[:19]})

    return {
        "data": {"datasets": n_ds, "versions": n_ver, "episodes": n_ep,
                 "total_bytes": int(total_bytes)},
        "training": {"jobs": n_job, "running": n_running, "queued": n_queued,
                     "failed": n_failed, "completed": n_done},
        "gpu": {"count": len(gpu_rows),
                "free_mb_total": sum(g["memory_total_mb"] - g["memory_used_mb"] for g in gpu_rows),
                "cards": gpu_rows},
        "models": {"models": n_model, "versions": n_mver},
        "recent": recent,
    }


@router.get("/api/datasets")
def list_datasets(limit: int = 50, q: str = ""):
    """数据版本列表。q 在 SQL 里过滤（不能先 limit 再过滤，否则只能搜到最近 N 条）。"""
    with session_scope() as s:
        stmt = select(DatasetVersion, Dataset).join(
            Dataset, Dataset.id == DatasetVersion.dataset_id)
        if q:
            stmt = stmt.where(Dataset.name.ilike(f"%{q}%"))
        stmt = stmt.order_by(DatasetVersion.id.desc()).limit(limit)
        rows = [(v, d) for v, d in s.execute(stmt).all()]
        out = []
        for v, d in rows:
            out.append({"id": v.id, "label": f"{d.name if d else '?'} / {v.version}",
                        "status": v.status, "episodes": v.episode_count})
        return out


@router.post("/api/training/jobs", status_code=201)
def create_training(payload: TrainIn):
    d, v, path = _dataset_path(payload.dataset_version_id)
    import re as _re

    if d and _re.search(r"(?:^|[_-])(val|valid|validation|eval|test)(?:[_-]|$)",
                        f"{d.name}_{v.version}".lower()):
        raise HTTPException(400, "这是验证/测试数据版本，请选择 train 或合并版本")
    if not path.exists():
        raise HTTPException(400, f"数据目录不存在：{path}")
    data_dir = path / "data"
    if not data_dir.is_dir() or not any(data_dir.glob("**/*.parquet")):
        raise HTTPException(400, "数据版本没有可训练的 Parquet 数据")
    info_path = path / "meta" / "info.json"
    if info_path.is_file():
        features = json.loads(info_path.read_text(encoding="utf-8")).get("features", {})
        if "observation.state" not in features or "action" not in features:
            raise HTTPException(400, "当前 BC 配方需要 observation.state 和 action 字段")
    selected_episodes = _episode_indices(v)
    episode_count = len(selected_episodes) if selected_episodes is not None else v.episode_count
    if selected_episodes is not None and not selected_episodes:
        raise HTTPException(400, "该派生数据版本没有可训练 Episode")
    if episode_count and episode_count <= payload.holdout_episodes:
        raise HTTPException(400, "验证 Episode 数必须小于数据版本的 Episode 总数")

    # 每个提交拥有独立目录，重跑同一数据和超参数也不会覆盖旧权重。
    run_suffix = uuid.uuid4().hex[:8]
    out_dir = settings.data_root / "workbench" / (
        f"run_{payload.dataset_version_id}_{payload.epochs}e_{run_suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_file = None
    if selected_episodes is not None:
        episode_file = out_dir / "episode_indices.json"
        episode_file.write_text(json.dumps(selected_episodes), encoding="utf-8")
    cmd = [TORCH_PY, "-u", TRAINER,
           "--data", str(path), "--out", str(out_dir),
           "--epochs", str(payload.epochs), "--lr", str(payload.lr),
           "--batch", str(payload.batch_size), "--device", payload.device,
           "--optimizer", payload.optimizer,
           "--weight-decay", str(payload.weight_decay),
           "--holdout", str(payload.holdout_episodes),
           "--seed", str(payload.seed)]
    if episode_file:
        cmd.extend(["--episode-indices-file", str(episode_file)])
    mlflow_params = {
        "trainer": "bc_state_action",
        "dataset_name": d.name if d else "unknown",
        "dataset_version": v.version,
        "dataset_version_id": v.id,
        "epochs": payload.epochs,
        "learning_rate": payload.lr,
        "batch_size": payload.batch_size,
        "optimizer": payload.optimizer,
        "weight_decay": payload.weight_decay,
        "holdout_episodes": payload.holdout_episodes,
        "seed": payload.seed,
        "device": payload.device,
        "gpu_memory_mb": payload.gpu_memory_mb if payload.device == "cuda" else 0,
        "selected_episode_count": episode_count,
    }
    spec = {
        "command": cmd,
        "exp_dir": str(out_dir),
        "metrics_path": str(out_dir / "metrics.json"),
        "checkpoint_path": str(out_dir / "checkpoint.pt"),
        "artifact_paths": [str(out_dir / "checkpoint.pt"), str(out_dir / "norm.json")],
        "model_name": f"bc-{d.name}-v{v.version}",
        "model_family": "behavior_cloning",
        "mlflow_params": mlflow_params,
        "mlflow": True,
        "mlflow_experiment": "workbench-bc",
        **({"workflow_id": payload.workflow_id} if payload.workflow_id else {}),
    }
    requested_gpus = 1 if payload.device == "cuda" else 0
    if requested_gpus:
        # 给小型 BC 配方预留 8 GiB/卡；调度器另加安全余量。
        spec["gpu_memory_mb"] = payload.gpu_memory_mb
    with session_scope() as s:
        job = jobsvc.create_job(
            s, kind="train", name=f"bc-{d.name}-{payload.epochs}e",
            spec=spec, requested_gpus=requested_gpus, max_attempts=1,
            created_by="workbench",
            dataset_version_id=payload.dataset_version_id,
        )
        return {"job_id": job.id, "status": job.status, "out_dir": str(out_dir),
                "requested_gpus": requested_gpus,
                "gpu_memory_mb": payload.gpu_memory_mb if requested_gpus else 0,
                "params": mlflow_params}


@router.post("/api/jobs/{job_id}/retry")
def retry_training_job(job_id: int):
    """将失败的工作台训练任务重新排队，增加一次 attempt 并保留原配置。"""
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, f"任务 {job_id} 不存在")
        if job.kind not in {"train", "train_openpi", "train_act"} or job.created_by != "workbench":
            raise HTTPException(409, "只支持重试工作台训练任务")
        if job.status != "failed":
            raise HTTPException(409, "只有失败的训练任务可以重试")
        used = jobsvc.attempts_used(s, job.id)
        job.max_attempts = max(int(job.max_attempts or 1), used + 1)
        job.status = "queued"
        job.started_at = None
        job.finished_at = None
        job.error = None
        return {"job_id": job.id, "status": job.status,
                "attempt_no": used + 1, "out_dir": (job.spec or {}).get("exp_dir")}


class EvalIn(BaseModel):
    model_dir: str
    dataset_version_id: int
    workflow_id: str | None = Field(default=None, min_length=1, max_length=128)
    samples: int = Field(default=8, ge=1, le=32)


class OpenPIIn(BaseModel):
    dataset_version_id: int = Field(default=728, ge=1)
    workflow_id: str | None = Field(default=None, min_length=1, max_length=128)
    config: str = "4items_rtc_v2_lora"
    exp_name: str = "wb-pi05"
    gpus: int = Field(default=1, ge=1, le=8)
    gpu_memory_mb: int = Field(default=22000, ge=1024, le=49140)
    batch_size: int = Field(default=16, ge=1, le=512)
    steps: int = Field(default=5000, ge=1, le=500000)
    learning_rate: float = Field(default=1e-5, gt=0, le=1)
    lora_rank: int = Field(default=0, ge=0, le=64)
    freeze_strategy: str = Field(default="preset", pattern="^(preset|lora_only|vision_lora|full)$")
    num_workers: int = Field(default=8, ge=0, le=64)


class ACTIn(BaseModel):
    dataset_version_id: int
    workflow_id: str | None = Field(default=None, min_length=1, max_length=128)
    exp_name: str = "act"
    batch_size: int = Field(default=4, ge=1, le=128)
    steps: int = Field(default=5000, ge=1, le=500000)
    learning_rate: float = Field(default=1e-5, gt=0, le=1)
    chunk_size: int = Field(default=50, ge=1, le=200)
    n_action_steps: int = Field(default=50, ge=1, le=200)
    use_amp: bool = False
    gpu_memory_mb: int = Field(default=20000, ge=1024, le=49140)
    gpu_index: int | None = Field(default=None, ge=0, le=7)
    num_workers: int = Field(default=4, ge=0, le=32)
    pretrained_backbone: bool = True


OPENPI_ROOT = Path("/home/oyjt/workspace/openpi_v2")
OPENPI_CONFIGS = [
    "4items_rtc_v2_lora",
    "4items_pi05base_rtc_v2",
    "4items_pi05base_rtc_v2_all_lora",
    "uniform_rtc_v2",
    "uniform_table_rtc_v2",
    "debug_pi05",
]


@router.get("/api/training/capabilities")
def training_capability_catalog():
    """Versioned trainer contract shared by the workbench and MCP clients."""
    from services.training_capabilities import training_capabilities

    return training_capabilities(
        root=settings.root,
        torch_python=TORCH_PY,
        bc_trainer=Path(TRAINER),
        bc_eval=settings.root / "scripts" / "bc_eval.py",
        bc_server=Path(SERVER),
        act_python=ACT_PYTHON,
        act_cli=ACT_CLI,
        act_trainer=settings.root / "scripts" / "act_workbench_train.py",
        act_eval=settings.root / "scripts" / "act_workbench_eval.py",
        act_server=Path(ACT_SERVER),
        openpi_root=OPENPI_ROOT,
        openpi_trainer=settings.root / "scripts" / "openpi_workbench_train.py",
        openpi_eval=settings.root / "scripts" / "openpi_workbench_eval.py",
        openpi_server=Path(OPENPI_SERVER),
        openpi_configs=OPENPI_CONFIGS,
    )


@router.get("/api/openpi/configs")
def openpi_configs():
    """工作台可直接运行的 OpenPI 训练预设。"""
    return {"root": str(OPENPI_ROOT),
            "exists": OPENPI_ROOT.exists(),
            "has_train": (OPENPI_ROOT / "scripts" / "train.py").exists(),
            "source": "static",
            "configs": OPENPI_CONFIGS}


@router.post("/api/training/openpi", status_code=201)
def train_openpi(payload: OpenPIIn):
    if payload.config not in OPENPI_CONFIGS:
        raise HTTPException(422, f"不支持的 OpenPI 配置：{payload.config}")
    if payload.batch_size % payload.gpus:
        raise HTTPException(422, "batch size 必须能被 GPU 数量整除")
    dataset, dataset_version, dataset_path = _dataset_path(payload.dataset_version_id)
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file() or not (dataset_path / "data").is_dir():
        raise HTTPException(422, "所选版本不是可训练的 LeRobot 数据集")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    camera_keys = [key for key in info.get("features", {}) if key.startswith("observation.images.")]
    if not camera_keys or not any((dataset_path / "data").glob("**/*.parquet")):
        raise HTTPException(422, "π0.5 训练需要包含相机图像和 Episode 数据的 LeRobot 版本")
    exp_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", payload.exp_name).strip(".-")[:96]
    if not exp_name:
        exp_name = f"workbench-{uuid.uuid4().hex[:8]}"
    trainer = settings.root / "scripts" / "openpi_workbench_train.py"
    if not trainer.exists():
        raise HTTPException(400, f"未找到训练适配器：{trainer}")
    dataset_repo_id = f"embodiedflow/dataset-{dataset_version.id}"
    lerobot_home = settings.data_root / "workbench" / "openpi_lerobot" / exp_name
    dataset_link = lerobot_home / dataset_repo_id
    dataset_link.parent.mkdir(parents=True, exist_ok=True)
    dataset_link.symlink_to(dataset_path.resolve(), target_is_directory=True)
    exp_dir = Path("/home/oyjt/workspace/checkpoints") / payload.config / exp_name
    metrics_path = Path("/home/oyjt/workspace/embodiedflow_data/runs") / f"openpi_{exp_name}_metrics.json"
    cmd = [
        TORCH_PY, "-u", str(trainer), "--config", payload.config,
        "--exp-name", exp_name, "--batch-size", str(payload.batch_size),
        "--dataset-repo-id", dataset_repo_id,
        "--steps", str(payload.steps), "--learning-rate", str(payload.learning_rate),
        "--lora-rank", str(payload.lora_rank), "--freeze-strategy", payload.freeze_strategy,
        "--num-workers", str(payload.num_workers), "--metrics-path", str(metrics_path),
    ]
    spec = {
        "command": cmd,
        "cwd": str(OPENPI_ROOT),
        "exp_dir": str(exp_dir),
        "metrics_path": str(metrics_path),
        "gpu_memory_mb": payload.gpu_memory_mb,
        "mlflow": True,
        "mlflow_experiment": "openpi-workbench",
        "mlflow_params": {
            "config": payload.config, "batch_size": payload.batch_size,
            "dataset_name": dataset.name, "dataset_version": dataset_version.version,
            "dataset_version_id": dataset_version.id, "dataset_repo_id": dataset_repo_id,
            "camera_keys": camera_keys,
            "steps": payload.steps, "learning_rate": payload.learning_rate,
            "siglip_lora_rank": payload.lora_rank,
            "freeze_strategy": payload.freeze_strategy,
            "num_workers": payload.num_workers,
        },
        "model_name": f"OpenPI {payload.config}",
        "model_family": "openpi",
        **({"workflow_id": payload.workflow_id} if payload.workflow_id else {}),
        "env": {
            "OPENPI_DATA_HOME": "/home/oyjt/.cache/openpi",
            "HF_LEROBOT_HOME": str(lerobot_home),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.65",
        },
    }
    with session_scope() as s:
        job = jobsvc.create_job(
            s, kind="train_openpi", name=f"openpi-{payload.config}-{exp_name}",
            spec=spec, requested_gpus=payload.gpus, max_attempts=1,
            dataset_version_id=dataset_version.id,
            created_by="workbench")
        return {"job_id": job.id, "config": payload.config, "exp_name": exp_name,
                "dataset_version_id": dataset_version.id,
                "dataset": dataset.name, "camera_keys": camera_keys,
                "exp_dir": spec["exp_dir"], "requested_gpus": payload.gpus,
                "gpu_memory_mb": payload.gpu_memory_mb, "command": cmd}


ACT_PYTHON = "/home/oyjt/workspace/envs/lingbotvla/bin/python"
ACT_CLI = "/home/oyjt/workspace/envs/lingbotvla/bin/lerobot-train"
ACT_SERVER = str(settings.root / "scripts" / "act_workbench_server.py")


@router.post("/api/training/act", status_code=201)
def train_act(payload: ACTIn):
    d, v, path = _dataset_path(payload.dataset_version_id)
    if payload.n_action_steps > payload.chunk_size:
        raise HTTPException(422, "动作步数不能大于动作块长度")
    if not Path(ACT_PYTHON).is_file() or not Path(ACT_CLI).is_file():
        raise HTTPException(400, "ACT 训练环境不可用")
    info_path = path / "meta" / "info.json"
    if not info_path.is_file() or not any((path / "data").glob("**/*.parquet")):
        raise HTTPException(400, "数据版本缺少 LeRobot 元数据或 Parquet")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    cameras = [key for key in features if key.startswith("observation.images.")]
    missing = [key for key in ("observation.state", "action") if key not in features]
    if missing or not cameras:
        raise HTTPException(400, "ACT 需要图像、observation.state 和 action")
    selected_episodes = _episode_indices(v)
    episodes = selected_episodes
    if episodes is None:
        episodes = list(range(int(info.get("total_episodes") or v.episode_count or 0)))
    if not episodes:
        raise HTTPException(400, "数据版本没有可训练 Episode")

    exp_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", payload.exp_name).strip(".-")[:64]
    if not exp_name:
        exp_name = f"act-{uuid.uuid4().hex[:8]}"
    run_id = uuid.uuid4().hex[:8]
    workbench_root = settings.data_root / "workbench"
    exp_dir = workbench_root / "act" / f"{exp_name}-{run_id}"
    episode_file = workbench_root / "act_specs" / f"{exp_name}-{run_id}.json"
    episode_file.parent.mkdir(parents=True, exist_ok=True)
    episode_file.write_text(json.dumps(episodes), encoding="utf-8")
    metrics_path = workbench_root / "act_metrics" / f"{exp_name}-{run_id}.json"
    checkpoint_path = exp_dir / "checkpoints" / "last" / "pretrained_model"
    trainer = settings.root / "scripts" / "act_workbench_train.py"
    if not trainer.is_file():
        raise HTTPException(400, f"未找到 ACT 训练适配器：{trainer}")
    cmd = [
        sys.executable, "-u", str(trainer),
        "--lerobot-train", ACT_CLI,
        "--dataset-repo-id", f"embodiedflow/{d.name if d else 'dataset'}",
        "--dataset-root", str(path), "--episodes-file", str(episode_file),
        "--output-dir", str(exp_dir), "--metrics-path", str(metrics_path),
        "--steps", str(payload.steps), "--batch-size", str(payload.batch_size),
        "--learning-rate", str(payload.learning_rate),
        "--chunk-size", str(payload.chunk_size),
        "--n-action-steps", str(payload.n_action_steps),
        "--num-workers", str(payload.num_workers),
        "--pretrained-backbone", "true" if payload.pretrained_backbone else "false",
        "--use-amp", "true" if payload.use_amp else "false",
    ]
    mlflow_params = {
        "trainer": "lerobot_act", "dataset_name": d.name if d else "unknown",
        "dataset_version": v.version, "dataset_version_id": v.id,
        "episodes": len(episodes), "camera_keys": cameras,
        "steps": payload.steps, "batch_size": payload.batch_size,
        "learning_rate": payload.learning_rate, "chunk_size": payload.chunk_size,
        "n_action_steps": payload.n_action_steps, "use_amp": payload.use_amp,
        "pretrained_backbone": payload.pretrained_backbone,
        "num_workers": payload.num_workers, "gpu_memory_mb": payload.gpu_memory_mb,
        "gpu_index": payload.gpu_index if payload.gpu_index is not None else "auto",
    }
    spec = {
        "command": cmd, "cwd": str(settings.root), "exp_dir": str(exp_dir),
        "metrics_path": str(metrics_path), "checkpoint_path": str(checkpoint_path),
        "artifact_paths": [str(checkpoint_path)], "gpu_memory_mb": payload.gpu_memory_mb,
        "mlflow": True, "mlflow_experiment": "workbench-act",
        "mlflow_params": mlflow_params,
        "model_name": f"ACT · {d.name if d else 'LeRobot'}",
        "model_family": "act", "source": "real",
        **({"workflow_id": payload.workflow_id} if payload.workflow_id else {}),
    }
    if payload.gpu_index is not None:
        spec["gpu_indices"] = [payload.gpu_index]
    with session_scope() as s:
        job = jobsvc.create_job(
            s, kind="train_act", name=f"act-{d.name if d else 'dataset'}-{exp_name}",
            spec=spec, requested_gpus=1, max_attempts=1,
            created_by="workbench", dataset_version_id=payload.dataset_version_id,
        )
        return {"job_id": job.id, "status": job.status, "exp_dir": str(exp_dir),
                "requested_gpus": 1, "gpu_memory_mb": payload.gpu_memory_mb,
                "gpu_index": payload.gpu_index,
                "params": mlflow_params}


@router.post("/api/evaluations", status_code=201)
def evaluate(payload: EvalIn):
    d, v, path = _dataset_path(payload.dataset_version_id)
    model_dir = Path(payload.model_dir).expanduser()
    act_checkpoint = _resolve_act_checkpoint(model_dir)
    if (model_dir / "checkpoint.pt").exists():
        out = model_dir / f"eval_report_{payload.dataset_version_id}.json"
        cmd = [TORCH_PY, "-u", str(settings.root / "scripts" / "bc_eval.py"),
               "--data", str(path), "--model", str(model_dir), "--out", str(out)]
        selected_episodes = _episode_indices(v)
        if selected_episodes is not None:
            episode_file = model_dir / f"eval_episodes_{payload.dataset_version_id}.json"
            episode_file.write_text(json.dumps(selected_episodes), encoding="utf-8")
            cmd.extend(["--episode-indices-file", str(episode_file)])
        spec = {
            "command": cmd, "exp_dir": str(model_dir), "metrics_path": str(out),
            **({"workflow_id": payload.workflow_id} if payload.workflow_id else {}),
        }
        requested_gpus = 0
    elif act_checkpoint is not None:
        if not Path(ACT_PYTHON).is_file():
            raise HTTPException(400, "ACT 推理环境不可用")
        info_path = path / "meta" / "info.json"
        if not info_path.is_file():
            raise HTTPException(422, "ACT 评测需要 LeRobot 数据版本")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info.get("features", {})
        if "observation.state" not in features or "action" not in features:
            raise HTTPException(422, "ACT 评测数据需包含 observation.state 和 action")
        cameras = [key for key in features if key.startswith("observation.images.")]
        if not cameras:
            raise HTTPException(422, "ACT 评测数据缺少相机图像")
        out = model_dir / f"eval_report_{payload.dataset_version_id}.json"
        cmd = [ACT_PYTHON, "-u", str(settings.root / "scripts" / "act_workbench_eval.py"),
               "--dataset-root", str(path), "--repo-id", f"embodiedflow/{d.name}",
               "--checkpoint", str(act_checkpoint), "--samples", str(payload.samples), "--out", str(out)]
        selected_episodes = _episode_indices(v)
        if selected_episodes is not None:
            episode_file = model_dir / f"eval_episodes_{payload.dataset_version_id}.json"
            episode_file.write_text(json.dumps(selected_episodes), encoding="utf-8")
            cmd.extend(["--episodes-file", str(episode_file)])
        spec = {
            "command": cmd, "cwd": str(settings.root), "exp_dir": str(act_checkpoint),
            "metrics_path": str(out), "model_dir": str(act_checkpoint),
            "gpu_memory_mb": 5000, "mlflow": True,
            "mlflow_experiment": "act-evaluation",
            "mlflow_params": {"model_family": "act", "model_dir": str(model_dir),
                              "dataset_name": d.name, "dataset_version": v.version,
                              "dataset_version_id": v.id, "samples": payload.samples,
                              "camera_keys": cameras},
            **({"workflow_id": payload.workflow_id} if payload.workflow_id else {}),
        }
        requested_gpus = 1
    else:
        checkpoint = _resolve_openpi_checkpoint(model_dir)
        config_name = _openpi_config_for_path(model_dir)
        if checkpoint is None or config_name is None:
            raise HTTPException(400, f"不是可识别的 BC、ACT 或 OpenPI checkpoint：{model_dir}")
        out = model_dir / f"eval_report_{payload.dataset_version_id}.json"
        episode_file = model_dir / f"eval_episodes_{payload.dataset_version_id}.json"
        selected_episodes = _episode_indices(v)
        if selected_episodes is not None:
            episode_file.write_text(json.dumps(selected_episodes), encoding="utf-8")
        cmd = [TORCH_PY, "-u", str(settings.root / "scripts" / "openpi_workbench_eval.py"),
               "--data", str(path), "--checkpoint", str(checkpoint), "--config", config_name,
               "--samples", str(payload.samples), "--out", str(out)]
        if selected_episodes is not None:
            cmd.extend(["--episode-indices-file", str(episode_file)])
        spec = {
            "command": cmd, "cwd": str(OPENPI_ROOT), "metrics_path": str(out),
            "model_dir": str(checkpoint), "gpu_memory_mb": 18000,
            **({"workflow_id": payload.workflow_id} if payload.workflow_id else {}),
            "mlflow": True, "mlflow_experiment": "openpi-evaluation",
            "mlflow_params": {"config": config_name, "model_dir": str(checkpoint),
                              "dataset_name": d.name, "dataset_version": v.version,
                              "dataset_version_id": v.id, "samples": payload.samples},
            "env": {"OPENPI_DATA_HOME": "/home/oyjt/.cache/openpi",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.40",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        }
        requested_gpus = 1
    with session_scope() as s:
        job = jobsvc.create_job(
            s, kind="eval", name=f"eval-{d.name}", spec=spec,
            requested_gpus=requested_gpus, max_attempts=1, created_by="workbench",
            dataset_version_id=payload.dataset_version_id)
        return {"job_id": job.id, "report_path": str(out)}


@router.get("/api/evaluations/{job_id}/report")
def eval_report(job_id: int):
    with session_scope() as s:
        j = s.get(Job, job_id)
        if j is None:
            raise HTTPException(404, "任务不存在")
        if j.status != "completed":
            raise HTTPException(409, f"任务状态 {j.status}，尚无报告")
        path = Path((j.spec or {}).get("metrics_path", ""))
    if not path.exists():
        raise HTTPException(404, f"报告文件不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


class DeployIn(BaseModel):
    model_dir: str


class PredictIn(BaseModel):
    state: list[float] = Field(default_factory=list)
    images: dict[str, str] | None = None
    prompt: str = ""


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _openpi_config_for_path(path: Path) -> str | None:
    return next((p.name for p in (path, *path.parents) if p.name in OPENPI_CONFIGS), None)


def _resolve_openpi_checkpoint(path: Path) -> Path | None:
    if (path / "params").exists() and (path / "assets").exists():
        return path
    candidates = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()
                  and (p / "params").exists() and (p / "assets").exists()] if path.is_dir() else []
    return max(candidates, key=lambda p: int(p.name)) if candidates else None


def _is_act_checkpoint(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    policy_type = str(config.get("type") or config.get("policy_type") or "").lower()
    weights = any((path / name).is_file() for name in (
        "model.safetensors", "pytorch_model.bin", "model.pt", "policy.safetensors"))
    return weights and (policy_type == "act" or "input_features" in config)


def _resolve_act_checkpoint(path: Path) -> Path | None:
    candidates = [
        path,
        path / "checkpoints" / "last" / "pretrained_model",
        path / "checkpoints" / "best" / "pretrained_model",
        path / "last" / "pretrained_model",
    ]
    if path.is_dir():
        candidates.extend(sorted(path.glob("checkpoints/*/pretrained_model")))
    return next((candidate for candidate in candidates if _is_act_checkpoint(candidate)), None)


def _gpu_for_openpi() -> str:
    from services.resources import schedulable_gpus, sync_gpus

    with session_scope() as s:
        sync_gpus(s)
        choices = schedulable_gpus(s, need_mb=18000)
        if not choices:
            raise HTTPException(409, "当前没有可用于 OpenPI 的 GPU（需约 18 GiB 空闲显存）")
        return choices[0]


def _gpu_for_act() -> str:
    from services.resources import schedulable_gpus, sync_gpus
    with session_scope() as s:
        sync_gpus(s)
        choices = schedulable_gpus(s, need_mb=5000)
        if not choices:
            raise HTTPException(409, "当前没有可用于 ACT 推理的 GPU（需约 5 GiB 空闲显存）")
        return choices[0]


def _launch_workbench_model(model_dir: Path, port: int):
    if (model_dir / "checkpoint.pt").exists():
        proc = subprocess.Popen(
            [TORCH_PY, "-u", SERVER, "--model", str(model_dir), "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return proc, f"http://127.0.0.1:{port}", model_dir, "behavior_cloning"

    act_checkpoint = _resolve_act_checkpoint(model_dir)
    if act_checkpoint is not None:
        if not Path(ACT_PYTHON).is_file() or not Path(ACT_SERVER).is_file():
            raise HTTPException(400, "ACT 推理环境或服务脚本不可用")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = _gpu_for_act()
        proc = subprocess.Popen(
            [ACT_PYTHON, "-u", ACT_SERVER, "--model", str(act_checkpoint), "--port", str(port)],
            cwd=str(settings.root), env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        return proc, f"http://127.0.0.1:{port}", act_checkpoint, "act"

    checkpoint = _resolve_openpi_checkpoint(model_dir)
    config_name = _openpi_config_for_path(model_dir)
    if checkpoint is None or config_name is None:
        raise HTTPException(400, f"不是可识别的 BC 或 OpenPI checkpoint：{model_dir}")
    gpu_uuid = _gpu_for_openpi()
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": gpu_uuid,
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.65",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "OPENPI_DATA_HOME": "/home/oyjt/.cache/openpi"})
    proc = subprocess.Popen(
        [TORCH_PY, "-u", OPENPI_SERVER, "--config", config_name,
         "--checkpoint", str(checkpoint), "--port", str(port)],
        cwd=str(OPENPI_ROOT), env=env, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True)
    return proc, f"http://127.0.0.1:{port}", checkpoint, "openpi"


@router.post("/api/deployments", status_code=201)
def deploy(payload: DeployIn):
    from persistence.models import WorkbenchDeployment

    md = Path(payload.model_dir).expanduser()
    port = _free_port()
    proc, url, checkpoint, model_family = _launch_workbench_model(md, port)
    ready = False
    for _ in range(300 if model_family == "openpi" else 60):
        try:
            with urllib.request.urlopen(url + "/health", timeout=1) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.3)
    if not ready:
        proc.terminate()
        raise HTTPException(500, "推理服务启动超时")

    with session_scope() as s:
        rec = WorkbenchDeployment(model_dir=str(checkpoint), port=port, pid=proc.pid,
                                  endpoint=f"{url}/predict", status="running")
        s.add(rec)
        s.flush()
        dep_id = rec.id
    return {"deployment_id": dep_id, "endpoint": f"{url}/predict", "model_family": model_family,
            "predict_url": f"/workbench/api/deployments/{dep_id}/predict",
            "status": "running"}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


@router.post("/api/deployments/{dep_id}/predict")
def predict(dep_id: int, payload: PredictIn):
    from persistence.models import WorkbenchDeployment

    with session_scope() as s:
        rec = s.get(WorkbenchDeployment, dep_id)
        if rec is None:
            raise HTTPException(404, "部署不存在")
        if rec.status != "running":
            raise HTTPException(404, "部署不存在或未运行")
        base = rec.endpoint.rsplit("/predict", 1)[0]
        pid = rec.pid

    if not _alive(pid):
        raise HTTPException(503, "推理进程已退出（可能是服务重启），请重新部署")

    req = urllib.request.Request(
        base + "/predict",
        data=_json.dumps(payload.dict()).encode(),
        headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return _json.loads(r.read())
    except Exception as e:
        raise HTTPException(502, f"推理调用失败：{type(e).__name__}: {e}")


@router.post("/api/deployments/{dep_id}/stop")
def stop(dep_id: int):
    import signal as _sig

    from persistence.models import WorkbenchDeployment, utcnow

    with session_scope() as s:
        rec = s.get(WorkbenchDeployment, dep_id)
        if rec is None:
            raise HTTPException(404, "部署不存在")
        if rec.status == "running":
            try:
                os.killpg(os.getpgid(rec.pid), _sig.SIGTERM)
            except Exception:
                pass
            rec.status = "stopped"
            rec.stopped_at = utcnow()
        return {"deployment_id": dep_id, "status": rec.status}


def restore_deployments(max_restore: int = 3) -> dict:
    """API 启动时恢复上次处于 running 且进程已死的推理服务。

    防护：
    - 每次最多恢复 max_restore 个，避免异常时批量拉起
    - 只恢复目录与 checkpoint 都存在的记录
    - 已存活的不动（推理进程用独立会话，通常能熬过 API 重启）
    """
    from persistence.models import WorkbenchDeployment

    restored, skipped, failed = [], [], []
    with session_scope() as s:
        rows = list(s.query(WorkbenchDeployment)
                    .filter(WorkbenchDeployment.status == "running")
                    .order_by(WorkbenchDeployment.id.desc()).all())
    for rec in rows:
        if len(restored) >= max_restore:
            skipped.append({"id": rec.id, "reason": "超过恢复上限"})
            continue
        if _alive(rec.pid):
            skipped.append({"id": rec.id, "reason": "进程仍存活"})
            continue
        md = Path(rec.model_dir)
        try:
            port = _free_port()
            proc, url, checkpoint, model_family = _launch_workbench_model(md, port)
            ok = False
            for _ in range(300 if model_family == "openpi" else 60):
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as r:
                        if r.status == 200:
                            ok = True
                            break
                except Exception:
                    time.sleep(0.3)
            if not ok:
                proc.terminate()
                failed.append({"id": rec.id, "reason": "启动超时"})
                continue
            with session_scope() as s:
                r = s.get(WorkbenchDeployment, rec.id)
                if r:
                    r.model_dir = str(checkpoint)
                    r.port, r.pid, r.endpoint = port, proc.pid, f"{url}/predict"
                    r.status = "running"
            restored.append({"id": rec.id, "port": port})
        except Exception as e:
            failed.append({"id": rec.id, "reason": f"{type(e).__name__}: {e}"})
    return {"restored": restored, "skipped": skipped, "failed": failed}


@router.get("/api/deployments")
def list_deployments():
    """列出部署；顺带把"进程已死但仍标 running"的记录纠正为 stopped。"""
    from persistence.models import WorkbenchDeployment, utcnow

    out = []
    with session_scope() as s:
        for rec in s.query(WorkbenchDeployment).order_by(WorkbenchDeployment.id.desc()).limit(20):
            if rec.status == "running" and not _alive(rec.pid):
                rec.status = "stopped"
                rec.stopped_at = utcnow()
            out.append({"id": rec.id, "status": rec.status,
                        "model_dir": rec.model_dir, "port": rec.port,
                        "endpoint": rec.endpoint})
    return out


@router.get("/api/deployments/{dep_id}/lineage")
def deployment_lineage(dep_id: int):
    """Resolve a deployed checkpoint back to its Run, data version and evaluations."""
    from sqlalchemy import desc

    from persistence.models import (
        Dataset, DatasetVersion, Job, JobAttempt, Model, ModelVersion,
        PipelineTask, Project, WorkbenchDeployment,
    )

    with session_scope() as s:
        deployment = s.get(WorkbenchDeployment, dep_id)
        if deployment is None:
            raise HTTPException(404, "部署不存在")

        deployed_path = Path(deployment.model_dir)
        act_checkpoint = _resolve_act_checkpoint(deployed_path)
        checkpoint = str(deployed_path if (deployed_path / "params").exists()
                         else act_checkpoint if act_checkpoint is not None
                         else deployed_path / "checkpoint.pt")
        versions = list(s.scalars(select(ModelVersion).order_by(desc(ModelVersion.id))))
        model_version = next((v for v in versions if v.weights_uri and
                              os.path.normpath(v.weights_uri) == os.path.normpath(checkpoint)), None)
        model_info = None
        training_info = None
        data_info = None
        processing_chain = []
        mlflow_run_id = None
        if model_version:
            model = s.get(Model, model_version.model_id)
            model_info = {
                "model": model.name if model else None,
                "family": model.family if model else None,
                "version": model_version.version,
                "sha256": model_version.sha256,
                "weights_uri": model_version.weights_uri,
            }
            mlflow_run_id = model_version.mlflow_run_id
            attempt = s.get(JobAttempt, model_version.attempt_id) if model_version.attempt_id else None
            job = s.get(Job, attempt.job_id) if attempt else None
            if job:
                training_info = {
                    "job_id": job.id, "name": job.name, "status": job.status,
                    "parameters": (job.spec or {}).get("mlflow_params", {}),
                }
                current_id = job.dataset_version_id
                data_seen = set()
                while current_id and current_id not in data_seen:
                    data_seen.add(current_id)
                    version = s.get(DatasetVersion, current_id)
                    if not version:
                        break
                    dataset = s.get(Dataset, version.dataset_id)
                    step = s.scalar(select(PipelineTask)
                                    .where(PipelineTask.output_version_id == version.id)
                                    .order_by(desc(PipelineTask.id)).limit(1))
                    project = s.get(Project, step.project_id) if step and step.project_id else None
                    processing_chain.append({
                        "dataset_version_id": version.id,
                        "dataset": dataset.name if dataset else None,
                        "version": version.version,
                        "episodes": version.episode_count,
                        "operation": (step.steps or {}).get("operator") if step else "source",
                        "project_id": step.project_id if step else None,
                        "project": project.name if project else None,
                    })
                    if len(processing_chain) == 1:
                        data_info = {**processing_chain[-1], "status": version.status}
                    current_id = step.input_version_id if step else None

        evaluations = []
        target_dir = os.path.normpath(deployment.model_dir)
        eval_jobs = list(s.scalars(select(Job).where(Job.kind == "eval")
                                   .order_by(desc(Job.id)).limit(100)))
        for job in eval_jobs:
            spec = job.spec or {}
            eval_model_dir = spec.get("model_dir") or spec.get("exp_dir")
            if os.path.normpath(str(eval_model_dir or "")) != target_dir:
                continue
            eval_data = s.get(DatasetVersion, job.dataset_version_id) if job.dataset_version_id else None
            report = {}
            report_path = spec.get("metrics_path")
            if report_path and Path(report_path).is_file():
                try:
                    raw = json.loads(Path(report_path).read_text(encoding="utf-8"))
                    report = {key: raw.get(key) for key in
                              ("samples", "episodes", "mae", "rmse", "baseline_mean_action_mae",
                               "improvement_vs_baseline", "mean_inference_ms", "holdout_episodes")}
                except (OSError, ValueError):
                    report = {}
            evaluations.append({
                "job_id": job.id, "status": job.status,
                "dataset_version_id": job.dataset_version_id,
                "dataset_version": eval_data.version if eval_data else None,
                "report": report,
            })

        return {
            "deployment": {"id": deployment.id, "status": deployment.status,
                           "model_dir": deployment.model_dir, "endpoint": deployment.endpoint},
            "model": model_info,
            "mlflow_run_id": mlflow_run_id,
            "training": training_info,
            "dataset": data_info,
            "processing_chain": processing_chain,
            "evaluations": evaluations,
        }


@router.get("/api/jobs/{job_id}")
def job_status(job_id: int):
    with session_scope() as s:
        j = s.get(Job, job_id)
        if j is None:
            raise HTTPException(404, "任务不存在")
        tail = ""
        for lg in sorted((settings.data_root / "runs" / f"job_{j.id}").glob("attempt_*/worker.log")):
            tail = "\n".join(jobsvc.tail(lg, 30))
            break
        return {"id": j.id, "status": j.status, "name": j.name,
                "spec": j.spec, "log_tail": tail}


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: int):
    """从工作台取消排队/运行中的任务；运行进程由 Worker 收到状态后结束。"""
    with session_scope() as s:
        if not jobsvc.cancel(s, job_id):
            raise HTTPException(409, "任务不存在或已结束")
    return {"ok": True, "job_id": job_id, "status": "cancelled"}


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def page():
    if ROOT_HTML.exists():
        return HTMLResponse(ROOT_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>workbench</h1><p>页面尚未生成</p>", status_code=200)

@router.get("/api/agent/confirmations")
def list_confirmations_history(project_id: int | None = None,
                               status: str | None = None, limit: int = 20):
    """确认记录（只读）。与 MCP/Agent 共用 agent_confirmations 表。"""
    import sqlite3

    db = settings.data_root / "metadata" / "platform.db"
    if not db.exists():
        raise HTTPException(404, "平台库不存在")
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        where, params = [], []
        if project_id is not None:
            where.append("project_id=?")
            params.append(project_id)
        if status:
            where.append("status=?")
            params.append(status)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = [dict(r) for r in conn.execute(
            f"SELECT confirmation_id, project_id, action, status, created_at, "
            f"updated_at FROM agent_confirmations {clause} "
            f"ORDER BY created_at DESC LIMIT ?",
            (*params, max(1, min(limit, 100)))).fetchall()]
        return {"count": len(rows), "db": db.name, "items": rows}
    finally:
        conn.close()


@router.get("/api/agent/workflows")
def list_workflows_history(project_id: int | None = None, limit: int = 20):
    """Workflow 执行历史（只读）。"""
    import sqlite3

    db = settings.data_root / "metadata" / "platform.db"
    if not db.exists():
        raise HTTPException(404, "平台库不存在")
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        sql = ("SELECT workflow_id, project_id, kind, name, status, updated_at "
               "FROM agent_workflows ")
        params: tuple = ()
        if project_id is not None:
            sql += "WHERE project_id=? "
            params = (project_id,)
        sql += "ORDER BY updated_at DESC LIMIT ?"
        rows = [dict(r) for r in conn.execute(
            sql, (*params, max(1, min(limit, 100)))).fetchall()]
        return {"count": len(rows), "items": rows}
    finally:
        conn.close()

@router.get("/api/agent/workflows/{workflow_id}")
def workflow_detail(workflow_id: str):
    """Workflow 多步结果编排：步骤定义 + 各步对应 Job 的当前状态与结果。"""
    import json as _json
    import sqlite3

    db = settings.data_root / "metadata" / "platform.db"
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM agent_workflows WHERE workflow_id=?", (workflow_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Workflow 不存在")
        wf = dict(row)
        try:
            steps = _json.loads(wf.get("steps_json") or "[]")
        except Exception:
            steps = []
        try:
            result = _json.loads(wf.get("result_json") or "{}")
        except Exception:
            result = {}

        job_ids = []
        for st in steps if isinstance(steps, list) else []:
            jid = (st or {}).get("job_id") if isinstance(st, dict) else None
            if jid:
                job_ids.append(jid)
        if isinstance(result, dict) and result.get("job_id"):
            job_ids.append(result["job_id"])
        job_ids = list(dict.fromkeys(int(j) for j in job_ids if str(j).isdigit()))

        jobs = {}
        if job_ids:
            marks = ",".join("?" * len(job_ids))
            for j in conn.execute(
                f"SELECT id, kind, name, status, created_at, started_at, finished_at "
                f"FROM jobs WHERE id IN ({marks})", tuple(job_ids)
            ).fetchall():
                jobs[j["id"]] = dict(j)

        for st in steps if isinstance(steps, list) else []:
            if isinstance(st, dict) and str(st.get("job_id", "")).isdigit():
                st["job"] = jobs.get(int(st["job_id"]))
        wf.pop("payload_json", None)
        return {"workflow": {k: wf[k] for k in
                             ("workflow_id", "project_id", "kind", "name",
                              "status", "created_at", "updated_at")},
                "steps": steps, "result": result, "jobs": list(jobs.values())}
    finally:
        conn.close()
