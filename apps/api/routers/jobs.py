from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from apps.api.schemas import JobCreate, JobOut, LogOut
from persistence.db import session_scope
from persistence.models import Job, JobAttempt
from services import jobs as jobsvc

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobOut, status_code=202)
def create(payload: JobCreate):
    with session_scope() as s:
        spec = dict(payload.spec or {})
        if payload.gpu_memory_mb:
            spec["gpu_memory_mb"] = int(payload.gpu_memory_mb)
        job = jobsvc.create_job(
            s,
            kind=payload.kind,
            name=payload.name,
            spec=spec,
            requested_gpus=payload.requested_gpus,
            max_attempts=payload.max_attempts,
            priority=payload.priority,
            created_by=payload.created_by,
        )
        s.flush()
        return JobOut.model_validate(job)


@router.get("", response_model=list[JobOut])
def list_jobs(status: str | None = None, kind: str | None = None, limit: int = 50):
    with session_scope() as s:
        stmt = select(Job).order_by(Job.id.desc()).limit(limit)
        if status:
            stmt = stmt.where(Job.status == status)
        if kind:
            stmt = stmt.where(Job.kind == kind)
        return [JobOut.model_validate(j) for j in s.scalars(stmt)]


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: int):
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return JobOut.model_validate(job)


@router.get("/{job_id}/attempts")
def attempts(job_id: int):
    with session_scope() as s:
        rows = list(
            s.scalars(
                select(JobAttempt).where(JobAttempt.job_id == job_id).order_by(JobAttempt.attempt_no)
            )
        )
        return [
            {
                "id": a.id,
                "attempt_no": a.attempt_no,
                "status": a.status,
                "exit_code": a.exit_code,
                "gpu_uuids": a.gpu_uuids,
                "mlflow_run_id": getattr(a, "mlflow_run_id", None),
                "log_uri": a.log_uri,
            }
            for a in rows
        ]


@router.post("/{job_id}/cancel")
def cancel(job_id: int):
    with session_scope() as s:
        ok = jobsvc.cancel(s, job_id)
        if not ok:
            raise HTTPException(409, "job not cancellable")
    return {"ok": True, "job_id": job_id}


@router.get("/{job_id}/logs", response_model=LogOut)
def logs(job_id: int, tail: int = 200):
    with session_scope() as s:
        attempt = s.scalars(
            select(JobAttempt).where(JobAttempt.job_id == job_id).order_by(JobAttempt.attempt_no.desc())
        ).first()
        if attempt is None:
            raise HTTPException(404, "no attempt yet")
        path = jobsvc.log_path(attempt)
        return LogOut(
            job_id=job_id,
            attempt_id=attempt.id,
            path=str(path),
            lines=jobsvc.tail(path, tail),
        )


@router.get("/{job_id}/logs/stream")
def stream_logs(job_id: int):
    def gen():
        sent = 0
        for _ in range(600):
            with session_scope() as s:
                attempt = s.scalars(
                    select(JobAttempt)
                    .where(JobAttempt.job_id == job_id)
                    .order_by(JobAttempt.attempt_no.desc())
                ).first()
                if attempt is None:
                    yield "event: waiting\ndata: {}\n\n"
                    time.sleep(0.5)
                    continue
                path = jobsvc.log_path(attempt)
                lines = jobsvc.tail(path, 5000)
                job_status = s.scalar(select(Job.status).where(Job.id == job_id))
            for line in lines[sent:]:
                yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
            sent = len(lines)
            if job_status in {"completed", "failed", "cancelled"}:
                yield f"event: end\ndata: {json.dumps({'status': job_status})}\n\n"
                return
            time.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")
