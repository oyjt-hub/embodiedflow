from __future__ import annotations

import datetime as dt
from pathlib import Path

from sqlalchemy import func, select, update

from core.config import settings
from core.enums import AttemptStatus, JobStatus
from persistence.models import Job, JobAttempt, GPULease, utcnow


def create_job(
    session,
    *,
    kind: str,
    name: str = "",
    spec: dict | None = None,
    requested_gpus: int = 0,
    priority: int = 100,
    max_attempts: int = 2,
    created_by: str = "system",
    dataset_version_id: int | None = None,
) -> Job:
    job = Job(
        kind=kind,
        name=name or kind,
        spec=spec or {},
        requested_gpus=requested_gpus,
        priority=priority,
        max_attempts=max_attempts,
        created_by=created_by,
        dataset_version_id=dataset_version_id,
        status=JobStatus.QUEUED.value,
    )
    session.add(job)
    session.flush()
    return job


def claim_next(session, worker_id: str, allocator=None) -> tuple[Job, JobAttempt, list[str]] | None:
    """资源感知的乐观领取。

    allocator(session, job) -> list[uuid] | None
        None 表示资源暂时不足，跳过该任务（留在队列里，不消耗 attempt）。
    """
    candidates = list(
        session.scalars(
            select(Job)
            .where(Job.status == JobStatus.QUEUED.value)
            .order_by(Job.priority.asc(), Job.id.asc())
            .limit(20)
        )
    )
    for job in candidates:
        allocation = allocator(session, job) if allocator is not None else []
        if allocator is not None and allocation is None:
            continue  # 资源不够，让它继续排队
        claimed = session.execute(
            update(Job)
            .where(Job.id == job.id, Job.status == JobStatus.QUEUED.value)
            .values(status=JobStatus.RUNNING.value, started_at=utcnow())
        ).rowcount
        if claimed != 1:
            continue
        last = session.scalars(
            select(JobAttempt)
            .where(JobAttempt.job_id == job.id)
            .order_by(JobAttempt.attempt_no.desc())
            .limit(1)
        ).first()
        attempt = JobAttempt(
            job_id=job.id,
            attempt_no=(last.attempt_no + 1) if last else 1,
            status=AttemptStatus.RUNNING.value,
            worker_id=worker_id,
            log_uri="",
        )
        session.add(attempt)
        session.flush()
        attempt.log_uri = settings.relative(log_path(attempt))
        session.flush()
        for uuid in allocation or []:
            session.add(GPULease(gpu_uuid=uuid, attempt_id=attempt.id))
        attempt.gpu_uuids = list(allocation or [])
        session.flush()
        session.refresh(job)
        return job, attempt, list(allocation or [])
    return None


def attempts_used(session, job_id: int) -> int:
    return int(
        session.scalar(select(func.count()).select_from(JobAttempt).where(JobAttempt.job_id == job_id)) or 0
    )


def requeue(session, job: Job, reason: str) -> None:
    job.status = JobStatus.QUEUED.value
    job.started_at = None
    job.error = reason[:2000]


def log_path(attempt: JobAttempt) -> Path:
    d = settings.runs_dir / f"job_{attempt.job_id}" / f"attempt_{attempt.attempt_no}"
    d.mkdir(parents=True, exist_ok=True)
    return d / "worker.log"


def lease_gpus(session, attempt: JobAttempt, uuids: list[str]) -> None:
    for uuid in uuids:
        session.add(GPULease(gpu_uuid=uuid, attempt_id=attempt.id))
    attempt.gpu_uuids = list(uuids)
    session.flush()


def release_leases(session, attempt_id: int) -> None:
    session.execute(
        update(GPULease)
        .where(GPULease.attempt_id == attempt_id, GPULease.released_at.is_(None))
        .values(released_at=utcnow())
    )


def finish_attempt(
    session,
    attempt: JobAttempt,
    *,
    status: str,
    exit_code: int | None = None,
    error: str | None = None,
) -> None:
    attempt.status = status
    attempt.exit_code = exit_code
    attempt.finished_at = utcnow()
    release_leases(session, attempt.id)
    job = session.get(Job, attempt.job_id)
    if job is not None:
        job.status = {
            AttemptStatus.COMPLETED.value: JobStatus.COMPLETED.value,
            AttemptStatus.FAILED.value: JobStatus.FAILED.value,
            AttemptStatus.CANCELLED.value: JobStatus.CANCELLED.value,
        }.get(status, JobStatus.FAILED.value)
        job.finished_at = utcnow()
        if error:
            job.error = error[:2000]


def cancel(session, job_id: int) -> bool:
    job = session.get(Job, job_id)
    if job is None:
        return False
    if job.status in {JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value}:
        return False
    job.status = JobStatus.CANCELLED.value
    job.finished_at = utcnow()
    return True


def heartbeat(session, attempt_id: int) -> None:
    session.execute(
        update(JobAttempt).where(JobAttempt.id == attempt_id).values(heartbeat_at=utcnow())
    )


def is_cancelled(session, job_id: int) -> bool:
    status = session.scalar(select(Job.status).where(Job.id == job_id))
    return status == JobStatus.CANCELLED.value


def tail(log_file: Path, max_lines: int = 200) -> list[str]:
    if not log_file.exists():
        return []
    with log_file.open("r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]
