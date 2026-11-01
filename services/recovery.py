"""Worker 失联恢复：核实进程是否存活，再决定重排或判失败。"""
from __future__ import annotations

import datetime as dt
import os
import signal
from pathlib import Path

from sqlalchemy import select

from core.enums import AttemptStatus, JobStatus
from persistence.models import GPULease, Job, JobAttempt, utcnow
from services import jobs as jobsvc

STALE_AFTER_S = 30


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    return Path(f"/proc/{int(pid)}").exists()


def _kill_group(pid: int) -> bool:
    try:
        os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
        return True
    except Exception:  # noqa: BLE001
        return False


def recover(session, stale_after_s: int = STALE_AFTER_S, current_worker: str | None = None) -> dict:
    """把心跳超时且进程已消失的 running attempt 收尾。

    - 进程仍在 → 跳过（避免误杀别人/自己仍在跑的训练）
    - 进程已消失 → 释放租约；还有重试预算则任务回队，否则判失败
    """
    stats = {"checked": 0, "alive_skipped": 0, "orphans_killed": 0,
             "released": 0, "requeued": 0, "failed": 0}
    cutoff = utcnow() - dt.timedelta(seconds=stale_after_s)
    attempts = list(
        session.scalars(
            select(JobAttempt).where(
                JobAttempt.status == AttemptStatus.RUNNING.value,
                (JobAttempt.heartbeat_at.is_(None)) | (JobAttempt.heartbeat_at < cutoff),
            )
        )
    )
    for att in attempts:
        stats["checked"] += 1
        if _pid_alive(att.pid):
            # 安全护栏 1：自己这个 Worker 的 attempt，永远不动
            if current_worker and att.worker_id == current_worker:
                stats["alive_skipped"] += 1
                continue
            # 安全护栏 2：同一进程组内的进程绝不杀（否则会误杀自己/同机进程）
            #   注意：启动的作业用 start_new_session=True，属于独立进程组，
            #   所以"同组"基本只可能是本 Worker 自身或其父进程。
            try:
                if os.getpgid(int(att.pid)) == os.getpgid(0):
                    stats["alive_skipped"] += 1
                    continue
            except Exception:  # noqa: BLE001
                stats["alive_skipped"] += 1
                continue
            # 其余情况：心跳已过期且不是本进程组的 → 判定为失联 Worker 留下的孤儿
            if _kill_group(att.pid):
                stats["orphans_killed"] += 1
            else:
                stats["alive_skipped"] += 1
                continue

        session.execute(
            __import__("sqlalchemy").update(GPULease)
            .where(GPULease.attempt_id == att.id, GPULease.released_at.is_(None))
            .values(released_at=utcnow())
        )
        stats["released"] += 1

        att.status = AttemptStatus.FAILED.value
        att.finished_at = utcnow()

        job = session.get(Job, att.job_id)
        if job is None:
            continue
        used = jobsvc.attempts_used(session, job.id)
        budget = int(getattr(job, "max_attempts", 1) or 1)
        if used < budget:
            jobsvc.requeue(session, job, f"worker 失联，第 {used} 次尝试已回收，重新排队")
            stats["requeued"] += 1
        else:
            job.status = JobStatus.FAILED.value
            job.error = f"worker 失联，已达重试上限 {budget} 次"
            job.finished_at = utcnow()
            stats["failed"] += 1
    return stats
