"""P0.3 验收：争卡串行、留队等待、Worker 失联恢复与重试预算。"""
from __future__ import annotations

import datetime as dt
import os

import pytest
from sqlalchemy import delete, select

from persistence.db import session_scope
from persistence.migrate import upgrade
from persistence.models import GPULease, GPUResource, Job, JobAttempt, utcnow
from services import jobs as jobsvc
from services import recovery as recsvc
from services import resources as ressvc


def _seed_gpus(n: int = 3, used_mb: int = 1000) -> None:
    with session_scope() as s:
        s.execute(delete(GPULease))
        s.execute(delete(GPUResource))  # 清掉真实卡，测试只认真假卡
        for i in range(n):
            uuid = f"GPU-test-{i}"
            gpu = s.get(GPUResource, uuid)
            if gpu is None:
                gpu = GPUResource(uuid=uuid)
                s.add(gpu)
            gpu.index = i
            gpu.name = "TEST-4090"
            gpu.memory_total_mb = 49140
            gpu.memory_used_mb = used_mb
            gpu.utilization = 50
            gpu.state = "partial"
            gpu.host = "test"


def _clear_jobs() -> None:
    with session_scope() as s:
        s.execute(delete(GPULease))
        s.execute(delete(JobAttempt))
        s.execute(delete(Job))


@pytest.fixture()
def env():
    upgrade(verbose=False)
    _seed_gpus()
    _clear_jobs()
    yield
    _clear_jobs()


def test_no_gpu_job_and_pinned_job(env):
    with session_scope() as s:
        a = jobsvc.create_job(s, kind="smoke", name="cpu-job", spec={}, requested_gpus=0)
        s.flush()
        assert ressvc.plan_allocation(s, a) == []

        pinned = jobsvc.create_job(
            s, kind="smoke", name="pin-7", spec={"gpu_indices": [7]}, requested_gpus=1
        )
        s.flush()
        assert ressvc.plan_allocation(s, pinned) is None, "GPU7 不在资源表里，应规划失败"


def test_two_jobs_competing_for_one_card(env):
    """同一张卡：第一个拿到租约后，第二个必须留队（返回 None，不消耗 attempt）。"""
    with session_scope() as s:
        j1 = jobsvc.create_job(s, kind="smoke", name="j1", spec={"gpu_indices": [0]},
                               requested_gpus=1, max_attempts=2)
        j2 = jobsvc.create_job(s, kind="smoke", name="j2", spec={"gpu_indices": [0]},
                               requested_gpus=1, max_attempts=2)
        s.flush()
        first = jobsvc.claim_next(s, "w1", allocator=ressvc.plan_allocation)
        assert first is not None
        job1, att1, alloc1 = first
        assert job1.id == j1.id and alloc1 == ["GPU-test-0"]

        second = jobsvc.claim_next(s, "w1", allocator=ressvc.plan_allocation)
        assert second is None, "第二张卡被占后不应抢占"
        s.refresh(j2)
        assert j2.status == "queued", "j2 必须留在队列里等待"

    # 释放后第二个才能跑
    with session_scope() as s:
        att = s.get(JobAttempt, att1.id)
        jobsvc.finish_attempt(s, att, status="completed", exit_code=0)
    with session_scope() as s:
        third = jobsvc.claim_next(s, "w1", allocator=ressvc.plan_allocation)
        assert third is not None
        job2, _, alloc2 = third
        assert job2.id == j2.id and alloc2 == ["GPU-test-0"]


def test_unrunnable_job_does_not_block_queue(env):
    """需要一个不存在的卡的任务不能把后面的任务堵死。"""
    with session_scope() as s:
        blocked = jobsvc.create_job(s, kind="smoke", name="needs-3", spec={"gpu_indices": [0]},
                                    requested_gpus=3, max_attempts=1)
        s.flush()
        ok = jobsvc.create_job(s, kind="smoke", name="cpu", spec={}, requested_gpus=0, max_attempts=1)
        s.flush()
        got = jobsvc.claim_next(s, "w1", allocator=ressvc.plan_allocation)
        assert got is not None
        assert got[0].id == ok.id, "应跳过资源不足的任务，领取后面的 CPU 任务"
        s.refresh(blocked)
        assert blocked.status == "queued"


def test_recovery_requeues_dead_attempt(env):
    with session_scope() as s:
        jobsvc.create_job(s, kind="smoke", name="dead", spec={}, requested_gpus=1, max_attempts=2)
    with session_scope() as s:
        job, att, alloc = jobsvc.claim_next(s, "w1", allocator=ressvc.plan_allocation)
        job_id, att_id = job.id, att.id
        att.pid = None
        att.heartbeat_at = utcnow() - dt.timedelta(seconds=120)

    with session_scope() as s:
        stats = recsvc.recover(s, stale_after_s=30)
    assert stats["requeued"] == 1 and stats["released"] == 1

    with session_scope() as s:
        job = s.get(Job, job_id)
        assert job.status == "queued", "有重试预算时应回队"
        leases = list(s.scalars(select(GPULease).where(GPULease.attempt_id == att_id)))
        assert leases and all(l.released_at is not None for l in leases)


def test_recovery_skips_alive_process(env):
    with session_scope() as s:
        jobsvc.create_job(s, kind="smoke", name="alive", spec={}, requested_gpus=1, max_attempts=2)
    with session_scope() as s:
        job, att, _ = jobsvc.claim_next(s, "w1", allocator=ressvc.plan_allocation)
        job_id = job.id
        att.pid = os.getpid()  # 假装进程还活着
        att.heartbeat_at = utcnow() - dt.timedelta(seconds=300)

    with session_scope() as s:
        stats = recsvc.recover(s, stale_after_s=30)
    assert stats["alive_skipped"] == 1
    with session_scope() as s:
        assert s.get(Job, job_id).status == "running", "进程还在就不能判死"


def test_recovery_fails_when_budget_exhausted(env):
    with session_scope() as s:
        jobsvc.create_job(s, kind="smoke", name="nobudget", spec={}, requested_gpus=1, max_attempts=1)
    with session_scope() as s:
        job, att, _ = jobsvc.claim_next(s, "w1", allocator=ressvc.plan_allocation)
        job_id = job.id
        att.pid = None
        att.heartbeat_at = utcnow() - dt.timedelta(seconds=120)
    with session_scope() as s:
        stats = recsvc.recover(s, stale_after_s=30)
    assert stats["failed"] == 1
    with session_scope() as s:
        job = s.get(Job, job_id)
        assert job.status == "failed" and "重试上限" in (job.error or "")
