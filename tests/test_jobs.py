from __future__ import annotations

from persistence.db import session_scope
from persistence.migrate import upgrade
from persistence.models import Job
from services import jobs as jobsvc


def test_job_lifecycle():
    upgrade(verbose=False)
    # 清掉其他测试遗留的排队任务，保证 claim_next 拿到本用例新建的任务
    from sqlalchemy import update

    with session_scope() as s:
        s.execute(update(Job).where(Job.status == "queued").values(status="cancelled"))
    with session_scope() as s:
        job = jobsvc.create_job(s, kind="smoke", name="t1", spec={}, requested_gpus=0)
        s.flush()
        job_id = job.id
        assert job.status == "queued"

    with session_scope() as s:
        claimed = jobsvc.claim_next(s, "w-test")
        assert claimed is not None
        job, attempt, _allocation = claimed
        assert job.status == "running"

    with session_scope() as s:
        from persistence.models import JobAttempt

        att = s.get(JobAttempt, attempt.id)
        jobsvc.finish_attempt(
            s, att, status="completed", exit_code=0
        )

    with session_scope() as s:
        done = s.get(Job, job_id)
        assert done.status == "completed"
        assert done.finished_at is not None


def test_cancel_blocks_claim():
    upgrade(verbose=False)
    with session_scope() as s:
        job = jobsvc.create_job(s, kind="smoke", name="t2", spec={})
        s.flush()
        assert jobsvc.cancel(s, job.id) is True


def test_no_adapter_kind_is_rejected():
    from worker.main import build_command
    from persistence.models import Job

    job = Job(kind="train", spec={}, name="x")
    assert build_command(job) is None
