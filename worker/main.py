from __future__ import annotations

import argparse
import os
import shlex
import signal
import socket
import subprocess
import sys
import time

from core.enums import AttemptStatus, JobStatus
from core.config import ensure_dirs
from persistence.db import session_scope
from persistence.migrate import upgrade
from persistence.models import Job, JobAttempt
from services import jobs as jobsvc
from services import recovery as recsvc
from services import training_autolog
from services import resources as ressvc

TERMINAL = {JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value}


def build_command(job: Job) -> list[str] | None:
    kind = job.kind
    if kind == "smoke":
        script = (
            "import os,time,sys;"
            "print('embodiedflow worker ok');"
            "print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES',''));"
            "print('python', sys.version.split()[0]);"
            "[ (print('tick', i, flush=True), time.sleep(0.4)) for i in range(5) ];"
            "print('done')"
        )
        return [sys.executable, "-u", "-c", script]
    if kind == "gpu_probe":
        return [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ]
    cmd = (job.spec or {}).get("command")
    if not cmd:
        return None
    return shlex.split(cmd) if isinstance(cmd, str) else [str(c) for c in cmd]


def run_job(worker_id: str) -> bool:
    with session_scope() as s:
        ressvc.sync_gpus(s)
        claimed = jobsvc.claim_next(
            s, worker_id, allocator=lambda sess, j: ressvc.plan_allocation(sess, j)
        )
        if claimed is None:
            return False
        job, attempt, take = claimed
        cmd = build_command(job)
        if cmd is None:
            jobsvc.finish_attempt(
                s, attempt,
                status=AttemptStatus.FAILED.value,
                error=f"no adapter implemented for kind={job.kind}",
            )
            return True
        job_id, attempt_id = job.id, attempt.id
        job_spec = dict(job.spec or {})
        log_file = jobsvc.log_path(attempt)

    env = dict(os.environ)
    env["EF_JOB_ID"] = str(job_id)
    env["EF_ATTEMPT_ID"] = str(attempt_id)
    env["PYTHONUNBUFFERED"] = "1"
    if take:
        # 用租约里的 GPU UUID，而不是重新编号的 0..n-1：
        # 重新编号会让"申请物理卡 3"的任务实际跑在卡 0 上。
        env["CUDA_VISIBLE_DEVICES"] = ",".join(take)
    for k, v in (job_spec.get("env") or {}).items():
        env[str(k)] = str(v)
    run_cwd = job_spec.get("cwd") or None

    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[worker] job={job_id} attempt={attempt_id} cmd={' '.join(cmd)[:120]}", flush=True)
    with log_file.open("w", encoding="utf-8") as fh:  # 每次 attempt 覆盖，避免复用旧文件
        fh.write(f"# job={job_id} attempt={attempt_id} kind={job.kind}\n")
        fh.write(f"# cmd={' '.join(cmd)}\n")
        fh.write(f"# CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','')}\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
            start_new_session=True, cwd=run_cwd,
        )
        with session_scope() as s:
            att = s.get(JobAttempt, attempt_id)
            att.pid = proc.pid

        cancelled = False
        while proc.poll() is None:
            time.sleep(1.0)
            with session_scope() as s:
                jobsvc.heartbeat(s, attempt_id)
                if jobsvc.is_cancelled(s, job_id):
                    cancelled = True
            if cancelled:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(2)
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break

        code = proc.wait()
        fh.write(f"\n# exit_code={code} cancelled={cancelled}\n")

    status = (
        AttemptStatus.CANCELLED.value
        if cancelled
        else (AttemptStatus.COMPLETED.value if code == 0 else AttemptStatus.FAILED.value)
    )
    with session_scope() as s:
        att = s.get(JobAttempt, attempt_id)
        job_row = s.get(Job, att.job_id)
        auto = {}
        if code == 0 and job_row is not None:
            try:
                auto = training_autolog.finalize(s, job_row, att, code)
            except Exception as exc:  # noqa: BLE001
                print(f"[worker] autolog failed: {exc!r}", flush=True)
        jobsvc.finish_attempt(
            s, att, status=status, exit_code=code,
            error=None if code == 0 else f"exit code {code}",
        )
        if auto:
            print(f"[worker] autolog: {auto}", flush=True)
    print(f"[worker] job={job_id} finished status={status} exit={code}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="EmbodiedFlow worker")
    ap.add_argument("--id", default=f"{socket.gethostname()}-w0")
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    upgrade(verbose=True)
    print(f"[worker] online id={args.id} poll={args.poll}s", flush=True)
    with session_scope() as s:
        stats = recsvc.recover(s, current_worker=args.id)
        print(f"[worker] startup recovery: {stats}", flush=True)

    loop = 0
    while True:
        loop += 1
        if loop % 15 == 0:  # 周期性回收僵尸 attempt
            try:
                with session_scope() as s:
                    stats = recsvc.recover(s, current_worker=args.id)
                if any(stats[k] for k in ("released", "requeued", "failed", "orphans_killed")):
                    print(f"[worker] recovery: {stats}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[worker] recovery error: {exc!r}", flush=True)
        try:
            worked = run_job(args.id)
        except Exception as exc:  # noqa: BLE001
            print(f"[worker] error: {exc!r}", flush=True)
            worked = False
        if args.once and not worked:
            return 0
        if not worked:
            time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
