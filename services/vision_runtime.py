"""Single leased SAM2 runtime slot, shared by serialized interactive requests."""
import atexit
import json
import os
import select
import subprocess
import threading
import time
import uuid
from pathlib import Path

from persistence.db import session_scope
from persistence.models import Job, JobAttempt, utcnow
from services import jobs as jobsvc, resources


class VisionRuntime:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.attempt_id = None
        self.job_id = None
        self.log = None
        self.owners = set()

    def snapshot(self):
        with self.lock:
            active = self.process is not None and self.process.poll() is None
            return {"active": active, "job_id": self.job_id if active else None,
                    "pid": self.process.pid if active else None,
                    "model": "SAM2.1 / Florence-2-large", "idle_seconds": 120,
                    "gpu_lease_held": active, "session_count": len(self.owners) if active else 0}

    def release(self, owner):
        with self.lock:
            self.owners.discard(owner)
            if not self.owners:
                self.close()
            return self.snapshot()

    def _finish(self, failed=False):
        if self.attempt_id:
            with session_scope() as session:
                attempt = session.get(JobAttempt, self.attempt_id)
                if attempt and attempt.status == "running":
                    job = session.get(Job, attempt.job_id)
                    status = "cancelled" if job and job.status == "cancelled" else "failed" if failed else "completed"
                    jobsvc.finish_attempt(session, attempt, status=status,
                                          exit_code=self.process.returncode if self.process else None)
        if self.log:
            self.log.close()
        self.process = self.attempt_id = self.job_id = self.log = None
        self.owners.clear()

    def close(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self._finish()

    def _watch(self, process, attempt_id):
        while process.poll() is None:
            with session_scope() as session:
                jobsvc.heartbeat(session, attempt_id)
                attempt = session.get(JobAttempt, attempt_id)
                cancelled = bool(attempt and jobsvc.is_cancelled(session, attempt.job_id))
            if cancelled:
                process.terminate()
            time.sleep(3)
        with self.lock:
            if self.process is process:
                self._finish(failed=process.returncode != 0)

    def _start(self, python, script, vendor, cwd):
        with session_scope() as session:
            resources.sync_gpus(session)
            candidates = resources.schedulable_gpus(session, need_mb=4096)
            if not candidates:
                raise RuntimeError("没有可用的视觉 GPU，稍后重试")
            gpu = candidates[0]
            job = Job(kind="vision_runtime", name="SAM2 interactive runtime", status="running",
                      spec={"runtime": "sam2_florence", "idle_seconds": 120,
                            "python": str(python), "script": str(script)}, requested_gpus=1,
                      max_attempts=1, created_by="vision_runtime", started_at=utcnow())
            session.add(job); session.flush()
            attempt = JobAttempt(job_id=job.id, status="running", worker_id="api-vision-runtime",
                                 heartbeat_at=utcnow(), gpu_uuids=[gpu])
            session.add(attempt); session.flush()
            jobsvc.lease_gpus(session, attempt, [gpu])
            log_path = jobsvc.log_path(attempt)
            attempt.log_uri = str(log_path)
            self.job_id, self.attempt_id = job.id, attempt.id
        try:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                       PYTHONPATH=str(vendor), OMP_NUM_THREADS="4",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            self.log = Path(log_path).open("a", encoding="utf-8")
            self.process = subprocess.Popen([str(python), "-u", str(script), "--serve", "120"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                bufsize=1, cwd=str(cwd), env=env, start_new_session=True)
            with session_scope() as session:
                session.get(JobAttempt, self.attempt_id).pid = self.process.pid
            threading.Thread(target=self._watch, args=(self.process, self.attempt_id), daemon=True).start()
        except Exception:
            self._finish(failed=True)
            raise

    def execute(self, request, python, script, vendor, cwd, owner="api"):
        with self.lock:
            if self.process and self.process.poll() is not None:
                self._finish(failed=self.process.returncode != 0)
            if not self.process:
                self._start(python, script, vendor, cwd)
            self.owners.add(owner)
            request_id = uuid.uuid4().hex
            self.process.stdin.write(json.dumps({**request, "request_id": request_id}) + "\n")
            self.process.stdin.flush()
            if not select.select([self.process.stdout], [], [], 180)[0]:
                self.close()
                raise RuntimeError("SAM2 runtime response timed out")
            line = self.process.stdout.readline()
            if not line:
                self.close()
                raise RuntimeError("SAM2 runtime exited; check runtime Job log")
            response = json.loads(line)
            if response.get("request_id") != request_id or not response.get("success"):
                raise RuntimeError(response.get("error", "invalid runtime response"))
            return {**response["result"], "runtime_job_id": self.job_id}


runtime = VisionRuntime()
atexit.register(runtime.close)

