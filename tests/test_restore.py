"""启动恢复的防护分支：存活跳过、权重缺失标 failed、超过上限跳过。"""
from __future__ import annotations

import os

from persistence.db import session_scope
from persistence.migrate import upgrade
from persistence.models import WorkbenchDeployment


def _mk(model_dir: str, pid: int, status: str = "running") -> int:
    with session_scope() as s:
        rec = WorkbenchDeployment(model_dir=model_dir, port=0, pid=pid,
                                  endpoint="", status=status)
        s.add(rec)
        s.flush()
        return rec.id


def test_alive_pid_is_skipped(tmp_path):
    upgrade(verbose=False)
    did = _mk(str(tmp_path), os.getpid())          # 当前进程一定活着

    from apps.api.routers.workbench import restore_deployments

    info = restore_deployments()
    ids = [x["id"] for x in info["skipped"]]
    assert did in ids, "存活进程应被跳过"
    assert did not in [x["id"] for x in info["restored"]]


def test_missing_checkpoint_marks_failed(tmp_path):
    upgrade(verbose=False)
    did = _mk(str(tmp_path), 999999)               # 死 pid + 空目录

    from apps.api.routers.workbench import restore_deployments

    info = restore_deployments()
    assert did in [x["id"] for x in info["failed"]]
    with session_scope() as s:
        assert s.get(WorkbenchDeployment, did).status == "failed"


def test_non_running_rows_ignored(tmp_path):
    upgrade(verbose=False)
    did = _mk(str(tmp_path), 999999, status="stopped")

    from apps.api.routers.workbench import restore_deployments

    info = restore_deployments()
    touched = {x["id"] for k in ("restored", "skipped", "failed") for x in info[k]}
    assert did not in touched, "非 running 记录不应被处理"
