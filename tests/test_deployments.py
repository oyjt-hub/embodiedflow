"""部署记录的生命周期：持久化、僵尸纠正、停止后拒绝调用。"""
from __future__ import annotations

from persistence.db import session_scope
from persistence.migrate import upgrade
from persistence.models import WorkbenchDeployment


def _mk(status: str = "running", pid: int = 999999) -> int:
    with session_scope() as s:
        rec = WorkbenchDeployment(model_dir="/tmp/fake-model", port=1234,
                                  pid=pid, endpoint="http://127.0.0.1:1234/predict",
                                  status=status)
        s.add(rec)
        s.flush()
        return rec.id


def test_record_persists_with_fields():
    upgrade(verbose=False)
    did = _mk()
    with session_scope() as s:
        rec = s.get(WorkbenchDeployment, did)
        assert rec.model_dir == "/tmp/fake-model"
        assert rec.status == "running"
        assert rec.endpoint.endswith("/predict")


def test_dead_pid_is_reconciled_to_stopped():
    """进程已死但标记 running 的记录，读列表时必须被纠正。"""
    upgrade(verbose=False)
    _mk(status="running", pid=999999)

    from apps.api.routers.workbench import list_deployments

    rows = list_deployments()
    stale = [r for r in rows if r["model_dir"] == "/tmp/fake-model"]
    assert stale, "应能读到记录"
    assert all(r["status"] == "stopped" for r in stale), "死进程必须被改成 stopped"


def test_stop_is_idempotent():
    upgrade(verbose=False)
    did = _mk(status="stopped")

    from apps.api.routers.workbench import stop

    assert stop(did)["status"] == "stopped"
    assert stop(did)["status"] == "stopped"


def test_predict_rejects_non_running():
    import pytest
    from fastapi import HTTPException

    from apps.api.routers.workbench import PredictIn, predict

    upgrade(verbose=False)
    did = _mk(status="stopped")
    with pytest.raises(HTTPException) as e:
        predict(did, PredictIn(state=[0.0] * 14))
    assert e.value.status_code == 404
