"""基准必须按 task 分发，未知 task 必须拒绝——否则会给出标错任务的结果。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app


def test_arm_task_goes_to_arm_module():
    with TestClient(app) as c:
        r = c.post("/api/benchmarks/run",
                   json={"name": "arm-reach-t", "version": "v1", "seeds": [0, 1],
                         "task": "arm_reach"})
        assert r.status_code == 201, r.text
        d = r.json()
        assert d["task"] == "arm_reach"
        assert d["policy"] == "proportional"
        # 机械臂成功时距离应接近 0（cartpole 的结果字段完全不同）
        eps = d["episodes"]
        assert all("final_distance_m" in e for e in eps)


def test_cartpole_task_still_works():
    with TestClient(app) as c:
        r = c.post("/api/benchmarks/run",
                   json={"name": "cp-t", "version": "v1", "seeds": [0, 1],
                         "task": "cartpole_balance"})
        assert r.status_code == 201, r.text
        assert r.json()["task"] == "cartpole_balance"


def test_unknown_task_rejected():
    with TestClient(app) as c:
        r = c.post("/api/benchmarks/run",
                   json={"name": "x", "version": "v1", "seeds": [0], "task": "not_a_task"})
        assert r.status_code == 400
        assert "未知 task" in r.json()["detail"]
