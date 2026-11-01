"""P1.1：基准必须可复现（同 seed 同结果），成功率带分母。"""
from __future__ import annotations

from services import benchmark as bms


def test_same_seed_same_result():
    a = bms.run_episode(7)
    b = bms.run_episode(7)
    assert a == b, "同一种子必须得到完全相同的结果"


def test_different_seed_may_differ():
    results = {bms.run_episode(s)["max_angle_rad"] for s in (1, 2, 3)}
    assert len(results) > 1, "不同种子应产生不同初态"


def test_benchmark_reports_denominator():
    r = bms.run_benchmark([0, 1, 2])
    assert r["success"]["denominator"] == 3
    assert 0 <= r["success"]["value"] <= 1
    assert len(r["episodes"]) == 3
