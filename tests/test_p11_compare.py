"""P1.1：多策略必须在同场景同种子下对比，且结果可复现。"""
from __future__ import annotations

from services import benchmark as bms


def test_compare_runs_all_policies():
    r = bms.compare([0, 1, 2, 3], ["passive", "pd_balance"])
    assert set(r["policies"]) == {"passive", "pd_balance"}
    for p in ("passive", "pd_balance"):
        assert r["policies"][p]["success"]["denominator"] == 4
        assert len(r["episodes"][p]) == 4


def test_compare_is_reproducible():
    a = bms.compare([0, 1, 2], ["passive", "pd_balance"])
    b = bms.compare([0, 1, 2], ["passive", "pd_balance"])
    assert a["episodes"] == b["episodes"]


def test_pd_policy_actually_differs():
    """控制必须作用到轨迹：逐种子的 max_angle 序列不能完全相同。

    不设人为阈值——只要控制生效，两条轨迹在浮点精度上就必然不同。
    """
    r = bms.compare([0, 1, 2, 3, 4, 5], ["passive", "pd_balance"], {"steps": 600})
    a = [e["max_angle_rad"] for e in r["episodes"]["passive"]]
    b = [e["max_angle_rad"] for e in r["episodes"]["pd_balance"]]
    print(f"  passive={a}")
    print(f"  pd     ={b}")
    assert a != b, "两策略逐种子结果完全相同，控制未生效"
    assert all(e["policy"] == "passive" for e in r["episodes"]["passive"])
    assert all(e["policy"] == "pd_balance" for e in r["episodes"]["pd_balance"])
