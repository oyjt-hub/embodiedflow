"""P1.1：arm_reach 必须有区分度——比例控制应显著优于随机策略。"""
from __future__ import annotations

import pytest

from services import benchmark_arm as bma


def test_ik_reaches_exact_target():
    q1, q2 = bma.inverse_kinematics(0.35, 0.15)
    x, y = bma.forward_kinematics(q1, q2)
    assert abs(x - 0.35) < 1e-6 and abs(y - 0.15) < 1e-6


def test_proportional_beats_random():
    """用最终距离衡量区分度——不依赖容差是否标定得刚好。"""
    r = bma.compare([0, 1, 2, 3, 4], ["proportional", "random"])

    def mean_dist(policy: str) -> float:
        eps = r["episodes"][policy]
        return sum(e["final_distance_m"] for e in eps) / len(eps)

    dp, dr = mean_dist("proportional"), mean_dist("random")
    print(f"  mean_final_distance: proportional={dp:.4f} m  random={dr:.4f} m")
    pp = r["policies"]["proportional"]["success"]["numerator"]
    assert pp >= 1, f"比例策略 0 次成功（距离 {dp:.4f} m），执行器很可能没生效"
    assert dp < dr - 0.05, f"区分度不足：proportional={dp:.4f} random={dr:.4f}（差 <5cm）"


def test_reproducible():
    a = bma.compare([0, 1, 2], ["proportional", "random"])
    b = bma.compare([0, 1, 2], ["proportional", "random"])
    assert a["episodes"] == b["episodes"]
