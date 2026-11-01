"""P1.2 前置：看门狗、限位、急停三条必须成立。"""
from __future__ import annotations

import time

import pytest

from services import edge


@pytest.fixture()
def bridge():
    b = edge.MockBridge()
    b.heartbeat()
    return b


def test_watchdog_blocks_without_heartbeat():
    b = edge.MockBridge()           # 从未心跳
    ok, info = b.submit_actions([0.0] * 6, now=time.time())
    assert not ok and "看门狗" in info["error"]


def test_watchdog_expires(bridge):
    past = time.time() - 5
    ok, info = bridge.submit_actions([0.0] * 6, now=past + 10)
    assert not ok and "看门狗" in info["error"]


def test_joint_limit_and_delta(bridge):
    ok, info = bridge.submit_actions([9.9] + [0.0] * 5, now=time.time())
    assert not ok and "越限" in info["error"]

    assert bridge.submit_actions([0.1] * 6, now=time.time())[0]
    ok, info = bridge.submit_actions([1.5] * 6, now=time.time())   # 单步 Δ1.4
    assert not ok and "变化过大" in info["error"]


def test_estop_requires_manual_reset(bridge):
    bridge.estop("test")
    ok, info = bridge.submit_actions([0.0] * 6, now=time.time())
    assert not ok and "急停" in info["error"]

    bridge.heartbeat()               # 心跳不能解除急停
    ok, info = bridge.submit_actions([0.0] * 6, now=time.time())
    assert not ok, "心跳不应解除急停"

    bridge.reset()
    bridge.heartbeat()
    ok, _ = bridge.submit_actions([0.0] * 6, now=time.time())
    assert ok, "人工复位+心跳后应可执行"


def test_wrong_dimension_rejected(bridge):
    ok, info = bridge.submit_actions([0.0] * 3, now=time.time())
    assert not ok and "维度" in info["error"]
