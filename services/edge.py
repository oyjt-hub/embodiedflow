"""机器人侧执行契约（P1.2 前置，先跑 Mock）。

真机接入前必须定死的三件事：
1. 看门狗：超过 timeout 没收到心跳 → 立即停机，且拒绝后续动作
2. 动作校验：维度、关节限位、单步变化率 → 违规直接拒绝，不"尽力执行"
3. 急停：一旦触发，必须人工复位（不可被下一次心跳自动解除）

Mock 实现把状态放内存里，接口形状与真机一致，接真机时只换 transport。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

DOF = 6
JOINT_LIMIT = 3.2          # rad
MAX_STEP_DELTA = 0.35      # rad / 单次下发
WATCHDOG_TIMEOUT_S = 1.0


@dataclass
class EdgeState:
    online: bool = False
    estopped: bool = False
    last_heartbeat: float = 0.0
    last_action: list[float] = field(default_factory=list)
    action_count: int = 0
    rejected: list[str] = field(default_factory=list)

    def watchdog_ok(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.last_heartbeat > 0 and (now - self.last_heartbeat) <= WATCHDOG_TIMEOUT_S


class MockBridge:
    def __init__(self) -> None:
        self.state = EdgeState()

    # --- 心跳 ---
    def heartbeat(self, now: float | None = None) -> dict:
        self.state.last_heartbeat = time.time() if now is None else now
        self.state.online = True
        # 注意：急停不会被心跳自动解除
        return {"online": True, "estopped": self.state.estopped,
                "watchdog_ok": True}

    def estop(self, reason: str = "manual") -> dict:
        self.state.estopped = True
        self.state.online = False
        self.state.rejected.append(f"estop: {reason}")
        return {"estopped": True, "reason": reason}

    def reset(self) -> dict:
        """人工复位：清急停，但需要重新心跳才会恢复在线。"""
        self.state.estopped = False
        self.state.online = False
        self.state.last_heartbeat = 0.0
        return {"estopped": False, "online": False, "note": "需重新心跳恢复在线"}

    # --- 动作 ---
    def submit_actions(self, actions: list[float], now: float | None = None) -> tuple[bool, dict]:
        if self.state.estopped:
            return False, {"error": "已急停，需人工复位"}
        if not self.state.watchdog_ok(now):
            return False, {"error": f"看门狗超时（>{WATCHDOG_TIMEOUT_S}s 无心跳）"}
        if len(actions) != DOF:
            return False, {"error": f"动作维度应为 {DOF}，收到 {len(actions)}"}
        if any(abs(a) > JOINT_LIMIT for a in actions):
            return False, {"error": f"关节越限（|q|>{JOINT_LIMIT} rad）"}
        if self.state.last_action:
            delta = max(abs(a - b) for a, b in zip(actions, self.state.last_action))
            if delta > MAX_STEP_DELTA:
                return False, {"error": f"单步变化过大（{delta:.2f} > {MAX_STEP_DELTA} rad）"}
        self.state.last_action = list(actions)
        self.state.action_count += 1
        return True, {"applied": True, "count": self.state.action_count}

    def status(self) -> dict:
        return {
            "online": self.state.online and self.state.watchdog_ok(),
            "estopped": self.state.estopped,
            "watchdog_ok": self.state.watchdog_ok(),
            "action_count": self.state.action_count,
            "dof": DOF,
            "joint_limit": JOINT_LIMIT,
            "max_step_delta": MAX_STEP_DELTA,
            "watchdog_timeout_s": WATCHDOG_TIMEOUT_S,
            "rejected": self.state.rejected[-5:],
        }


BRIDGE = MockBridge()
