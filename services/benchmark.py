"""P1.1 最小基准：确定性 MuJoCo cart-pole，固定种子集合。

成功判据（冻结在 BenchmarkVersion.spec 里）：仿真 N 步后摆杆角度绝对值 < 阈值。
同一 seed 必须得到同一结果（可复现），这是 Benchmark 可信的前提。
"""
from __future__ import annotations

import numpy as np

MJCF = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="cart" pos="0 0 0.1">
      <joint name="slider" type="slide" axis="1 0 0" range="-2 2"/>
      <geom type="box" size="0.12 0.06 0.04" rgba="0.3 0.6 0.9 1"/>
      <body name="pole" pos="0 0 0.04">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-3.14 3.14" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" rgba="0.9 0.4 0.2 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" ctrlrange="-8 8" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

RULES_VERSION = "1.0"

# 两个可在同一场景/种子下对比的启发式策略（用于打通"多策略同条件评测"流程）
POLICIES = {
    "passive": {"desc": "零控制：不施加力矩，自然摆动", "kp": 0.0, "kd": 0.0},
    "pd_balance": {"desc": "PD 直立控制：u = kp*θ + kd*θ̇", "kp": 25.0, "kd": 4.0},
}
DEFAULT_SPEC = {
    "engine": "mujoco",
    "task": "cartpole_balance",
    "steps": 300,
    "angle_threshold_rad": 0.30,
    "control": "zero",  # 零控制下的自然摆动，用于验证管道
}


def _build():
    import mujoco

    return mujoco.MjModel.from_xml_string(MJCF)


def run_episode(seed: int, spec: dict | None = None, policy: str = "passive") -> dict:
    import mujoco

    s = {**DEFAULT_SPEC, **(spec or {})}
    pol = POLICIES.get(policy, POLICIES["passive"])
    model = _build()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    # 初态扰动（固定种子 → 固定初态）
    data.qpos[0] = float(rng.uniform(-0.2, 0.2))
    data.qpos[1] = float(rng.uniform(-0.5, 0.5))
    mujoco.mj_forward(model, data)

    max_angle = 0.0
    for _ in range(int(s["steps"])):
        data.ctrl[0] = pol["kp"] * data.qpos[1] + pol["kd"] * data.qvel[1]
        mujoco.mj_step(model, data)
        max_angle = max(max_angle, float(abs(data.qpos[1])))

    final_angle = float(abs(data.qpos[1]))
    # 判据：整个 rollout 期间摆角从未超过阈值（保持平衡）
    success = max_angle < float(s["angle_threshold_rad"])
    return {
        "seed": seed,
        "policy": policy,
        "max_angle_rad": round(max_angle, 6),
        "final_angle_rad": round(final_angle, 6),
        "success": success,
        "steps": int(s["steps"]),
    }


def run_benchmark(seeds: list[int], spec: dict | None = None, policy: str = "passive",
                  replay_dir: str | None = None) -> dict:
    """replay_dir 仅为与 arm_reach 保持同一调用签名；cart-pole 不做回放。"""
    s = {**DEFAULT_SPEC, **(spec or {})}
    episodes = [run_episode(seed, s, policy) for seed in seeds]
    ok = sum(1 for e in episodes if e["success"])
    return {
        "rules_version": RULES_VERSION,
        "engine": s["engine"],
        "task": s["task"],
        "spec": s,
        "policy": policy,
        "episodes": episodes,
        "success": {
            "value": round(ok / len(episodes), 6) if episodes else None,
            "numerator": ok,
            "denominator": len(episodes),
        },
    }


def compare(seeds: list[int], policies: list[str], spec: dict | None = None,
            replay_dir: str | None = None) -> dict:
    """同一场景 / 同一初态 / 同一组种子下对比多个策略。"""
    results = {p: run_benchmark(seeds, spec, p) for p in policies}
    return {
        "rules_version": RULES_VERSION,
        "seeds": list(seeds),
        "spec": {**DEFAULT_SPEC, **(spec or {})},
        "policies": {p: {"success": r["success"], "desc": POLICIES.get(p, {}).get("desc", "")}
                     for p, r in results.items()},
        "episodes": {p: r["episodes"] for p, r in results.items()},
    }
