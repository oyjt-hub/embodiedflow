"""P1.1 机械臂 reach 基准（可用）。

关键坑：MuJoCo 的 autolimits 会按几何尺寸自动生成关节限位，把
`range="-3.14 3.14"` 覆盖成 ±0.0548 rad，导致关节被约束力卡死、
执行器出力被完全抵消（qfrc_constraint ≡ -qfrc_actuator）。
修法：`<compiler autolimits="false"/>` + 关节显式 `limited="false"`。



任务：末端从初始姿态移动到目标点，容差 5 cm。
策略：proportional（解析 IK → PD 跟踪）vs random（随机力矩）vs passive（零力矩）。
判据、容差、步数全部冻结在 spec 中，同 seed → 同结果。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

L1, L2 = 0.30, 0.25
MJCF = f"""
<mujoco>
  <compiler autolimits="false"/>
  <option timestep="0.005" gravity="0 0 0"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" damping="2.0" armature="0.05" limited="false"/>
      <geom type="capsule" fromto="0 0 0 {L1} 0 0" size="0.03" rgba="0.3 0.6 0.9 1"/>
      <body name="link2" pos="{L1} 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" damping="2.0" armature="0.05" limited="false"/>
        <geom type="capsule" fromto="0 0 0 {L2} 0 0" size="0.025" rgba="0.9 0.5 0.2 1"/>
        <geom name="tip" type="sphere" pos="{L2} 0 0" size="0.02" rgba="1 0 0 1"/>
      </body>
    </body>
    <body name="target_marker" pos="0.35 0.15 0">
      <geom type="sphere" size="0.03" rgba="0.2 0.9 0.3 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="j1" kp="60" kv="8" ctrlrange="-3.14 3.14" ctrllimited="true"/>
    <position joint="j2" kp="60" kv="8" ctrlrange="-3.14 3.14" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

RULES_VERSION = "1.0"
DEFAULT_SPEC = {
    "engine": "mujoco",
    "task": "arm_reach",
    "steps": 300,
    "target": [0.35, 0.15],
    "tolerance_m": 0.05,
}


def forward_kinematics(q1: float, q2: float) -> tuple[float, float]:
    return (L1 * math.cos(q1) + L2 * math.cos(q1 + q2),
            L1 * math.sin(q1) + L2 * math.sin(q1 + q2))


def inverse_kinematics(x: float, y: float, elbow_up: bool = False) -> tuple[float, float] | None:
    d = (x ** 2 + y ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if d < -1.0 or d > 1.0:
        return None
    q2 = math.acos(d) * (1 if elbow_up else -1)
    q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return q1, q2


def run_episode(seed: int, spec: dict | None = None, policy: str = "proportional",
                collect_trace: bool = False) -> dict:
    import mujoco

    s = {**DEFAULT_SPEC, **(spec or {})}
    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)

    data.qpos[0] = float(rng.uniform(-0.3, 0.3))
    data.qpos[1] = float(rng.uniform(-0.3, 0.3))
    mujoco.mj_forward(model, data)

    tx, ty = float(s["target"][0]), float(s["target"][1])
    ik = inverse_kinematics(tx, ty)

    best = float("inf")
    trace: list[list[float]] = []
    for _ in range(int(s["steps"])):
        if policy == "proportional" and ik is not None:
            u = [ik[0], ik[1]]                      # 位置执行器：直接给目标角
        elif policy == "random":
            u = rng.uniform(-3.0, 3.0, size=2).tolist()
        else:
            u = [0.0, 0.0]                          # passive：保持 0 位姿
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        x, y = forward_kinematics(float(data.qpos[0]), float(data.qpos[1]))
        d_now = math.hypot(x - tx, y - ty)
        best = min(best, d_now)
        if collect_trace:
            trace.append([round(float(data.time), 4), round(float(data.qpos[0]), 5),
                          round(float(data.qpos[1]), 5), round(x, 5), round(y, 5),
                          round(d_now, 5)])

    x, y = forward_kinematics(float(data.qpos[0]), float(data.qpos[1]))
    final = math.hypot(x - tx, y - ty)
    out = {
        "seed": seed,
        "policy": policy,
        "final_distance_m": round(final, 6),
        "best_distance_m": round(best, 6),
        "success": final < float(s["tolerance_m"]),
        "steps": int(s["steps"]),
    }
    if collect_trace:
        out["trace"] = trace
    return out


def run_benchmark(seeds: list[int], spec: dict | None = None, policy: str = "proportional",
                  replay_dir: str | None = None) -> dict:
    s = {**DEFAULT_SPEC, **(spec or {})}
    episodes = []
    for seed in seeds:
        ep = run_episode(seed, s, policy, collect_trace=bool(replay_dir))
        if replay_dir and not ep["success"]:
            trace = ep.pop("trace", [])
            d = Path(replay_dir)
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"fail_{s['task']}_{policy}_seed{seed}.jsonl"
            with f.open("w", encoding="utf-8") as fh:
                fh.write("# t,q1,q2,tip_x,tip_y,distance_m\n")
                for row in trace:
                    fh.write(",".join(str(v) for v in row) + "\n")
            ep["replay"] = str(f)
        else:
            ep.pop("trace", None)
        episodes.append(ep)
    ok = sum(1 for e in episodes if e["success"])
    return {
        "rules_version": RULES_VERSION, "engine": s["engine"], "task": s["task"],
        "policy": policy, "spec": s, "episodes": episodes,
        "success": {"value": round(ok / len(episodes), 6) if episodes else None,
                    "numerator": ok, "denominator": len(episodes)},
    }


def compare(seeds: list[int], policies: list[str], spec: dict | None = None,
            replay_dir: str | None = None) -> dict:
    results = {p: run_benchmark(seeds, spec, p, replay_dir=replay_dir) for p in policies}
    return {
        "rules_version": RULES_VERSION, "task": (spec or {}).get("task", DEFAULT_SPEC["task"]),
        "seeds": list(seeds), "spec": {**DEFAULT_SPEC, **(spec or {})},
        "policies": {p: {"success": r["success"]} for p, r in results.items()},
        "episodes": {p: r["episodes"] for p, r in results.items()},
    }
