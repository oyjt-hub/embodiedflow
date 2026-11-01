"""离线评测：从执行日志算时长、延迟、成功率、MAE/RMSE、速度/加速度方差、Jerk。

口径（与计划 P0.5 一致）：
- 所有比例都输出 value + numerator + denominator
- 参考动作缺失时 MAE/RMSE 给 None 并附 reason，不伪造成 0
- 成功率按冻结的任务协议处理终态与超时；未标注结果单列覆盖率
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RULES_VERSION = "1.0"


@dataclass
class EpisodeLog:
    steps: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    episode_id: str = "ep0"

    @property
    def duration_s(self) -> float:
        if len(self.steps) < 2:
            return 0.0
        return float(self.steps[-1]["t"] - self.steps[0]["t"])

    def paired(self) -> tuple[np.ndarray, np.ndarray]:
        pred, ref = [], []
        for s in self.steps:
            if s.get("pred") is not None and s.get("ref") is not None:
                pred.append(s["pred"])
                ref.append(s["ref"])
        if not pred:
            return np.zeros((0, 0)), np.zeros((0, 0))
        return np.asarray(pred, dtype=np.float64), np.asarray(ref, dtype=np.float64)

    def times(self) -> np.ndarray:
        return np.asarray([s["t"] for s in self.steps], dtype=np.float64)

    def latencies(self) -> np.ndarray:
        return np.asarray([s["latency_ms"] for s in self.steps if s.get("latency_ms") is not None],
                          dtype=np.float64)


def parse_log(path: str | Path) -> EpisodeLog:
    p = Path(path)
    log = EpisodeLog(episode_id=p.stem)
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("event"):
            log.events.append(row)
        else:
            log.steps.append(row)
    return log


def _per_joint(a: np.ndarray, b: np.ndarray) -> list[float] | None:
    if a.size == 0:
        return None
    return np.abs(a - b).mean(axis=0).round(6).tolist()


def _var_or_none(x: np.ndarray) -> float | None:
    """单样本的方差按 0 处理（定义上无离散）；无样本才返回 None。"""
    if x.size == 0:
        return None
    if x.size == 1:
        return 0.0
    return float(np.var(x))


def evaluate(log: EpisodeLog) -> dict:
    t = log.times()
    dur = log.duration_s
    n = len(log.steps)
    pred, ref = log.paired()
    lat = log.latencies()

    metrics: dict = {}
    metrics["duration_s"] = round(dur, 6)
    metrics["steps"] = n
    metrics["throughput_hz"] = round(n / dur, 6) if dur > 0 else None

    # 延迟
    if lat.size:
        metrics["latency_p50_ms"] = round(float(np.percentile(lat, 50)), 3)
        metrics["latency_p95_ms"] = round(float(np.percentile(lat, 95)), 3)
        metrics["latency_n"] = int(lat.size)
    else:
        metrics["latency_p50_ms"] = None
        metrics["latency_p95_ms"] = None
        metrics["latency_n"] = 0

    # 精度
    if pred.size:
        diff = pred - ref
        metrics["mae"] = round(float(np.abs(diff).mean()), 6)
        metrics["rmse"] = round(float(np.sqrt((diff ** 2).mean())), 6)
        metrics["mae_per_joint"] = _per_joint(pred, ref)
        metrics["accuracy_source"] = "paired_pred_ref"
    else:
        metrics["mae"] = None
        metrics["rmse"] = None
        metrics["mae_per_joint"] = None
        metrics["accuracy_source"] = "N/A：日志缺少成对的 pred/ref"

    # 运动学（用参考轨迹，缺则用预测）
    traj = ref if ref.size else pred
    if traj.size > 1 and dur > 0:
        dt = np.diff(t) if t.size == traj.shape[0] else None
        if dt is None or np.any(dt <= 0):
            dt = np.full(traj.shape[0] - 1, dur / max(traj.shape[0] - 1, 1))
        vel = np.diff(traj, axis=0) / dt[:, None]
        metrics["speed_variance"] = _var_or_none(np.linalg.norm(vel, axis=1))
        if vel.shape[0] > 1:
            dt2 = dt[1:][:, None]
            acc = np.diff(vel, axis=0) / dt2
            metrics["accel_variance"] = _var_or_none(np.linalg.norm(acc, axis=1))
            if acc.shape[0] > 1:
                dt3 = dt[2:][:, None] if dt.size > 2 else np.full(acc.shape[0] - 1, 1.0)[:, None]
                jerk = np.diff(acc, axis=0) / dt3
                metrics["jerk_mean"] = round(float(np.abs(jerk).mean()), 6) if jerk.size else None
            else:
                metrics["jerk_mean"] = None
        else:
            metrics["accel_variance"] = None
            metrics["jerk_mean"] = None
    else:
        metrics["speed_variance"] = None
        metrics["accel_variance"] = None
        metrics["jerk_mean"] = None

    # 任务结果
    outcome = None
    interventions = 0
    for e in log.events:
        if e["event"] in {"success", "failure", "timeout"}:
            outcome = e["event"]
        if e["event"] == "intervention":
            interventions += 1
    metrics["outcome"] = outcome
    metrics["interventions"] = interventions
    metrics["success"] = None if outcome is None else (outcome == "success")

    return {
        "rules_version": RULES_VERSION,
        "episode": log.episode_id,
        "metrics": metrics,
        "denominators": {
            "steps": n,
            "latency_samples": int(lat.size),
            "paired_samples": int(pred.shape[0]),
            "outcome_labeled": 0 if outcome is None else 1,
        },
    }


def summarize(reports: list[dict]) -> dict:
    """跨 Episode 汇总：成功率带覆盖率，未标注结果单列。"""
    total = len(reports)
    labeled = [r for r in reports if r["metrics"]["success"] is not None]
    ok = [r for r in labeled if r["metrics"]["success"]]
    return {
        "episodes": total,
        "outcome_labeled": len(labeled),
        "outcome_coverage": round(len(labeled) / total, 6) if total else None,
        "success": {
            "value": round(len(ok) / len(labeled), 6) if labeled else None,
            "numerator": len(ok),
            "denominator": len(labeled),
        },
        "unlabeled": total - len(labeled),
    }


def to_csv(reports: list[dict]) -> str:
    keys = ["episode", "duration_s", "steps", "throughput_hz", "latency_p50_ms", "latency_p95_ms",
            "mae", "rmse", "speed_variance", "accel_variance", "jerk_mean", "outcome",
            "interventions", "success"]
    lines = [",".join(keys)]
    for r in reports:
        m = r["metrics"]
        lines.append(",".join(str(m.get(k, "")) if m.get(k) is not None else "" for k in keys))
    return "\n".join(lines) + "\n"
