"""12 类可控异常的质检规则 + 带分母的指标汇总。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from adapters.datasets.lerobot import EpisodeData

RULES_VERSION = "1.0"

DEFAULT_CFG: dict = {
    "missing_frame_tolerance": 0.05,     # 与主时间轴长度差超过 5% 记为缺帧
    "max_timestamp_gap_s": 0.25,         # 相邻时间戳间隔上限
    "camera_clock_tolerance_s": 0.05,    # 相机与动作时钟允许的最大偏移
    "joint_limit": 3.2,                  # 关节位置限位（rad，按本体版本化）
    "action_jump_threshold": 2.0,        # 单步动作跳变阈值（rad）
    "trajectory_break_threshold": 10.0,  # 单步状态断裂阈值（rad）
    "gripper_range": (0.0, 1.0),
    "static_ratio_threshold": 0.90,      # 静止帧占比超过该值判为静止段
    "static_eps": 1e-4,
    "corrupt_zero_ratio": 0.50,          # 全零帧占比超过该值判为视频损坏
}


@dataclass
class Finding:
    rule: str
    severity: str            # error | warn
    episode: int
    stream: str
    t_start: float | None
    t_end: float | None
    detail: str
    count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeReport:
    episode: int
    findings: list[Finding] = field(default_factory=list)
    frames: int = 0
    steps: int = 0
    static_frames: int = 0
    image_missing: int = 0
    image_expected: int = 0
    joint_violations: int = 0
    action_discontinuities: int = 0
    timestamps_missing: int = 0
    timestamps_expected: int = 0
    incomplete: bool = False

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)


def _t_at(stream, idx: int) -> float | None:
    try:
        val = float(stream.t[idx])
        return None if np.isnan(val) else val
    except Exception:  # noqa: BLE001
        return None


def check_episode(ep: EpisodeData, cfg: dict | None = None) -> EpisodeReport:
    c = {**DEFAULT_CFG, **(cfg or {})}
    rep = EpisodeReport(episode=ep.index)
    dom = ep.dominant_length()
    rep.frames = dom

    continuous = {n: s for n, s in ep.streams.items() if s.kind == "continuous"}
    discrete = {n: s for n, s in ep.streams.items() if s.kind == "discrete"}
    images = {n: s for n, s in ep.streams.items() if s.kind == "image"}

    # --- 1. 缺时间戳 / 2. 时间倒退 / 3. 异常间隔 ---
    for name, s in {**continuous, **discrete}.items():
        n_expected = s.length
        rep.timestamps_expected += n_expected
        if n_expected == 0:
            continue
        nan_mask = np.isnan(s.t)
        n_nan = int(nan_mask.sum())
        rep.timestamps_missing += n_nan
        if n_nan:
            idx = int(np.argmax(nan_mask))
            rep.add(Finding("missing_timestamp", "error", ep.index, name, _t_at(s, max(idx - 1, 0)),
                            _t_at(s, min(idx + 1, n_expected - 1)),
                            f"{n_nan}/{n_expected} 个时间戳为空", n_nan))
        d = np.diff(s.t)
        bad = np.where(d < 0)[0]
        if bad.size:
            i = int(bad[0])
            rep.add(Finding("time_reversal", "error", ep.index, name, _t_at(s, i), _t_at(s, i + 1),
                            f"{bad.size} 处时间倒退，首处 Δt={d[i]:.4f}s", int(bad.size)))
        finite = d[np.isfinite(d)]
        if finite.size:
            med = float(np.median(finite)) or 1e-6
            limit = max(c["max_timestamp_gap_s"], 5 * med)
            gaps = np.where(finite > limit)[0]
            if gaps.size:
                i = int(gaps[0])
                rep.add(Finding("irregular_gap", "warn", ep.index, name, _t_at(s, i), _t_at(s, i + 1),
                                f"{gaps.size} 处异常间隔，最大 {float(np.max(finite)):.3f}s > {limit:.3f}s",
                                int(gaps.size)))

    # --- 4. 缺帧（与主时间轴比）--- 10. Episode 不完整（与声明长度比）
    incomplete = False
    if ep.declared_length:
        ratio = abs(dom - ep.declared_length) / max(ep.declared_length, 1)
        if ratio > c["missing_frame_tolerance"]:
            incomplete = True
            rep.incomplete = True
            state_stream = ep.streams.get("observation.state")
            t0 = _t_at(state_stream, 0) if state_stream is not None and state_stream.length else None
            t1 = _t_at(state_stream, state_stream.length - 1) if state_stream is not None and state_stream.length else None
            rep.add(Finding("episode_incomplete", "error", ep.index, "(episode)", t0, t1,
                            f"实际 {dom} 帧，声明 {ep.declared_length} 帧（差 {ratio:.1%}）", 1))
    if not incomplete and dom:
        for name, s in ep.streams.items():
            if s.length == 0:
                continue
            actual = s.length  # 图像流长度 = 容器内真实帧数（ffmpeg 读取，不受解码上限影响）
            ratio = abs(actual - dom) / dom
            if ratio > c["missing_frame_tolerance"]:
                rep.add(Finding("missing_frames", "error", ep.index, name, None, None,
                                f"{actual} 帧，主时间轴 {dom} 帧（差 {ratio:.1%}）",
                                abs(actual - dom)))

    # --- 5. 视频损坏 ---
    for name, s in images.items():
        frames = s.frames
        n_declared = s.length
        rep.image_expected += n_declared
        if frames is None or frames.size == 0:
            rep.image_missing += n_declared
            rep.add(Finding("video_corrupt", "error", ep.index, name, None, None,
                            "视频无法解码或帧数为 0", n_declared))
            continue
        if not s.meta.get("decode_truncated"):
            rep.image_missing += max(0, n_declared - frames.shape[0])
        zero = [i for i, f in enumerate(frames) if not f.any()]
        if frames.shape[0] and len(zero) / frames.shape[0] > c["corrupt_zero_ratio"]:
            rep.add(Finding("video_corrupt", "error", ep.index, name,
                            float(zero[0]) / (ep.fps or 1.0) if zero else None,
                            float(zero[-1]) / (ep.fps or 1.0) if zero else None,
                            f"{len(zero)}/{frames.shape[0]} 帧为全零", len(zero)))

    # --- 6. 相机与动作时钟错位 ---
    if images:
        ref = ep.streams.get("action") or ep.streams.get("observation.state")
        if ref is not None and len(ref.t) and any(not np.isnan(x) for x in ref.t):
            ref_mean = float(np.nanmean(ref.t))
            for name, s in images.items():
                if not len(s.t):
                    continue
                off = float(np.nanmean(s.t)) - ref_mean
                if abs(off) > c["camera_clock_tolerance_s"]:
                    rep.add(Finding("camera_action_misalign", "error", ep.index, name, None, None,
                                    f"相机相对动作偏移 {off:+.3f}s（容差 ±{c['camera_clock_tolerance_s']}s）", 1))

    # --- 7. 关节越限 / 8. 动作跳变 / 12. 轨迹断裂 ---
    state = ep.streams.get("observation.state")
    if state is not None and state.values is not None and state.length:
        arr = np.asarray(state.values, dtype=np.float64)
        over = np.abs(arr) > c["joint_limit"]
        n_over = int(over.any(axis=1).sum()) if arr.ndim > 1 else int(over.sum())
        rep.joint_violations += n_over
        if n_over:
            i = int(np.where(over.any(axis=1) if arr.ndim > 1 else over)[0][0])
            rep.add(Finding("joint_limit", "error", ep.index, state.name, _t_at(state, i), _t_at(state, i),
                            f"{n_over} 帧越限（|q|>{c['joint_limit']} rad，最大 {float(np.nanmax(np.abs(arr))):.2f}）",
                            n_over))
        if arr.ndim > 1 and arr.shape[0] > 1:
            d = np.abs(np.diff(arr, axis=0)).max(axis=1)
            rep.steps += int(d.size)
            breaks = np.where(d > c["trajectory_break_threshold"])[0]
            if breaks.size:
                i = int(breaks[0])
                rep.add(Finding("trajectory_break", "error", ep.index, state.name,
                                _t_at(state, i), _t_at(state, i + 1),
                                f"{breaks.size} 处状态断裂，最大 Δ={float(d.max()):.1f}", int(breaks.size)))
            static = np.where(d < c["static_eps"])[0]
            ratio = static.size / d.size
            rep.static_frames += int(static.size)
            if ratio > c["static_ratio_threshold"]:
                rep.add(Finding("static_frames", "warn", ep.index, state.name, _t_at(state, 0),
                                _t_at(state, state.length - 1),
                                f"{ratio:.0%} 的相邻帧几乎无变化（阈值 {c['static_ratio_threshold']:.0%}）",
                                int(static.size)))

    action = ep.streams.get("action")
    if action is not None and action.values is not None and action.length > 1:
        arr = np.asarray(action.values, dtype=np.float64)
        d = np.abs(np.diff(arr, axis=0)).max(axis=1)
        jumps = np.where(d > c["action_jump_threshold"])[0]
        if jumps.size:
            i = int(jumps[0])
            rep.action_discontinuities += int(jumps.size)
            rep.add(Finding("action_jump", "error", ep.index, action.name, _t_at(action, i),
                            _t_at(action, i + 1),
                            f"{jumps.size} 处动作跳变，最大 Δ={float(d.max()):.1f} > {c['action_jump_threshold']}",
                            int(jumps.size)))

    # --- 9. 夹爪异常 ---
    for name, s in {**continuous, **discrete}.items():
        if "gripper" not in name or s.values is None or not s.length:
            continue
        arr = np.asarray(s.values, dtype=np.float64).reshape(s.length, -1)[:, 0]
        lo, hi = c["gripper_range"]
        bad = np.where((arr < lo) | (arr > hi) | ~np.isfinite(arr))[0]
        if bad.size:
            i = int(bad[0])
            rep.add(Finding("gripper_anomaly", "error", ep.index, name, _t_at(s, i), _t_at(s, i),
                            f"{bad.size} 帧越界（应在 [{lo},{hi}]，实际 {float(np.nanmin(arr)):.2f}~"
                            f"{float(np.nanmax(arr)):.2f}）", int(bad.size)))

    return rep


def build_report(adapter, cfg: dict | None = None, max_episodes: int | None = None) -> dict:
    """对整个数据集跑一遍质检，返回带分母的汇总。"""
    task_map = adapter.tasks()
    reports: list[EpisodeReport] = []
    metas = adapter.episode_metas()
    if max_episodes:
        metas = metas[: int(max_episodes)]
    for meta in metas:
        idx = int(meta["episode_index"])
        ep = adapter.read_episode(idx, task_map)
        reports.append(check_episode(ep, cfg))

    findings = [f.to_dict() for r in reports for f in r.findings]

    def metric(num: int, den: int) -> dict:
        return {
            "value": round(num / den, 6) if den else None,
            "numerator": int(num),
            "denominator": int(den),
        }

    incomplete = sum(1 for r in reports if r.incomplete)
    metrics = {
        "timestamp_completeness": metric(
            sum(r.timestamps_expected - r.timestamps_missing for r in reports),
            sum(r.timestamps_expected for r in reports),
        ),
        "episode_completeness": metric(len(reports) - incomplete, len(reports)),
        "joint_violation_rate": metric(sum(r.joint_violations for r in reports),
                                       sum(r.frames for r in reports)),
        "image_missing_rate": metric(sum(r.image_missing for r in reports),
                                     sum(r.image_expected for r in reports)),
        "action_discontinuity_rate": metric(sum(r.action_discontinuities for r in reports),
                                            sum(r.steps for r in reports)),
        "static_frame_ratio": metric(sum(r.static_frames for r in reports),
                                     sum(r.steps for r in reports)),
    }

    by_episode: dict[str, list[str]] = {}
    for f in findings:
        by_episode.setdefault(str(f["episode"]), []).append(f["rule"])

    return {
        "rules_version": RULES_VERSION,
        "episodes": len(reports),
        "findings": findings,
        "by_episode": {k: sorted(set(v)) for k, v in sorted(by_episode.items(), key=lambda kv: int(kv[0]))},
        "rule_hits": {rule: sum(1 for f in findings if f["rule"] == rule)
                      for rule in sorted({f["rule"] for f in findings})},
        "metrics": metrics,
        "config": {**DEFAULT_CFG, **(cfg or {})},
    }
