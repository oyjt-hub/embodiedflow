"""P0.2 验收：12 类异常能被定位到 Episode。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from adapters.datasets.lerobot import LeRobotAdapter
from services import align, quality
from tests.fixtures.make_lerobot_fixture import ANOMALIES, build

EXPECTED = {
    "missing_timestamp": 1,
    "time_reversal": 2,
    "irregular_gap": 3,
    "missing_frames": 4,
    "video_corrupt": 5,
    "camera_action_misalign": 6,
    "joint_limit": 7,
    "action_jump": 8,
    "gripper_anomaly": 9,
    "episode_incomplete": 10,
    "static_frames": 11,
    "trajectory_break": 12,
}


@pytest.fixture(scope="module")
def fixture_root() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="ef_fixture_"))
    return build(tmp / "ds")


@pytest.fixture(scope="module")
def report(fixture_root: Path) -> dict:
    adapter = LeRobotAdapter(fixture_root, decode_frames=40)
    return quality.build_report(adapter)


def test_all_twelve_rules_detected(report: dict):
    hits = report["rule_hits"]
    missing = [rule for rule in EXPECTED if rule not in hits]
    assert not missing, f"未检出的规则: {missing}；实际命中 {hits}"


@pytest.mark.parametrize("rule,episode", list(EXPECTED.items()))
def test_rule_points_at_right_episode(report: dict, rule: str, episode: int):
    eps = [f["episode"] for f in report["findings"] if f["rule"] == rule]
    assert eps, f"{rule} 未命中任何 Episode"
    assert episode in eps, f"{rule} 应命中 Episode {episode}，实际 {eps}"


def test_clean_episode_has_no_error(report: dict):
    errs = [f for f in report["findings"] if f["episode"] == 0 and f["severity"] == "error"]
    assert not errs, f"干净 Episode 报错: {errs}"


def test_metrics_have_denominators(report: dict):
    for name, m in report["metrics"].items():
        assert set(m) == {"value", "numerator", "denominator"}, name
        assert m["denominator"] is not None and m["denominator"] > 0, name


def test_split_is_deterministic(fixture_root: Path):
    adapter = LeRobotAdapter(fixture_root)
    idx = [int(m["episode_index"]) for m in adapter.episode_metas()]
    a = align.split_episodes(idx, (0.8, 0.1, 0.1), seed=7)
    b = align.split_episodes(idx, (0.8, 0.1, 0.1), seed=7)
    assert a == b
    assert sorted(a["train"] + a["val"] + a["test"]) == sorted(idx)


def test_alignment_keeps_100hz_source_intact(fixture_root: Path):
    adapter = LeRobotAdapter(fixture_root)
    ep = adapter.read_episode(0)
    before = ep.streams["action"].values.copy()
    aligned = align.align_to_camera_clock(ep, max_dt=0.05)
    assert aligned["action"]["interpolation"] == "linear"
    assert aligned["observation.gripper"]["interpolation"] == "hold"
    assert (ep.streams["action"].values == before).all(), "对齐不应修改原始数据"
