"""P0.5 数值测试：期望值全部可手算。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from services import evaluation


@pytest.fixture()
def logfile(tmp_path: Path) -> Path:
    """5 步、1 秒一步；参考是线性斜坡（加速度恒为 0）。"""
    p = tmp_path / "ep0.jsonl"
    rows = []
    lat = [10, 20, 30, 40, 50]
    for i in range(5):
        rows.append({
            "t": float(i),
            "pred": [float(i), float(i)],
            "ref": [2.0 * i, 2.0 * i],
            "latency_ms": lat[i],
        })
    rows += [{"event": "success"}, {"event": "intervention"}]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_hand_computable_numbers(logfile: Path):
    m = evaluation.evaluate(evaluation.parse_log(logfile))["metrics"]

    # 时长 4 s / 5 步
    assert m["duration_s"] == 4.0
    assert m["throughput_hz"] == pytest.approx(1.25)

    # 每步每关节误差 0,1,2,3,4 → MAE=2.0；RMSE=sqrt(30/5)=sqrt(6)
    assert m["mae"] == pytest.approx(2.0, abs=1e-6)
    assert m["rmse"] == pytest.approx(6 ** 0.5, abs=1e-6)

    # 延迟 [10..50]：p50=30；p95 位置 0.95*4=3.8 → 40+0.8*10=48
    assert m["latency_p50_ms"] == pytest.approx(30.0)
    assert m["latency_p95_ms"] == pytest.approx(48.0)

    # 线性斜坡：速度恒定（方差 0），加速度 0（方差 0），Jerk 0
    assert m["speed_variance"] == pytest.approx(0.0, abs=1e-9)
    assert m["accel_variance"] == pytest.approx(0.0, abs=1e-9)
    assert m["jerk_mean"] == pytest.approx(0.0, abs=1e-9)

    assert m["outcome"] == "success" and m["success"] is True
    assert m["interventions"] == 1


def test_nonzero_accel_and_jerk(tmp_path: Path):
    """二次曲线：加速度恒定非零，Jerk 仍为 0。"""
    p = tmp_path / "quad.jsonl"
    rows = [{"t": float(i), "pred": [float(i ** 2)], "ref": [float(i ** 2)],
             "latency_ms": 1} for i in range(5)]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    m = evaluation.evaluate(evaluation.parse_log(p))["metrics"]
    assert m["accel_variance"] == pytest.approx(0.0, abs=1e-6)   # 恒定加速度 → 方差 0
    assert m["jerk_mean"] == pytest.approx(0.0, abs=1e-6)
    assert m["mae"] == pytest.approx(0.0)


def test_missing_reference_is_na(tmp_path: Path):
    p = tmp_path / "noref.jsonl"
    p.write_text(json.dumps({"t": 0.0, "pred": [0.1], "latency_ms": 5}), encoding="utf-8")
    rep = evaluation.evaluate(evaluation.parse_log(p))
    assert rep["metrics"]["mae"] is None
    assert "N/A" in rep["metrics"]["accuracy_source"]


def test_summary_coverage(logfile: Path, tmp_path: Path):
    a = evaluation.evaluate(evaluation.parse_log(logfile))
    b_log = tmp_path / "b.jsonl"
    b_log.write_text(json.dumps({"t": 0.0, "pred": [0.0], "latency_ms": 1}), encoding="utf-8")
    b = evaluation.evaluate(evaluation.parse_log(b_log))
    s = evaluation.summarize([a, b])
    assert s["episodes"] == 2 and s["outcome_labeled"] == 1
    assert s["outcome_coverage"] == 0.5
    assert s["success"] == {"value": 1.0, "numerator": 1, "denominator": 1}
    assert s["unlabeled"] == 1


def test_csv_export_has_header(logfile: Path):
    rep = evaluation.evaluate(evaluation.parse_log(logfile))
    assert evaluation.to_csv([rep]).splitlines()[0].startswith("episode,duration_s")
