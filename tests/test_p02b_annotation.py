"""P0.2b：FLORA 自动标注 —— 作业可复现、抽检分层、派生版本不改原始。"""
from __future__ import annotations

import uuid

from persistence.db import session_scope
from persistence.migrate import upgrade
from persistence.models import Dataset, DatasetVersion
from services import annotation as ann


def _seed() -> int:
    with session_scope() as s:
        d = Dataset(name=f"flora-ds-{uuid.uuid4().hex[:10]}", format="lerobot")
        s.add(d); s.flush()
        v = DatasetVersion(dataset_id=d.id, version="v1", episode_count=10, status="ready")
        s.add(v); s.flush()
        return v.id


def test_run_is_deterministic():
    upgrade(verbose=False)
    vid = _seed()
    with session_scope() as s:
        r1 = ann.run_annotation(s, vid, [0, 1, 2])
    with session_scope() as s:
        r2 = ann.run_annotation(s, vid, [0, 1, 2])
    with session_scope() as s:
        from persistence.models import Annotation
        from sqlalchemy import select
        a1 = [(a.episode_index, a.confidence, a.payload) for a in
              s.scalars(select(Annotation).where(Annotation.run_id == r1.id).order_by(Annotation.episode_index))]
        a2 = [(a.episode_index, a.confidence, a.payload) for a in
              s.scalars(select(Annotation).where(Annotation.run_id == r2.id).order_by(Annotation.episode_index))]
    assert a1 == a2, "同一 Episode 的 Mock 标注必须可复现"
    assert r1.model_version == "mock"


def test_sampling_stratifies_by_confidence():
    upgrade(verbose=False)
    vid = _seed()
    with session_scope() as s:
        ann.run_annotation(s, vid, list(range(10)))
    with session_scope() as s:
        plan = ann.sampling_plan(s, vid)
    assert plan["total"] == 10
    assert plan["low"] + plan["mid"] + plan["high"] == 10
    assert plan["coverage"]["denominator"] == 10
    assert plan["review_selected"] >= plan["low"], "低置信度必须全检"


def test_publish_does_not_touch_source():
    upgrade(verbose=False)
    vid = _seed()
    with session_scope() as s:
        ann.run_annotation(s, vid, [0, 1, 2])
    with session_scope() as s:
        d = ann.publish(s, vid, "v1-flora")
        assert d.status == "derived" and d.episode_count == 3
    with session_scope() as s:
        assert s.get(DatasetVersion, vid).status == "ready"
