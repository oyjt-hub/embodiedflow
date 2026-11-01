"""P1.4：只读助手的权限边界与引用完整性。"""
from __future__ import annotations

from persistence.db import session_scope
from persistence.migrate import upgrade
from persistence.models import Dataset, DatasetVersion, QualityReport
from services import assistant


def test_write_intent_is_rejected():
    upgrade(verbose=False)
    with session_scope() as s:
        r = assistant.ask(s, "帮我启动一个训练任务")
        assert r["allowed"] is False
        assert "只读" in r["answer"]


def test_quality_intent_has_citations():
    upgrade(verbose=False)
    with session_scope() as s:
        d = Dataset(name="assistant-ds", format="lerobot")
        s.add(d); s.flush()
        v = DatasetVersion(dataset_id=d.id, version="v1", episode_count=3, status="ready")
        s.add(v); s.flush()
        s.add(QualityReport(dataset_version_id=v.id, episodes=3, finding_count=2))
    with session_scope() as s:
        r = assistant.ask(s, "哪些数据有质检异常")
        assert r["allowed"] is True
        assert r["citations"], "必须带引用"
