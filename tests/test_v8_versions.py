"""数据版本必须自带"包含哪些 Episode"的权威清单。"""
from __future__ import annotations

import json

from persistence.db import session_scope
from persistence.migrate import upgrade


def _seed(version: str, n: int, status: str = "ready"):
    from persistence.models import Dataset, DatasetVersion

    with session_scope() as s:
        d = Dataset(name=f"v8-{version}-{id(object())}", format="lerobot")
        s.add(d)
        s.flush()
        v = DatasetVersion(dataset_id=d.id, version=version, status=status,
                           episode_count=n, episode_indices=json.dumps(list(range(n))))
        s.add(v)
        s.flush()
        return v.id


def test_episode_indices_matches_count():
    upgrade(verbose=False)
    vid = _seed("v1", 5)
    from persistence.models import DatasetVersion

    with session_scope() as s:
        v = s.get(DatasetVersion, vid)
        idx = json.loads(v.episode_indices)
        assert len(idx) == v.episode_count, "清单长度必须等于声明数量"
        assert idx == sorted(idx)


def test_derived_version_can_carry_subset():
    upgrade(verbose=False)
    vid = _seed("derived", 3, status="derived")
    from persistence.models import DatasetVersion

    with session_scope() as s:
        v = s.get(DatasetVersion, vid)
        assert v.status == "derived"
        assert len(json.loads(v.episode_indices)) == 3
