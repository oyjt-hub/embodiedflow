from __future__ import annotations

from services import resources as ressvc


def test_sample_gpus_shape():
    rows = ressvc.sample_gpus()
    assert isinstance(rows, list)
    for row in rows:
        assert {"index", "uuid", "name", "memory_total_mb"} <= set(row)
