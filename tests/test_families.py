"""模型家族注册：如实标注，不把 eval_only 说成可训练。"""
from __future__ import annotations

from services import families


def test_all_families_declared():
    names = {f["name"] for f in families.all_families()}
    assert {"openpi", "holobrain", "lingbotvla"} <= names


def test_status_is_honest():
    for f in families.all_families():
        assert f["status"] in {"available", "eval_only", "missing"}
        if f["kind"] == "eval":
            assert f["status"] != "available", f"{f['name']} 只是评测配置，不能标 available"


def test_missing_weights_are_listed():
    hb = families.describe("holobrain")
    assert set(hb["weights_found"]) | set(hb["weights_missing"]) == set(hb.get("weights") or [])
