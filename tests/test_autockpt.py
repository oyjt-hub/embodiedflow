"""训练后自动发现 checkpoint：识别常见落盘形态，取最新。"""
from __future__ import annotations

import os
import time
from pathlib import Path

from services import modeling


def _mk(path: Path, marker: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if marker == "metadata":
        (path / "_CHECKPOINT_METADATA").write_text("{}")
    else:
        (path / marker).mkdir(exist_ok=True)
        (path / marker / "w.bin").write_bytes(b"x")
    (path / "payload.txt").write_text("d")


def test_finds_latest_of_two(tmp_path: Path):
    old = tmp_path / "0"
    new = tmp_path / "1"
    _mk(old, "params")
    time.sleep(0.05)
    _mk(new, "train_state")
    found = modeling.find_latest_checkpoint(tmp_path)
    assert found == new, f"应取最新的，实际 {found}"


def test_finds_orbax_style(tmp_path: Path):
    d = tmp_path / "exp" / "step0"
    _mk(d, "metadata")
    assert modeling.find_latest_checkpoint(tmp_path / "exp") == d


def test_returns_none_when_empty(tmp_path: Path):
    (tmp_path / "junk").mkdir()
    assert modeling.find_latest_checkpoint(tmp_path) is None
