"""目录/文件摘要都必须是内容相关的，不能是空串哈希。"""
from __future__ import annotations

from services.modeling import tree_digest

EMPTY_SHA = "e3b0c44298fc1c14"


def test_single_file_digest_not_empty(tmp_path):
    f = tmp_path / "checkpoint.pt"
    f.write_bytes(b"weights" * 100)
    digest, files, total = tree_digest(f)
    assert files == 1
    assert total == 700
    assert not digest.startswith(EMPTY_SHA), "单文件不应得到空串摘要"


def test_different_content_different_digest(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbb")
    assert tree_digest(a)[0] != tree_digest(b)[0]


def test_directory_digest_stable(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "params.bin").write_bytes(b"x" * 10)
    assert tree_digest(d)[0] == tree_digest(d)[0]
