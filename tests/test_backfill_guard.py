"""回填脚本必须尊重已有清单：有 episode_indices 的版本一律跳过。"""
from __future__ import annotations

import json
import sqlite3


def test_versions_with_indices_are_skipped(tmp_path):
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("create table dataset_versions (id integer primary key, version text,"
                " status text, episode_count int, episode_indices text, manifest_uri text)")
    con.execute("insert into dataset_versions values (1,'v1','ready',3,'[0,1,2]','/nope')")
    con.commit()
    con.close()

    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "scripts/backfill_episodes.py", "--db", str(db)],
        capture_output=True, text=True)
    assert "跳过" in out.stdout
    assert "计划写入" not in out.stdout

    con = sqlite3.connect(db)
    assert con.execute("select episode_indices from dataset_versions where id=1").fetchone()[0] == "[0,1,2]"
    con.close()
