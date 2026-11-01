"""按版本回填 Episode 元数据（帧数/时长）。

安全约束（这些是为了避免再次污染数据）：
1. 只处理 episode_indices 为空的版本，已有清单的跳过
2. 写入数量不超过 dataset_versions.episode_count
3. 默认 dry-run，加 --apply 才真正写库
4. 不修改已有行的 episode 归属，只补缺的行 / 补帧数
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    con = sqlite3.connect(a.db)
    con.row_factory = sqlite3.Row

    versions = con.execute(
        "select id, version, status, episode_count, episode_indices, manifest_uri "
        "from dataset_versions").fetchall()

    total_planned = 0
    for v in versions:
        listed = v["episode_indices"]
        if listed and json.loads(listed):
            print(f"  版本 {v['id']} 已有清单，跳过")
            continue

        mf = v["manifest_uri"] or ""
        if not mf.endswith("meta/info.json"):
            print(f"  版本 {v['id']} 无 manifest，跳过")
            continue
        root = os.path.dirname(os.path.dirname(mf))
        info_p = os.path.join(root, "meta", "info.json")
        if not os.path.exists(info_p):
            print(f"  版本 {v['id']} manifest 文件缺失，跳过")
            continue

        info = json.load(open(info_p))
        fps = float(info.get("fps") or 30)
        files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
        if not files:
            print(f"  版本 {v['id']} 无 parquet，跳过")
            continue

        import pyarrow.parquet as pq
        counts: dict[int, int] = {}
        for f in files:
            t = pq.read_table(f, columns=["episode_index"]).to_pydict()
            for e in t["episode_index"]:
                counts[int(e)] = counts.get(int(e), 0) + 1

        cap = v["episode_count"] or len(counts)
        picked = sorted(counts)[:cap]          # 上界保护
        print(f"  版本 {v['id']}: 计划写入 {len(picked)} 个（上界 {cap}）")
        total_planned += len(picked)

        if not a.apply:
            continue

        for ep in picked:
            n = counts[ep]
            dur = n / fps if fps else None
            row = con.execute(
                "select id from episodes where dataset_version_id=? and episode_index=?",
                (v["id"], ep)).fetchone()
            if row:
                con.execute("update episodes set n_frames=?, duration_s=?, fps=? where id=?",
                            (n, dur, fps, row["id"]))
            else:
                con.execute(
                    "insert into episodes (dataset_version_id, episode_index, n_frames,"
                    " duration_s, fps) values (?,?,?,?,?)", (v["id"], ep, n, dur, fps))
        con.execute("update dataset_versions set episode_indices=? where id=?",
                    (json.dumps(picked), v["id"]))

    if a.apply:
        con.commit()
        print(f"  已提交，共处理 {total_planned} 个 Episode")
    else:
        print(f"  [dry-run] 共 {total_planned} 个，加 --apply 才写库")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
