"""把服务器上现有的 LeRobot 数据集全部登记进平台数据库。

- 用 meta/info.json 判定 LeRobot 结构
- 名称取相对路径（可读、可追溯）
- 已存在的（同路径）跳过，不重复建版本
- Episode 明细只给 <=200 ep 的数据集建行，超大的只记数量（避免几十万行）
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOTS = ["/home/oyjt/workspace", "/home/oyjt/.cache/huggingface/hub"]
SKIP_PARTS = ("embodiedflow_data", ".git", "__pycache__", "node_modules")


def find_datasets() -> list[str]:
    out = []
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            if any(s in dirpath for s in SKIP_PARTS):
                dirnames[:] = []
                continue
            if dirpath.count(os.sep) - root.count(os.sep) > 6:
                dirnames[:] = []
                continue
            if "info.json" in filenames and os.path.basename(dirpath) == "meta":
                out.append(os.path.dirname(dirpath))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from persistence.db import session_scope
    from persistence.models import Dataset, DatasetVersion, Episode

    paths = find_datasets()
    print(f"  发现 {len(paths)} 个 LeRobot 数据集")

    added_ds = added_ver = added_ep = 0
    for p in paths:
        info_p = os.path.join(p, "meta", "info.json")
        try:
            info = json.load(open(info_p, encoding="utf-8"))
        except Exception:
            print(f"  跳过（info.json 读不了）: {p}")
            continue
        n_ep = int(info.get("total_episodes") or 0)
        fps = info.get("fps")
        name = p.replace("/home/oyjt/workspace/", "").replace("/", "__")[:120]
        total = 0
        for dp, _, fs in os.walk(p):
            for f in fs:
                try:
                    total += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
        if not a.apply:
            continue
        with session_scope() as s:
            ds = s.query(Dataset).filter(Dataset.name == name).first()
            if ds is None:
                ds = Dataset(name=name, format="lerobot", owner="import")
                s.add(ds)
                s.flush()
                added_ds += 1
            exists = (s.query(DatasetVersion)
                      .filter(DatasetVersion.dataset_id == ds.id,
                              DatasetVersion.version == "v1").first())
            if exists:
                continue
            idx = list(range(n_ep)) if 0 < n_ep <= 200 else []
            v = DatasetVersion(
                dataset_id=ds.id, version="v1", status="ready",
                episode_count=n_ep, total_bytes=total,
                manifest_uri=info_p,
                episode_indices=json.dumps(idx) if idx else None)
            s.add(v)
            s.flush()
            added_ver += 1
            if idx:
                for i in idx:
                    s.add(Episode(dataset_version_id=v.id, episode_index=i, fps=fps))
                added_ep += len(idx)

    if a.apply:
        print(f"  新增数据集 {added_ds}，版本 {added_ver}，Episode 行 {added_ep}")
    else:
        print("  [dry-run] 加 --apply 才写库")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
