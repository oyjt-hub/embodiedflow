"""把服务器上的训练/评测/任务日志索引进数据库（只记路径与摘要）。"""
from __future__ import annotations

import argparse
import os
import re

SKIP = ("embodiedflow_data/metadata", "node_modules", "__pycache__", ".git")
# 用较具体的词，避免 "rl" 命中 world、"test" 命中 latest 这类误判
CLASSIFY = [
    ("eval", ("eval_", "_eval", "evaluation", "benchmark")),
    ("train", ("train", "finetune", "sft_", "_sft", "lora")),
    ("task", ("/runs/job_",)),
    ("platform", ("api.log", "worker.log", "mlflow")),
]


def kind_of(path: str) -> str:
    low = path.lower()
    if "/runs/job_" in low:
        return "task"
    for k, words in CLASSIFY:
        if any(w in low for w in words):
            return k
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/oyjt/workspace")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, os.getcwd())
    from persistence.db import session_scope
    from persistence.models import LogRecord

    found = []
    for dirpath, dirnames, filenames in os.walk(a.root, followlinks=True):
        if any(s in dirpath for s in SKIP):
            dirnames[:] = []
            continue
        if dirpath.count(os.sep) - a.root.count(os.sep) > 5:
            dirnames[:] = []
            continue
        for f in filenames:
            if not f.endswith((".log", ".out", ".txt")):
                continue
            fp = os.path.join(dirpath, f)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            if sz < 1024:
                continue
            found.append((fp, sz))

    print(f"  发现 {len(found)} 个日志文件")
    if not a.apply:
        from collections import Counter
        c = Counter(kind_of(p) for p, _ in found)
        print("  按类型:", dict(c))
        return 0

    added = 0
    with session_scope() as s:
        existing = {r.source_path for r in s.query(LogRecord).all()}
        for fp, sz in found:
            if fp in existing:
                continue
            try:
                with open(fp, "r", errors="ignore") as fh:
                    head = "".join(fh.readline() for _ in range(3))[:500]
                with open(fp, "rb") as fh:
                    lines = sum(1 for _ in fh)
            except OSError:
                continue
            m = re.search(r"job_(\d+)", fp)
            s.add(LogRecord(kind=kind_of(fp), source_path=fp, size_bytes=sz,
                            line_count=lines, job_id=int(m.group(1)) if m else None,
                            head=head))
            added += 1
    print(f"  新增 {added} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
