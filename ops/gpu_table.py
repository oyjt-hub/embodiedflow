"""从 stdin 读 /api/resources/gpus 的 JSON，打印成表格。"""
import json
import sys

rows = json.load(sys.stdin)
print(f"  共 {len(rows)} 张卡")
for r in rows:
    idx = r["index"]
    name = r["name"][:24]
    used = r["memory_used_mb"]
    tot = r["memory_total_mb"]
    util = r["utilization"]
    st = r["state"]
    print(f"  GPU{idx}  {name:24s} {used:>6}/{tot} MiB  util={util:>3}%  state={st}")
free = sum(1 for r in rows if r["state"] == "free")
print(f"  可调度: {free} 张")
