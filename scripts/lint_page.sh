#!/bin/bash
set -eu
cd /home/oyjt/workspace/embodiedflow
source ops/env.sh

"$PY" - <<'PY'
import re
from pathlib import Path

html = Path("apps/web/workbench.html").read_text(encoding="utf-8")
router = Path("apps/api/routers/workbench.py").read_text(encoding="utf-8")

print("=== 1) onclick 引用的函数是否都已定义 ===")
defined = set(re.findall(r"(?:async\s+)?function\s+(\w+)\s*\(", html))
called = set(re.findall(r'onclick="(\w+)\(', html))
missing = called - defined
print(f"  调用的函数: {sorted(called)}")
print(f"  缺失定义  : {sorted(missing) if missing else '无'}")

print()
print("=== 2) $(\"id\") 引用的元素是否存在于 HTML ===")
used = set(re.findall(r'\$\("([\w-]+)"\)', html)) | set(re.findall(r'getElementById\("([\w-]+)"\)', html))
present = set(re.findall(r'id="([\w-]+)"', html))
missing_ids = sorted(used - present)
print(f"  引用 id 数: {len(used)}   缺失: {missing_ids if missing_ids else '无'}")

print()
print("=== 3) fetch 路径是否在路由中声明 ===")
paths = set(re.findall(r'fetch\(\s*["\']([^"\']+)', html))
declared = set(re.findall(r'@router\.\w+\("([^"]+)"', router))
bad = []
for p in sorted(paths):
    if p.startswith("/workbench/"):
        # 去掉 /workbench 前缀后与 router 声明比较
        rest = p[len("/workbench"):]
    else:
        rest = p
    rest = rest.split("?")[0]
    # 把 {xx} 占位符归一化
    norm = re.sub(r"\{[^}]+\}", "{}", rest)
    ok = any(re.sub(r"\{[^}]+\}", "{}", d) == norm for d in declared)
    if not ok:
        bad.append(p)
print(f"  fetch 路径数: {len(paths)}   未匹配: {bad if bad else '无'}")

print()
print("=== 4) 未闭合的模板字符串/括号粗检 ===")
for ch_pairs in [("`", "反引号")]:
    n = html.count(ch_pairs[0])
    print(f"  {ch_pairs[1]}数量: {n}  {'奇数（可能未闭合）' if n % 2 else '偶数 OK'}")
print(f"  花括号 {{ }}: {html.count('{')} / {html.count('}')}")
print(f"  圆括号 ( ): {html.count('(')} / {html.count(')')}")
PY
