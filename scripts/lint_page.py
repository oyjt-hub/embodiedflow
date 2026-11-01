"""页面静态校验：捕捉 HTTP 200 掩盖的运行时错误。

检查项：
1. onclick 调用的函数是否都已定义
2. $("id") / getElementById 引用的元素是否存在
3. fetch 路径是否在后端路由中声明（根路径 /api/* 与 /workbench/api/* 都查）
4. 反引号/花括号/圆括号是否配平
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
html = (ROOT / "apps/web/workbench.html").read_text(encoding="utf-8")
# 逐个路由文件解析：文件名里的 prefix 要拼上，否则 /compare 匹配不上 /api/benchmarks/compare
declared: set[str] = set()

# main.py 里 include_router 时的挂载前缀（如 /api）也要算进去
main_src = (ROOT / "apps/api/main.py").read_text(encoding="utf-8")
mount_prefixes = set(re.findall(
    r'include_router\([^,]+,\s*prefix="([^"]*)"', main_src))
mount_prefixes.add("")   # 也有 include_router(x.router) 无前缀的挂载
for rf in (ROOT / "apps/api/routers").glob("*.py"):
    src = rf.read_text(encoding="utf-8")
    m = re.search(r'APIRouter\(\s*prefix="([^"]*)"', src)
    prefix = m.group(1) if m else ""
    for route in re.findall(r'@router\.\w+\("([^"]+)"', src):
        for mp in mount_prefixes:
            declared.add((mp + prefix + route).replace("//", "/"))

problems: list[str] = []

defined = set(re.findall(r"(?:async\s+)?function\s+(\w+)\s*\(", html))
called = set(re.findall(r'onclick="(\w+)\(', html))
if called - defined:
    problems.append(f"未定义函数: {sorted(called - defined)}")

used = set(re.findall(r'\$\("([\w-]+)"\)', html)) | \
       set(re.findall(r'getElementById\("([\w-]+)"\)', html))
present = set(re.findall(r'id="([\w-]+)"', html))
if used - present:
    problems.append(f"缺失元素 id: {sorted(used - present)}")



def norm(p: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", p.split("?")[0])


for path in set(re.findall(r'fetch\(\s*["\']([^"\']+)', html)):
    if path.startswith(("http://", "https://")):
        continue
    # 以 "/" 结尾的是运行时拼接的动态路径（如 ".../timestamps/\" + vid），
    # 用前缀匹配判断是否有对应路由
    if path.endswith("/"):
        # 相对路径要先补上 workbench 前缀再比
        bases = [path] if path.startswith("/") else ["/workbench/" + path, "/" + path]
        if any(norm(d).startswith(norm(b)) for d in declared for b in bases):
            continue
        problems.append(f"动态路径无对应路由前缀: {path}")
        continue
    # 相对路径（api/gpu）在 /workbench/ 与 /ef/workbench/ 下都解析为 workbench 路由
    candidates = [path] if path.startswith("/") else ["/workbench/" + path, "/" + path]
    if not any(norm(d) == norm(c) for d in declared for c in candidates):
        problems.append(f"fetch 路径无对应路由: {path}")

# 页内锚点：href="#x" 必须能找到 id="x"
anchors = set(re.findall(r'href="#([\w-]+)"', html))
if anchors - present:
    problems.append(f"锚点无对应元素: {sorted(anchors - present)}")

if html.count("`") % 2:
    problems.append("反引号数量为奇数（模板字符串可能未闭合）")
if html.count("{") != html.count("}"):
    problems.append("花括号不配平")
if html.count("(") != html.count(")"):
    problems.append("圆括号不配平")

print(f"  函数引用 {len(called)} 个 · id 引用 {len(used)} 个 · 路由 {len(declared)} 条")
if problems:
    print("  ❌ 发现问题:")
    for p in problems:
        print("     -", p)
    sys.exit(1)
print("  ✅ 静态校验通过")
