#!/bin/bash
# EmbodiedFlow 系统自检：一条命令给出"能不能用"的完整证据
set -u
source /home/oyjt/workspace/embodiedflow/ops/env.sh
cd "$PROJ"
ok=0; bad=0
chk() { # chk 名称 命令
  if eval "$2" >/dev/null 2>&1; then printf '  ✅ %-34s\n' "$1"; ok=$((ok+1));
  else printf '  ❌ %-34s\n' "$1"; bad=$((bad+1)); fi
}

echo "=== 1. 服务进程 ==="
chk "API  (:18080)"            "curl -sf -m 5 http://127.0.0.1:18080/api/health"
chk "Web  (:8501)"             "curl -sf -o /dev/null -m 5 http://127.0.0.1:8501/"
chk "MLflow (:15000)"          "curl -sf -m 5 http://127.0.0.1:15000/health"
chk "Worker 进程"              "pgrep -f 'worker.main --id'"

echo
echo "=== 2. 核心接口 ==="
for p in summary datasets jobs models devices evaluations families annotation/versions; do
  chk "GET /api/$p" "curl -sf -m 8 http://127.0.0.1:18080/api/$p"
done
chk "GET /api/edge/status" "curl -sf -m 8 http://127.0.0.1:18080/api/edge/status"

echo
echo "=== 3. 数据库 ==="
chk "schema_version >= 9" "$PY -c \"import sqlite3;c=sqlite3.connect('$EF_DATA/metadata/platform.db');assert c.execute('select max(version) from schema_version').fetchone()[0]>=9\""
chk "表数量 >= 15" "$PY -c \"import sqlite3;c=sqlite3.connect('$EF_DATA/metadata/platform.db');assert len(c.execute(\\\"select name from sqlite_master where type='table'\\\").fetchall())>=15\""

echo
echo
echo "=== 3.5 页面静态校验 ==="
chk "workbench 页面无缺失元素/函数" "$PY scripts/lint_page.py"

echo "=== 4. 测试 ==="
chk "pytest 全量通过" "env -u EF_ROOT -u EF_DATA -u EF_DB -u EF_API PYTHONPATH=. .venv/bin/python -m pytest tests -q"

echo
echo "=== 5. 产物 ==="
chk "文档 >= 12 份" "[ \$(ls docs/*.md | wc -l) -ge 12 ]"
chk "基准报告存在" "ls '$EF_DATA'/artifacts/benchmarks/*.json"
chk "失败回放存在" "ls '$EF_DATA'/artifacts/benchmarks/replay/*.jsonl"

echo
echo "=== 汇总 ==="
printf "  通过 %d 项，失败 %d 项\n" "$ok" "$bad"
echo
echo "=== 数据概况 ==="
curl -s -m 5 http://127.0.0.1:18080/api/summary
echo
exit $([ "$bad" -eq 0 ] && echo 0 || echo 1)
