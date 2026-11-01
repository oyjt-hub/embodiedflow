#!/bin/bash
# P0.1 验收：提交任务 → 独立 Worker 执行 → 能查状态与日志
set -u
source "$(dirname "$0")/env.sh"
cd "$PROJ"
BASE="$API_BASE"

echo "=== [1] 健康检查 $BASE/api/health ==="
curl -sf --max-time 5 "$BASE/api/health" || { echo "FAIL: API 未就绪，看 $EF_DATA/metadata/api.log"; exit 1; }
echo

echo "=== [2] 提交 smoke 任务 ==="
RESP=$(curl -sf --max-time 10 -X POST "$BASE/api/jobs" \
  -H 'content-type: application/json' \
  -d '{"kind":"smoke","name":"p01-smoke","spec":{},"requested_gpus":0,"created_by":"smoke"}') || { echo "FAIL: 提交失败"; exit 1; }
echo "$RESP"
JOB_ID=$(echo "$RESP" | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["id"])')
echo "job_id=$JOB_ID"

echo
echo "=== [3] 等 Worker 领取并执行（最多 60s）==="
for i in $(seq 1 60); do
  ST=$(curl -sf --max-time 5 "$BASE/api/jobs/$JOB_ID" | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["status"])')
  printf '  %02ds status=%s\n' "$i" "$ST"
  case "$ST" in completed|failed|cancelled) break;; esac
  sleep 1
done

echo
echo "=== [4] 最终状态 ==="
curl -sf --max-time 5 "$BASE/api/jobs/$JOB_ID"; echo
echo "=== [5] 日志 ==="
curl -sf --max-time 5 "$BASE/api/jobs/$JOB_ID/logs?tail=50" | "$PY" -c '
import sys, json
d = json.load(sys.stdin)
print("日志文件:", d["path"])
print("\n".join(d["lines"]))
'

echo
echo "=== [6] 判定 ==="
FINAL=$(curl -sf --max-time 5 "$BASE/api/jobs/$JOB_ID" | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["status"])')
if [ "$FINAL" = "completed" ]; then
  echo "PASS: job #$JOB_ID 由独立 Worker 执行完成"
else
  echo "FAIL: 最终状态 $FINAL"; exit 2
fi
