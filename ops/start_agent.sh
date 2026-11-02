#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${EMBODIEDFLOW_DATA_ROOT:-/home/oyjt/workspace/embodiedflow_data}"
AGENT_ROOT="$DATA_ROOT/agent"
ENV_FILE="$AGENT_ROOT/agent.env"
mkdir -p "$AGENT_ROOT/workspace" "$AGENT_ROOT/dsh_home" "$DATA_ROOT/metadata" "$DATA_ROOT/logs"
if [[ -f "$ENV_FILE" ]]; then
  chmod 600 "$ENV_FILE"
  set -a
  . "$ENV_FILE"
  set +a
fi
PID_FILE="$DATA_ROOT/metadata/agent.pid"
LOG_FILE="$DATA_ROOT/logs/agent.log"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Agent already running: pid $(cat "$PID_FILE")"
  exit 0
fi

nohup env \
  EMBODIEDFLOW_DATA_ROOT="$DATA_ROOT" \
  EMBODIEDFLOW_DSH_HOME="$AGENT_ROOT/dsh_home" \
  EMBODIEDFLOW_AGENT_WORKSPACE="$AGENT_ROOT/workspace" \
  EMBODIEDFLOW_AGENT_HOST="${EMBODIEDFLOW_AGENT_HOST:-0.0.0.0}" \
  EMBODIEDFLOW_AGENT_PORT="${EMBODIEDFLOW_AGENT_PORT:-18082}" \
  DSH_MODEL="${DSH_MODEL:-deepseek-v4-flash}" \
  DSH_SYSTEM_PROMPT="你是 EmbodiedFlow 具身智能研发助手。关于数据集、训练任务、GPU、实验、评测与模型信息时，优先调用 embodiedflow MCP 工具并引用真实 ID。只读工具不能执行训练或部署，不要声称已提交这些操作。" \
  "$ROOT/.agent-venv/bin/python" "$ROOT/apps/agent/server.py" >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
sleep 1
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Agent started: http://172.19.5.252:18082/"
  if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then echo "Set DEEPSEEK_API_KEY in $ENV_FILE to enable chat"; fi
else
  cat "$LOG_FILE"
  exit 1
fi
