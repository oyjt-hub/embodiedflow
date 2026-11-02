#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${EMBODIEDFLOW_DATA_ROOT:-/home/oyjt/workspace/embodiedflow_data}"
mkdir -p "$DATA_ROOT/metadata" "$DATA_ROOT/logs"
PID_FILE="$DATA_ROOT/metadata/mcp.pid"
LOG_FILE="$DATA_ROOT/logs/mcp.log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "MCP already running: pid $(cat "$PID_FILE")"
  exit 0
fi

nohup env \
  EMBODIEDFLOW_API="${EMBODIEDFLOW_API:-http://127.0.0.1:18080}" \
  EMBODIEDFLOW_PROJECT_ID="${EMBODIEDFLOW_PROJECT_ID:-1}" \
  EMBODIEDFLOW_MCP_HOST="${EMBODIEDFLOW_MCP_HOST:-0.0.0.0}" \
  EMBODIEDFLOW_MCP_PORT="${EMBODIEDFLOW_MCP_PORT:-18081}" \
  "$ROOT/.mcp-venv/bin/python" "$ROOT/apps/mcp/server.py" >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
sleep 1
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "MCP started: pid $(cat "$PID_FILE"), http://0.0.0.0:18081/mcp"
else
  cat "$LOG_FILE"
  exit 1
fi
