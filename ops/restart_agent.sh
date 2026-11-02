#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${EMBODIEDFLOW_DATA_ROOT:-/home/oyjt/workspace/embodiedflow_data}"
PID_FILE="$DATA_ROOT/metadata/agent.pid"
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if ps -p "$PID" -o args= 2>/dev/null | grep -q '/apps/agent/server.py'; then
    kill "$PID"
    sleep 1
  fi
  rm -f "$PID_FILE"
fi
bash /home/oyjt/workspace/embodiedflow/ops/start_agent.sh
