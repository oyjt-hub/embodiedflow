#!/bin/bash
set -eu
source "$(dirname "$0")/env.sh"
cd "$PROJ"
export EF_API="$API_BASE"
nohup "$VENV/bin/streamlit" run apps/web/app.py --server.port "$WEB_PORT" --server.address 0.0.0.0 \
  --server.headless true > "$EF_DATA/metadata/web.log" 2>&1 < /dev/null &
echo $! > "$EF_DATA/metadata/web.pid"
sleep 4
echo "Web pid=$(cat "$EF_DATA/metadata/web.pid")  http://0.0.0.0:$WEB_PORT"
