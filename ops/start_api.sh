#!/bin/bash
set -eu
source "$(dirname "$0")/env.sh"
mkdir -p "$EF_DATA/metadata" "$EF_DATA/runs" "$EF_DATA/artifacts" "$EF_DATA/raw" "$EF_DATA/derived" "$EF_DATA/cache" "$EF_DATA/tmp"
cd "$PROJ"
nohup "$PY" -m uvicorn apps.api.main:app --host "$API_HOST" --port "$API_PORT" --log-level info \
  > "$EF_DATA/metadata/api.log" 2>&1 < /dev/null &
echo $! > "$EF_DATA/metadata/api.pid"
sleep 3
echo "API pid=$(cat "$EF_DATA/metadata/api.pid")  http://$API_HOST:$API_PORT/api/health"
curl -s --max-time 5 "http://$API_HOST:$API_PORT/api/health" || echo "(健康检查未就绪，看 metadata/api.log)"
