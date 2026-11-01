#!/bin/bash
set -eu
source "$(dirname "$0")/env.sh"
cd "$PROJ"
nohup "$PY" -m worker.main --id "worker-$(hostname)-0" --poll 2.0 \
  > "$EF_DATA/metadata/worker.log" 2>&1 < /dev/null &
echo $! > "$EF_DATA/metadata/worker.pid"
sleep 2
echo "Worker pid=$(cat "$EF_DATA/metadata/worker.pid")  日志 $EF_DATA/metadata/worker.log"
tail -5 "$EF_DATA/metadata/worker.log" || true
