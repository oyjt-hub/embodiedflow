#!/bin/bash
set -eu
source "$(dirname "$0")/env.sh"
mkdir -p "$EF_DATA/artifacts/mlflow" "$EF_DATA/metadata"
cd "$PROJ"
if pgrep -f "mlflow server" >/dev/null 2>&1; then
  echo "mlflow 已在运行"; exit 0
fi
nohup "$VENV/bin/mlflow" server \
  --host 127.0.0.1 --port 15000 \
  --backend-store-uri "sqlite:///$EF_DATA/metadata/mlflow.db" \
  --default-artifact-root "$EF_DATA/artifacts/mlflow" \
  > "$EF_DATA/metadata/mlflow.log" 2>&1 < /dev/null &
echo $! > "$EF_DATA/metadata/mlflow.pid"
sleep 6
curl -sf --max-time 5 http://127.0.0.1:15000/health && echo "  mlflow OK http://127.0.0.1:15000" || { echo "  启动失败，看 $EF_DATA/metadata/mlflow.log"; tail -10 "$EF_DATA/metadata/mlflow.log"; }
