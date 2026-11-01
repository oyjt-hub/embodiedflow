#!/bin/bash
source "$(dirname "$0")/env.sh"
echo "=== 进程 ==="
pgrep -af "uvicorn apps.api.main:app" || echo "  api: 未运行"
pgrep -af "worker.main --id" || echo "  worker: 未运行"
pgrep -af "streamlit run apps/web/app.py" || echo "  web: 未运行"
echo
echo "=== 健康 ==="
curl -s --max-time 5 "http://$API_HOST:$API_PORT/api/health" | head -c 600; echo
echo
echo "=== 最近任务 ==="
curl -s --max-time 5 "http://$API_HOST:$API_PORT/api/jobs?limit=5" | head -c 800; echo
