#!/bin/bash
# 注意：pkill 的匹配串必须写在本文件里，不能在命令行内联（否则会匹配到自己）
source "$(dirname "$0")/env.sh"
for name in api worker web; do
  pidfile="$EF_DATA/metadata/$name.pid"
  if [ -f "$pidfile" ]; then
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null && echo "stopped $name (pid $pid)"; fi
    rm -f "$pidfile"
  fi
done
pkill -f "uvicorn apps.api.main:app" 2>/dev/null || true
pkill -f "worker.main --id" 2>/dev/null || true
pkill -f "streamlit run apps/web/app.py" 2>/dev/null || true
echo "done"
