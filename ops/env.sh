#!/bin/bash
# 统一环境入口：所有 ops 脚本 source 这个文件
export PROJ="${PROJ:-/home/oyjt/workspace/embodiedflow}"
export EF_ROOT="$PROJ"
export EF_DATA="${EF_DATA:-/home/oyjt/workspace/embodiedflow_data}"
export EF_DB="${EF_DB:-$EF_DATA/metadata/platform.db}"
export PYTHONPATH="$PROJ"
export VENV="$PROJ/.venv"
export PY="$VENV/bin/python"
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-18080}"
export WEB_PORT="${WEB_PORT:-8501}"
export API_BASE="http://${API_HOST}:${API_PORT}"

export EF_API_TOKEN="${EF_API_TOKEN:-}"   # 设为非空即启用 Bearer/查询参数鉴权
export EF_API_UNAUTH=""
