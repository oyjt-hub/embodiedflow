#!/usr/bin/env bash
# 以 STDIO 方式启动 MCP Server，供 Codex/Claude 等客户端作为子进程调用。
# 用法（客户端配置示例）：
#   command: /home/oyjt/workspace/embodiedflow/ops/mcp_stdio.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EMBODIEDFLOW_API="${EMBODIEDFLOW_API:-http://127.0.0.1:18080}"
export EMBODIEDFLOW_PROJECT_ID="${EMBODIEDFLOW_PROJECT_ID:-1}"
exec "$ROOT/.mcp-venv/bin/python" "$ROOT/apps/mcp/server.py" --stdio
