#!/bin/bash
source /home/oyjt/workspace/embodiedflow/ops/env.sh
curl -sf --max-time 10 "$API_BASE/api/resources/gpus" | "$PY" "$PROJ/ops/gpu_table.py"
