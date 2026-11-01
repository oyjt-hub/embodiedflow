#!/bin/bash
set -eu
D="$(dirname "$0")"
bash "$D/stop_all.sh"
sleep 1
bash "$D/start_api.sh"
bash "$D/start_worker.sh"
bash "$D/start_web.sh"
