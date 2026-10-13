#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/hy/dp_pro6000
exec bash "$ROOT/run_pasdp_train_pending_two_4090.sh" b_flow_gpu1 "$@"
