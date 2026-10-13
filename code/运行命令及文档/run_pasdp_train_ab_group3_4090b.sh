#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/hy/dp_pro6000
exec bash "$ROOT/run_pasdp_train_remaining9_seed42.sh" group3 "$@"
