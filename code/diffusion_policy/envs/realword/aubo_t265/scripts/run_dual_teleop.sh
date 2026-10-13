#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)"
TELEOP_PYTHON="${TELEOP_PYTHON:-python}"
cd -- "$PROJECT_ROOT"
exec "$TELEOP_PYTHON" "$PROJECT_ROOT/demo_dual_teleop_aubo.py" "$@"
