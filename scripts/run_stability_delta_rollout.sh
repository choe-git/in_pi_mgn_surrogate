#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible name for the acceleration rollout configuration.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_stability_acceleration_rollout.sh" "$@"
