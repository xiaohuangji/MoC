#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$REPO_DIR/scripts/remote_control/runner.sh"
LOG_FILE="$REPO_DIR/logs/remote-control.user-daemon.log"

mkdir -p "$REPO_DIR/.state"

trap 'echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Received signal, exiting." >> "$LOG_FILE"; exit 0' INT TERM HUP

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] User daemon started." >> "$LOG_FILE"

while true; do
  "$RUNNER" >> "$LOG_FILE" 2>&1 || true
  sleep 30
done
