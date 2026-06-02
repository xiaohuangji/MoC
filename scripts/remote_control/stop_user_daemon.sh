#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="$REPO_DIR/.state"
PID_FILE="$STATE_DIR/user-daemon.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No pid file found."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Stopped user daemon: $PID"
else
  echo "Daemon not running: $PID"
fi

rm -f "$PID_FILE"
