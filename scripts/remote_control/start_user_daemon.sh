#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="$REPO_DIR/.state"
PID_FILE="$STATE_DIR/user-daemon.pid"
LOG_FILE="$REPO_DIR/logs/remote-control.user-daemon.log"
DAEMON="$REPO_DIR/scripts/remote_control/user_daemon.sh"

mkdir -p "$STATE_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "User daemon already running: $(cat "$PID_FILE")"
  exit 0
fi

nohup "$DAEMON" >/dev/null 2>&1 &
echo $! > "$PID_FILE"

echo "Started user daemon: $(cat "$PID_FILE")"
echo "Log: $LOG_FILE"
