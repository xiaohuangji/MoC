#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"

mkdir -p "$UNIT_DIR"

cat > "$UNIT_DIR/moc-remote-control.service" <<EOF
[Unit]
Description=MoC Remote Control Runner
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/scripts/remote_control/runner.sh
EOF

cat > "$UNIT_DIR/moc-remote-control.timer" <<EOF
[Unit]
Description=Run MoC Remote Control every 5 minutes

[Timer]
OnBootSec=1min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now moc-remote-control.timer

echo "Installed and started: moc-remote-control.timer"
