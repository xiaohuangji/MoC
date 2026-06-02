#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/data/hyh/MoC"
UNIT_DIR="/etc/systemd/system"
USER_NAME="huyihan"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Please run this script with sudo/root."
  exit 1
fi

cat > "$UNIT_DIR/moc-remote-control.service" <<EOF
[Unit]
Description=MoC Remote Control Runner
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=$USER_NAME
Group=$USER_NAME
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/scripts/remote_control/runner.sh
EOF

cat > "$UNIT_DIR/moc-remote-control.timer" <<EOF
[Unit]
Description=Run MoC Remote Control with adaptive polling

[Timer]
OnBootSec=30s
OnUnitActiveSec=30s
AccuracySec=1s
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now moc-remote-control.timer

echo "Installed and started system timer: moc-remote-control.timer"
