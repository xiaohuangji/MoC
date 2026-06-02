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
Description=Run MoC Remote Control with adaptive polling

[Timer]
OnBootSec=30s
OnUnitActiveSec=30s
AccuracySec=1s
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now moc-remote-control.timer

if command -v loginctl >/dev/null 2>&1; then
	if ! loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null | grep -qi '^yes$'; then
		echo "Warning: lingering is disabled. The user timer may stop after logout."
		echo "To keep it running after logout, run: sudo loginctl enable-linger $(id -un)"
	fi
fi

echo "Installed and started: moc-remote-control.timer"
