#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CMD_FILE="$REPO_DIR/control/commands.txt"

cd "$REPO_DIR"

# 追加一个安全测试命令
printf "echo 'remote test: %s'\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$CMD_FILE"

git add "$CMD_FILE"
git commit -m "test: add remote command" || true
git push || true

# 立即运行一次执行器
"$REPO_DIR/scripts/remote_control/runner.sh"
