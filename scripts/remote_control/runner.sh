#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CMD_FILE="$REPO_DIR/control/commands.txt"
LOG_FILE="$REPO_DIR/logs/remote-control.log"

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG_FILE"; }

cd "$REPO_DIR"

# 更新仓库
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git fetch origin || true
  LOCAL_HEAD="$(git rev-parse HEAD)"
  REMOTE_HEAD="$(git rev-parse origin/HEAD 2>/dev/null || true)"
  if [[ -n "$REMOTE_HEAD" && "$REMOTE_HEAD" != "$LOCAL_HEAD" ]]; then
    log "Pulling updates: $LOCAL_HEAD -> $REMOTE_HEAD"
    git pull --rebase --autostash || { log "git pull failed"; exit 1; }
  else
    log "No remote updates."
  fi
fi

# 确保命令文件存在
mkdir -p "$(dirname "$CMD_FILE")"
if [[ ! -f "$CMD_FILE" ]]; then
  : > "$CMD_FILE"
  git add "$CMD_FILE" || true
  git commit -m "chore: add command file" || true
  git push || true
fi

# 无命令直接退出
if [[ ! -s "$CMD_FILE" ]]; then
  log "No commands to run."
  exit 0
fi

log "Executing commands from $CMD_FILE"

idx=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line_trimmed="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  if [[ -z "$line_trimmed" || "$line_trimmed" == \#* ]]; then
    continue
  fi

  idx=$((idx + 1))
  log "Command $idx: $line_trimmed"

  set +e
  output=$(bash -lc "$line_trimmed" 2>&1)
  status=$?
  set -e

  if [[ -n "$output" ]]; then
    while IFS= read -r out_line; do
      log "Output $idx: $out_line"
    done <<< "$output"
  fi

  log "Exit $idx: $status"

done < "$CMD_FILE"

: > "$CMD_FILE"
log "Commands cleared."

git add "$CMD_FILE" "$LOG_FILE" || true
if git diff --cached --quiet; then
  log "No changes to commit."
else
  git commit -m "chore: update remote control log" || true
  git push || log "Push failed."
fi
