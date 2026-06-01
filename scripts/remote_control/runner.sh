#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CMD_FILE="$REPO_DIR/control/commands.txt"
LOG_FILE="$REPO_DIR/logs/remote-control.log"
STATE_DIR="$REPO_DIR/.state"
STATE_FILE="$STATE_DIR/remote-control.env"

DEFAULT_INTERVAL=30
MAX_INTERVAL=300

mkdir -p "$STATE_DIR"

load_state() {
  NEXT_INTERVAL_SECONDS=$DEFAULT_INTERVAL
  NEXT_CHECK_AT=0
  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE" || true
    NEXT_INTERVAL_SECONDS="${NEXT_INTERVAL_SECONDS:-$DEFAULT_INTERVAL}"
    NEXT_CHECK_AT="${NEXT_CHECK_AT:-0}"
  fi
}

save_state() {
  cat > "$STATE_FILE" <<EOF
NEXT_INTERVAL_SECONDS=$NEXT_INTERVAL_SECONDS
NEXT_CHECK_AT=$NEXT_CHECK_AT
EOF
}

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG_FILE"; }

cd "$REPO_DIR"
load_state

NOW_TS="$(date +%s)"
if [[ "$NEXT_CHECK_AT" =~ ^[0-9]+$ ]] && (( NOW_TS < NEXT_CHECK_AT )); then
  NEXT_RUN_HUMAN="$(date -u -d "@$NEXT_CHECK_AT" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || true)"
  if [[ -n "$NEXT_RUN_HUMAN" ]]; then
    log "Not due yet; next check at $NEXT_RUN_HUMAN."
  else
    log "Not due yet; next check is scheduled in ${NEXT_INTERVAL_SECONDS}s."
  fi
  exit 0
fi

# 更新仓库
REMOTE_UPDATED=0
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git fetch origin || true
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || true)"
  if [[ -z "$UPSTREAM" ]]; then
    UPSTREAM="origin/$BRANCH"
  fi
  LOCAL_HEAD="$(git rev-parse HEAD)"
  REMOTE_HEAD="$(git rev-parse "$UPSTREAM" 2>/dev/null || true)"
  if [[ -n "$REMOTE_HEAD" && "$REMOTE_HEAD" != "$LOCAL_HEAD" ]]; then
    log "Pulling updates from $UPSTREAM: $LOCAL_HEAD -> $REMOTE_HEAD"
    git pull --rebase --autostash origin "$BRANCH" || { log "git pull failed"; exit 1; }
    REMOTE_UPDATED=1
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
  if [[ "$REMOTE_UPDATED" -eq 0 ]]; then
    case "$NEXT_INTERVAL_SECONDS" in
      30) NEXT_INTERVAL_SECONDS=60 ;;
      60) NEXT_INTERVAL_SECONDS=120 ;;
      120) NEXT_INTERVAL_SECONDS=180 ;;
      180) NEXT_INTERVAL_SECONDS=240 ;;
      240) NEXT_INTERVAL_SECONDS=300 ;;
      *) NEXT_INTERVAL_SECONDS=$MAX_INTERVAL ;;
    esac
  else
    NEXT_INTERVAL_SECONDS=$DEFAULT_INTERVAL
  fi
  NEXT_CHECK_AT=$(( NOW_TS + NEXT_INTERVAL_SECONDS ))
  save_state
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

NEXT_INTERVAL_SECONDS=$DEFAULT_INTERVAL
NEXT_CHECK_AT=$(( NOW_TS + NEXT_INTERVAL_SECONDS ))
save_state
