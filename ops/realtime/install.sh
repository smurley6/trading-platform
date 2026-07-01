#!/usr/bin/env bash
# Install + (re)start the always-on real-time data scraper as a macOS launchd user agent.
#
# Idempotent: re-running re-renders the plist and bounces the service. Override the cadence
# or mode inline, e.g.:  INGEST_INTERVAL_SECONDS=60 CYCLE_TRADE_MODE=manual ./install.sh
#
# CYCLE_TRADE_MODE defaults to 'manual' (collect-only — places NO orders). Set it to 'auto'
# ONLY if you want the always-on loop to place paper trades itself.
set -euo pipefail

LABEL="com.stevenmurley.trading.realtime"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/backend/.venv/bin/python"
TEMPLATE="$REPO/ops/realtime/$LABEL.plist.template"
LOGDIR="$HOME/Library/Logs/trading-realtime"
AGENTS="$HOME/Library/LaunchAgents"
PLIST_DST="$AGENTS/$LABEL.plist"
UID_NUM="$(id -u)"

INTERVAL="${INGEST_INTERVAL_SECONDS:-120}"
MODE="${CYCLE_TRADE_MODE:-manual}"

# --- preflight ----------------------------------------------------------------------------
[ -x "$PY" ]                || { echo "ERROR: venv python not found/executable at $PY" >&2; exit 1; }
[ -f "$REPO/backend/.env" ] || { echo "ERROR: $REPO/backend/.env missing (needs DATABASE_URL + keys)" >&2; exit 1; }
[ -f "$TEMPLATE" ]          || { echo "ERROR: template missing at $TEMPLATE" >&2; exit 1; }

mkdir -p "$LOGDIR" "$AGENTS"

# --- render plist from template -----------------------------------------------------------
sed -e "s|{{LABEL}}|$LABEL|g" \
    -e "s|{{PY}}|$PY|g" \
    -e "s|{{WORKDIR}}|$REPO/backend|g" \
    -e "s|{{INTERVAL}}|$INTERVAL|g" \
    -e "s|{{MODE}}|$MODE|g" \
    -e "s|{{LOGDIR}}|$LOGDIR|g" \
    "$TEMPLATE" > "$PLIST_DST"

# --- (re)load via modern launchctl bootstrap ----------------------------------------------
launchctl bootout   "gui/$UID_NUM/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST_DST"
launchctl enable    "gui/$UID_NUM/$LABEL"
launchctl kickstart -k "gui/$UID_NUM/$LABEL"

echo "✓ installed + started: $LABEL"
echo "    poll interval : ${INTERVAL}s"
echo "    trade mode    : ${MODE}  (manual = places NO orders)"
echo "    backend       : http://127.0.0.1:8000"
echo "    logs          : $LOGDIR/stdout.log"
echo "                    $LOGDIR/stderr.log"
echo
echo "  tail the scraper:  tail -f \"$LOGDIR/stdout.log\""
echo "  status         :  launchctl print gui/$UID_NUM/$LABEL | grep -E 'state|pid'"
echo "  view dashboard :  (cd \"$REPO/frontend\" && npm run dev)  ->  http://localhost:5173"
echo "  stop/remove    :  $REPO/ops/realtime/uninstall.sh"
