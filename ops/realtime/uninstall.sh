#!/usr/bin/env bash
# Stop + remove the always-on real-time data scraper launchd agent. Leaves logs in place.
set -euo pipefail

LABEL="com.stevenmurley.trading.realtime"
UID_NUM="$(id -u)"

launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"

echo "✓ stopped + removed: $LABEL"
echo "  (logs kept at $HOME/Library/Logs/trading-realtime/)"
