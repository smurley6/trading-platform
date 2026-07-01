#!/usr/bin/env bash
# Pre-commit gate: a commit that changes architecture / the API surface / config / deployment
# MUST update MEMORY.md in the SAME commit (see MEMORY.md's maintenance convention). This checks
# only HIGH-SIGNAL files — routine edits (a data-source adapter, a component tweak) don't trip it.
# Bypass a genuine false positive with:  git commit --no-verify
set -uo pipefail

staged=$(git diff --cached --name-only --diff-filter=ACMR)

# Files whose changes almost always alter architecture, the API surface, config, or deployment.
high_signal_re='^(backend/src/trading/(main|config)\.py|backend/src/trading/interface/(app|schemas)\.py|backend/alembic/versions/|docs/DEPLOY\.md|\.github/workflows/|frontend/src/lib/(api|types)\.ts)'

hits=$(printf '%s\n' "$staged" | grep -E "$high_signal_re" || true)

if [ -n "$hits" ] && ! printf '%s\n' "$staged" | grep -qx "MEMORY.md"; then
  echo "✖ MEMORY.md was not updated, but this commit touches architecture/API/config/deploy files:"
  printf '%s\n' "$hits" | sed 's/^/    - /'
  echo
  echo "  Update MEMORY.md (add a newest-first Change log entry + edit the affected section)"
  echo "  and 'git add MEMORY.md', or bypass a false positive with:  git commit --no-verify"
  exit 1
fi
exit 0
