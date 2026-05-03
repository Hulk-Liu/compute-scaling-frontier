#!/usr/bin/env bash
# verify_setup.sh — Day 0 sanity check for the three Red Hat OSS libraries.
#
# Usage:
#   ./scripts/verify_setup.sh           # imports only (no network calls)
#   ./scripts/verify_setup.sh --live    # also runs tier-2 calls against OpenAI ($)
#
# Exit code is the number of failed library smoke tests.

set -u  # don't `set -e` — we want to run all three even if one fails.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LIVE_FLAG=""
if [[ "${1:-}" == "--live" ]]; then
    LIVE_FLAG="--live"
    echo "Running in LIVE mode — will call OpenAI API."
else
    echo "Running in IMPORT-ONLY mode. Pass --live to also test API calls."
fi
echo

failures=0
for lib in sdg_hub its_hub training_hub; do
    echo "================================================================"
    echo "  $lib"
    echo "================================================================"
    if uv run python "scripts/smoke_${lib}.py" $LIVE_FLAG; then
        :
    else
        failures=$((failures + 1))
    fi
    echo
done

echo "================================================================"
if [[ $failures -eq 0 ]]; then
    echo "  ALL SMOKE TESTS PASSED"
else
    echo "  $failures SMOKE TEST(S) FAILED — see logs above"
fi
echo "================================================================"

exit $failures
