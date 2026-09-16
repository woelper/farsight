#!/usr/bin/env bash
# Start predictd and launch predict-cli for interactive testing.
#
# Usage:
#   ./scripts/start.sh                # daemon + CLI (needs a terminal)
#   ./scripts/start.sh --daemon-only  # daemon only, leaves it running
#   ./scripts/start.sh --stop         # stop a running daemon
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=/tmp/predictd.log

stop_daemon() {
    if pkill -x predictd 2>/dev/null; then
        echo "stopped predictd"
    else
        echo "predictd not running"
    fi
}

if [[ "${1:-}" == "--stop" ]]; then
    stop_daemon
    exit 0
fi

cd "$ROOT"
cargo build -p predictd -p predict-cli
# A dev script owns the daemon while testing; clear any previous instance.
pkill -x predictd 2>/dev/null || true
(./target/debug/predictd >"$LOG" 2>&1 &)
echo "starting predictd (log: $LOG)..."

# Wait for the daemon to listen (up to ~5 s).
for _ in $(seq 1 50); do
    if grep -q "listening on" "$LOG" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if ! grep -q "listening on" "$LOG" 2>/dev/null; then
    echo "predictd failed to start; see $LOG" >&2
    tail -5 "$LOG" >&2 || true
    exit 1
fi
grep "listening on" "$LOG" | tail -1

if [[ "${1:-}" == "--daemon-only" ]]; then
    echo "daemon running; socket ready."
    echo "run ./target/debug/predict-cli in a terminal to test, then ./scripts/start.sh --stop"
    exit 0
fi

if [[ ! -t 0 ]]; then
    echo "no terminal on stdin; leaving daemon running." >&2
    echo "run ./target/debug/predict-cli in a terminal, then ./scripts/start.sh --stop" >&2
    exit 0
fi

status=0
./target/debug/predict-cli || status=$?
stop_daemon
exit "$status"
