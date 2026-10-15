#!/usr/bin/env bash
# Repeated cold starts: one /health request after the service has been idle WAIT seconds, N times.
#   scripts/coldstart_series.sh URL N WAIT OUTPREFIX     e.g. ... 8 1200 results/coldstart-l4-series/run
# Each sample is scripts/coldstart.sh's output in OUTPREFIX-<i>.txt. Between samples nothing else
# may call the service, or the next sample is warm. WAIT must exceed Cloud Run's scale-to-zero delay.
set -euo pipefail
URL=${1:?URL}; N=${2:?N}; WAIT=${3:?WAIT seconds}; OUT=${4:?OUTPREFIX}
mkdir -p "$(dirname "$OUT")"
for i in $(seq 1 "$N"); do
  sleep "$WAIT"
  "$(dirname "$0")/coldstart.sh" "$URL" > "$OUT-$i.txt"
  tail -n 2 "$OUT-$i.txt" | head -n 1
done
