#!/usr/bin/env bash
# Time the first request to an idle service (scaled to zero), then a second, warm one.
#   scripts/coldstart.sh URL
# Send it only after the service has had no traffic for long enough to scale to zero.
set -euo pipefail
URL=${1:?service URL}
date -u +%FT%TZ
curl -s -w '\nhttp=%{http_code} total_s=%{time_total}\n' "$URL/health"
curl -s -o /dev/null -w 'second http=%{http_code} total_s=%{time_total}\n' "$URL/health"
