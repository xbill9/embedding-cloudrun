"""Summarize scripts/coldstart.sh outputs: first-request time, the server's model load, and the rest.

    python3 scripts/coldstart_summary.py results/coldstart-*.txt

The rest is everything Cloud Run does before the model starts loading: placing the instance,
pulling the image, starting the container and importing Python packages.
"""

import json
import re
import sys

print(f"{'file':38s} {'variant':12s} {'first request s':>16s} {'model load s':>13s} {'the rest s':>11s} {'warm s':>7s}")
for path in sys.argv[1:]:
    text = open(path).read()
    health = json.loads(next(line for line in text.splitlines() if line.startswith("{")))
    first = float(re.search(r"^http=200 total_s=([\d.]+)", text, re.M).group(1))
    warm = float(re.search(r"^second http=200 total_s=([\d.]+)", text, re.M).group(1))
    name = path.rsplit("/", 1)[-1]
    print(f"{name:38s} {health['variant']:12s} {first:16.1f} {health['load_s']:13.1f} {first - health['load_s']:11.1f} {warm:7.2f}")
