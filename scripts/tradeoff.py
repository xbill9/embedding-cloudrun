"""CPU int8 (cpu-repack) against the L4 (l4-original): cost of a burst and of a month. All arithmetic is here.

    python3 scripts/tradeoff.py

Inputs, each from a file in results/:
- L4 hourly price: prices.json through cost.py's shape table.
- L4 billed seconds per isolated burst: results/billed-l4-series.txt (Cloud Monitoring, five cold cycles
  of one cold and one warm request after 20 minutes idle; the minutes before the first cycle belong to the
  sweep and are skipped).
- L4 throughput: results/sweep-l4-original-2026-10-15T181046Z.json, best client end-to-end docs/s.
- CPU USD per 1M documents: 27.67, cpu-repack median of the same-day runs (cost.py on results/runs).
"""

import datetime as dt
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost import HOURS_PER_MONTH, JEV_USD_PER_M_INPUT_TOKENS, SHAPES, per_second  # noqa: E402

CPU_USD_PER_M_DOCS = 27.67
TOKENS_PER_DOC = CPU_USD_PER_M_DOCS / 0.1963  # cost.py: USD 0.1963 per 1M input tokens for the same runs

prices = json.load(open("prices.json"))
l4_s = per_second(prices, SHAPES["l4-original"])

rows = [(dt.datetime.fromisoformat(t), float(s)) for t, s in
        re.findall(r"^(\S+)\s+([\d.]+)\s+\{", open("results/billed-l4-series.txt").read(), re.M)]
blocks, prev = [], None
for t, s in rows:
    if prev is None or (t - prev).total_seconds() > 60:
        blocks.append([])
    blocks[-1].append(s)
    prev = t
cycles = [sum(b) for b in blocks[1:]]
burst_s = sum(cycles) / len(cycles)

sweep = json.load(open("results/sweep-l4-original-2026-10-15T181046Z.json"))
cells = sweep["cells"]
l4_dps = max(c["docs_per_s_client"] for c in cells)

cpu_doc = CPU_USD_PER_M_DOCS / 1e6
print(f"L4 USD/hour {l4_s * 3600:.4f}; billed s per isolated burst {min(cycles):.0f}-{max(cycles):.0f}, mean {burst_s:.0f}")
print(f"  one isolated burst costs USD {burst_s * l4_s:.4f} (mean), the price of {burst_s * l4_s / cpu_doc:,.0f} documents on the CPU")
print(f"L4 best end-to-end {l4_dps:.1f} docs/s -> USD {l4_s / l4_dps * 1e6:.2f} per 1M docs while busy; CPU USD {CPU_USD_PER_M_DOCS}")

print(f"\nburst of N documents (one cold L4 cycle vs CPU)")
print(f"{'N':>10s} {'CPU USD':>9s} {'L4 USD':>9s}")
for n in (100, 1_000, 10_000, 100_000, 1_000_000):
    print(f"{n:10,d} {n * cpu_doc:9.4f} {(burst_s + n / l4_dps) * l4_s:9.4f}")
# break-even N: n*cpu = (burst + n/dps)*l4  ->  n = burst*l4 / (cpu - l4/dps)
be = burst_s * l4_s / (cpu_doc - l4_s / l4_dps)
print(f"break-even burst size {be:,.0f} documents")

month_on = l4_s * 3600 * HOURS_PER_MONTH
print(f"\nL4 kept warm all month (traffic at least every ~{burst_s / 60:.0f} min): USD {month_on:.0f}/month,"
      f" equal to {month_on / cpu_doc / 1e6:.1f}M documents/month on the CPU")
print(f"monthly documents  CPU USD  Jev USD (at {TOKENS_PER_DOC:.0f} tokens/doc)")
for m in (0.1, 1, 10, 30, 100):
    print(f"{m:10.1f}M {m * CPU_USD_PER_M_DOCS:10.2f} {m * TOKENS_PER_DOC * JEV_USD_PER_M_INPUT_TOKENS:10.2f}")
