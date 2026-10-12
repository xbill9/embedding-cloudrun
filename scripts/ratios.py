"""Ratios the article quotes, from the median of each variant's runs and results/scifact.json.

    python3 scripts/ratios.py > results/ratios.txt
"""

import glob
import json
import statistics

runs = {}
for p in glob.glob("results/runs/*.json"):
    r = json.load(open(p))
    runs.setdefault(r["variant"], []).append(r)
docs = {v: statistics.median(r["docs_per_s_server"] for r in rs) for v, rs in runs.items()}
query = {v: statistics.median(r["query_server_s_median"] for r in rs) for v, rs in runs.items()}
sci = json.load(open("results/scifact.json"))["variants"]

print(f"L4 docs/s over cpu-repack docs/s: {docs['l4-original'] / docs['cpu-repack']:.0f}x")
print(f"cpu-original query time over cpu-repack query time: {query['cpu-original'] / query['cpu-repack']:.2f}x")
print(f"cpu-repack docs/s over cpu-original docs/s: {docs['cpu-repack'] / docs['cpu-original']:.2f}x")
print(f"SciFact nDCG@10, repack over float32: {sci['repack']['ndcg@10'] / sci['float32']['ndcg@10']:.1%}")
for name in ("int4-table", "repack"):
    print(f"SciFact top 10 shared with float32, {name}: {sci[name]['top10_overlap_with_float32']:.1%}")
