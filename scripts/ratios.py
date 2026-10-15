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
# The published article's ratios use the runs it was written on (up to 2026-10-12); later runs
# are compared separately below.
DAY = "2026-10-15"
article = {v: [r for r in rs if r["date"] < DAY] for v, rs in runs.items()}
docs = {v: statistics.median(r["docs_per_s_server"] for r in rs) for v, rs in article.items() if rs}
query = {v: statistics.median(r["query_server_s_median"] for r in rs) for v, rs in article.items() if rs}
sci = json.load(open("results/scifact.json"))["variants"]

print(f"L4 docs/s over cpu-repack docs/s: {docs['l4-original'] / docs['cpu-repack']:.0f}x")
print(f"cpu-original query time over cpu-repack query time: {query['cpu-original'] / query['cpu-repack']:.2f}x")
print(f"cpu-repack docs/s over cpu-original docs/s: {docs['cpu-repack'] / docs['cpu-original']:.2f}x")
print(f"SciFact nDCG@10, repack over float32: {sci['repack']['ndcg@10'] / sci['float32']['ndcg@10']:.1%}")
print(f"SciFact nDCG@10, w4a16 over float32: {sci['w4a16']['ndcg@10'] / sci['float32']['ndcg@10']:.1%}")
for name in ("int4-table", "repack", "w4a16"):
    print(f"SciFact top 10 shared with float32, {name}: {sci[name]['top10_overlap_with_float32']:.1%}")

# W4A16 against the int8 repack: only the runs both services had on the same day, alternating.
same = {v: [r for r in rs if r["date"].startswith(DAY)] for v, rs in runs.items()}
d = {v: statistics.median(r["docs_per_s_server"] for r in same[v]) for v in ("cpu-w4a16", "cpu-repack")}
q = {v: statistics.median(r["query_server_s_median"] for r in same[v]) for v in ("cpu-w4a16", "cpu-repack")}
print(f"{DAY}, {len(same['cpu-w4a16'])} runs each: cpu-w4a16 query time over cpu-repack: "
      f"{q['cpu-w4a16'] / q['cpu-repack']:.2f}x; cpu-repack docs/s over cpu-w4a16: {d['cpu-repack'] / d['cpu-w4a16']:.2f}x")
