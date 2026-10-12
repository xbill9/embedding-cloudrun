"""Dollar cost of each variant, from prices.json and the bench results. All arithmetic is here.

    python3 scripts/cost.py                 hourly cost of each instance shape while it is up
    python3 scripts/cost.py results/runs/*.json  also cost per million queries and per million documents,
                                                 median run and range when a variant has several runs

Request-based (CPU variants): billed only while a request is in flight, so a query costs its
server-side seconds times the instance's per-second rate, plus the request fee. Instance-based
(the L4): billed for the instance's whole lifetime, including idle time before it scales to zero.
Monthly (Lightsail): a flat price for every hour of the month. For both, the per-million figures
are the floor reached only when the instance is kept fully busy.
Request fees use the paid tier; the first 2M requests each month are free.
"""

import json
import statistics
import sys

SHAPES = {  # must match scripts/deploy.sh
    "cpu-repack": {"billing": "request", "vcpu": 1, "gib": 2, "l4": 0},
    "cpu-original": {"billing": "request", "vcpu": 1, "gib": 4, "l4": 0},
    "l4-original": {"billing": "instance", "vcpu": 4, "gib": 16, "l4": 1},
    # Lightsail container service, power "medium": flat monthly price from
    # `aws lightsail get-container-service-powers` (USD 40.0 on 2026-10-10), billed whether busy or idle.
    "lightsail-medium": {"billing": "monthly", "usd_month": 40.0, "vcpu": 1, "gib": 2, "l4": 0},
}
HOURS_PER_MONTH = 730
# Jev (TypeSafe): "$42 Per Billion input tokens" on https://typesafe.ai, results/jev/typesafe-price.txt.
JEV_USD_PER_M_INPUT_TOKENS = 0.042


def rate(prices, key):
    return prices["rates"][key]["tiers"][-1]["usd"]


def per_second(prices, shape):
    if shape["billing"] == "monthly":
        return shape["usd_month"] / (HOURS_PER_MONTH * 3600)
    if shape["billing"] == "request":
        cpu, mem = rate(prices, "request_cpu_vcpu_s"), rate(prices, "request_mem_gib_s")
    else:
        cpu, mem = rate(prices, "instance_cpu_vcpu_s"), rate(prices, "instance_mem_gib_s")
    return shape["vcpu"] * cpu + shape["gib"] * mem + shape["l4"] * rate(prices, "l4_no_zonal_redundancy_s")


def main():
    prices = json.load(open("prices.json"))
    fee = rate(prices, "requests")
    print(f"prices fetched {prices['fetched']}, {prices['region']}\n")
    print(f"{'variant':14s} {'billing':9s} {'USD/hour up':>12s}")
    for name, shape in SHAPES.items():
        if shape["billing"] == "monthly" and not any(name in a for a in sys.argv[1:]):
            continue
        print(f"{name:14s} {shape['billing']:9s} {per_second(prices, shape) * 3600:12.4f}")
    runs = [json.load(open(p)) for p in sys.argv[1:]]
    if not runs:
        return
    by = {}
    for r in runs:
        by.setdefault(r["variant"], []).append(r)

    def cost(name, q_s, docs, per_request):
        shape = SHAPES[name]
        ps = per_second(prices, shape)
        f = fee if shape["billing"] == "request" else 0.0
        return q_s * ps + f, ps / docs + f / per_request

    # Several runs of one variant land on different hosts: report the median run and the range.
    print(f"\n{'variant':14s} {'runs':>4s} {'query ms':>17s} {'docs/s':>17s} {'USD / 1M queries':>17s} {'USD / 1M docs':>19s}")
    per_doc_by = {}
    for name, rs in by.items():
        qs = sorted(r["query_server_s_median"] for r in rs)
        ds = sorted(r["docs_per_s_server"] for r in rs)
        n = rs[0]["docs_per_request"]
        q_med, d_med = statistics.median(qs), statistics.median(ds)
        per_q, per_doc = cost(name, q_med, d_med, n)
        _, doc_hi = cost(name, q_med, ds[0], n)
        _, doc_lo = cost(name, q_med, ds[-1], n)
        per_doc_by[name] = per_doc
        rng = lambda lo, mid, hi, k, w: f"{mid * k:.{w}f} ({lo * k:.{w}f}-{hi * k:.{w}f})"
        print(f"{name:14s} {len(rs):4d} {rng(qs[0], q_med, qs[-1], 1000, 0):>17s} {rng(ds[0], d_med, ds[-1], 1, 1):>17s} "
              f"{per_q * 1e6:17.2f} {rng(doc_lo, per_doc, doc_hi, 1e6, 2):>19s}")
    gpu = [v for v in per_doc_by if SHAPES[v]["billing"] == "instance"]
    cpu = [v for v in per_doc_by if SHAPES[v]["billing"] == "request"]
    if gpu and cpu:
        # An instance-billed GPU busy a fraction u of the time it is up costs per_doc / u per document.
        best = min(cpu, key=per_doc_by.get)
        print(f"\nL4 cost per 1M docs when the instance is busy part of the time it is up (vs {best}, median runs):")
        print(f"{'busy':>6s} {'L4 USD / 1M docs':>17s} {best + ' USD / 1M docs':>26s}")
        for u in (1.0, 0.5, 0.25, 0.10, 0.05, 0.01):
            print(f"{u:6.0%} {per_doc_by[gpu[0]] / u * 1e6:17.2f} {per_doc_by[best] * 1e6:26.2f}")
        print(f"break-even: the L4 is cheaper per document while busy more than "
              f"{per_doc_by[gpu[0]] / per_doc_by[best]:.1%} of the time it is up")
    try:
        tokens = json.load(open("results/tokens.json"))
    except FileNotFoundError:
        return
    # Per input token, prompt and special tokens included (scripts/count_tokens.py).
    print(f"\nUSD per 1M input tokens, documents of {tokens['doc_tokens']} tokens (median runs):")
    for name, per_doc in per_doc_by.items():
        print(f"{name:14s} {per_doc / tokens['doc_tokens'] * 1e6:8.4f}"
              f"{'  at 100% busy' if SHAPES[name]['billing'] == 'instance' else ''}")
    for name in gpu:
        print(f"{name:14s} {per_doc_by[name] / 0.10 / tokens['doc_tokens'] * 1e6:8.4f}  at 10% busy")
    print(f"{'jev (hosted)':14s} {JEV_USD_PER_M_INPUT_TOKENS:8.4f}  list price")
    for name in gpu:
        per_m = per_doc_by[name] / tokens["doc_tokens"] * 1e6
        print(f"{name} matches Jev's price per input token while busy {per_m / JEV_USD_PER_M_INPUT_TOKENS:.1%} "
              f"of the time it is up; at 100% busy Jev costs {JEV_USD_PER_M_INPUT_TOKENS / per_m:.1f}x as much")
    for name in cpu:
        per_m = per_doc_by[name] / tokens["doc_tokens"] * 1e6
        print(f"{name} costs {per_m / JEV_USD_PER_M_INPUT_TOKENS:.1f}x Jev's price per input token")

if __name__ == "__main__":
    main()
