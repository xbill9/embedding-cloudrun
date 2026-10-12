"""Dollar cost of each variant, from prices.json and the bench results. All arithmetic is here.

    python3 scripts/cost.py                 hourly cost of each instance shape while it is up
    python3 scripts/cost.py results/*.json  also cost per million queries and per million documents

Request-based (CPU variants): billed only while a request is in flight, so a query costs its
server-side seconds times the instance's per-second rate, plus the request fee. Instance-based
(the L4): billed for the instance's whole lifetime, including idle time before it scales to zero.
Monthly (Lightsail): a flat price for every hour of the month. For both, the per-million figures
are the floor reached only when the instance is kept fully busy.
Request fees use the paid tier; the first 2M requests each month are free.
"""

import json
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
    per_doc_by = {}
    runs = [json.load(open(p)) for p in sys.argv[1:]]
    if not runs:
        return
    print(f"\n{'variant':14s} {'query ms':>9s} {'USD / 1M queries':>17s} {'docs/s':>8s} {'USD / 1M docs':>14s}")
    for r in runs:
        shape = SHAPES[r["variant"]]
        ps = per_second(prices, shape)
        q_s = r["query_server_s_median"]
        per_query = q_s * ps + (fee if shape["billing"] == "request" else 0.0)
        docs = r["docs_per_s_server"]
        per_doc = ps / docs + (fee / r["docs_per_request"] if shape["billing"] == "request" else 0.0)
        print(f"{r['variant']:14s} {q_s * 1000:9.1f} {per_query * 1e6:17.2f} {docs:8.1f} {per_doc * 1e6:14.2f}")
        per_doc_by[r["variant"]] = min(per_doc, per_doc_by.get(r["variant"], per_doc))
    gpu = [v for v in per_doc_by if SHAPES[v]["billing"] == "instance"]
    cpu = [v for v in per_doc_by if SHAPES[v]["billing"] == "request"]
    if gpu and cpu:
        # An instance-billed GPU busy a fraction u of the time it is up costs per_doc / u per document.
        best = min(cpu, key=per_doc_by.get)
        print(f"\nL4 cost per 1M docs when the instance is busy part of the time it is up (vs {best}):")
        print(f"{'busy':>6s} {'L4 USD / 1M docs':>17s} {best + ' USD / 1M docs':>26s}")
        for u in (1.0, 0.5, 0.25, 0.10, 0.05, 0.01):
            print(f"{u:6.0%} {per_doc_by[gpu[0]] / u * 1e6:17.2f} {per_doc_by[best] * 1e6:26.2f}")
        print(f"break-even: the L4 is cheaper per document while busy more than "
              f"{per_doc_by[gpu[0]] / per_doc_by[best]:.1%} of the time it is up")


if __name__ == "__main__":
    main()
