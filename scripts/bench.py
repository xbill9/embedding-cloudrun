"""Time a deployed service: first-request latency, single-query latency and document throughput.

    python3 scripts/bench.py VARIANT URL [--queries 50] [--batches 10] [--out results/VARIANT.json]

VARIANT is the deploy.sh name (cpu-repack, cpu-w4a16, cpu-original, l4-original); cost.py reads it to know
the instance shape. Each run is also kept as results/runs/VARIANT-DATE.json. The services are public, so requests carry no credentials. Documents are 32 copies of a
fixed ~135-token paragraph per request, so every variant embeds the same tokens.
"""

import argparse
import datetime
import json
import os
import statistics
import time
import urllib.request

QUERY = "What causes the northern lights?"
DOC = ("The northern lights are caused by charged particles from the sun colliding with gases in "
       "Earth's upper atmosphere, which emit light as they return to a lower energy state. ") * 4


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("variant")
    p.add_argument("url")
    p.add_argument("--queries", type=int, default=50)
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--out")
    a = p.parse_args()

    def call(path, body=None):
        req = urllib.request.Request(a.url.rstrip("/") + path, headers={
            "Content-Type": "application/json"},
            data=None if body is None else json.dumps(body).encode())
        t = time.perf_counter()
        r = json.load(urllib.request.urlopen(req, timeout=600))
        return r, time.perf_counter() - t

    health, first_s = call("/health")
    call("/embed", {"texts": [QUERY], "prompt_name": "SearchQuery"})
    q_client, q_server = [], []
    for _ in range(a.queries):
        r, s = call("/embed", {"texts": [QUERY], "prompt_name": "SearchQuery"})
        q_client.append(s)
        q_server.append(r["seconds"])
    d_client, d_server = [], []
    for _ in range(a.batches):
        r, s = call("/embed", {"texts": [DOC] * 32, "prompt_name": "Document"})
        d_client.append(s)
        d_server.append(r["seconds"])
    out = {
        "variant": a.variant, "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "health": health, "first_request_s": first_s,
        "query_client_s_median": statistics.median(q_client), "query_server_s_median": statistics.median(q_server),
        "query_client_s_max": max(q_client),
        "docs_per_request": 32, "doc_batch_client_s_median": statistics.median(d_client),
        "doc_batch_server_s_median": statistics.median(d_server),
        "docs_per_s_server": 32 / statistics.median(d_server),
    }
    path = a.out or os.path.join("results", f"{a.variant}.json")
    json.dump(out, open(path, "w"), indent=2)
    # Keep every run: one instance lands on whichever host Cloud Run picks, and hosts differ.
    os.makedirs(os.path.join("results", "runs"), exist_ok=True)
    stamp = out["date"].replace(":", "").replace("+0000", "Z")
    json.dump(out, open(os.path.join("results", "runs", f"{a.variant}-{stamp}.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
