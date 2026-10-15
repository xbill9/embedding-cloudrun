"""Throughput of one deployed instance as request size and concurrent requests grow.

    python3 scripts/sweep.py VARIANT URL [--sizes 32,64,128,256] [--concurrency 1,2,4] [--requests 8]

Every cell sends --requests requests of SIZE copies of bench.py's ~135-token document, SIZE texts
per forward pass, from CONCURRENCY client threads at once. docs/s (client) is all documents over
the cell's wall time, network and JSON included; docs/s (server) is SIZE over the median of the
server's own timer, the GPU's rate for one request at a time. /health is read after every cell,
so on a GPU the peak memory so far is recorded. Writes results/sweep-VARIANT-DATE.json.
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import DOC  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("variant")
    p.add_argument("url")
    p.add_argument("--sizes", default="32,64,128,256")
    p.add_argument("--concurrency", default="1,2,4")
    p.add_argument("--requests", type=int, default=8)
    a = p.parse_args()

    def call(path, body=None):
        req = urllib.request.Request(a.url.rstrip("/") + path, headers={"Content-Type": "application/json"},
                                     data=None if body is None else json.dumps(body).encode())
        return json.load(urllib.request.urlopen(req, timeout=900))

    call("/embed", {"texts": [DOC] * 32, "prompt_name": "Document"})  # warm
    cells = []
    for size in map(int, a.sizes.split(",")):
        body = {"texts": [DOC] * size, "prompt_name": "Document", "batch_size": size}
        for conc in map(int, a.concurrency.split(",")):
            t = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(conc) as pool:
                server = [r["seconds"] for r in pool.map(lambda _: call("/embed", body), range(a.requests))]
            wall = time.perf_counter() - t
            cell = {"size": size, "concurrency": conc, "requests": a.requests, "wall_s": wall,
                    "docs_per_s_client": size * a.requests / wall,
                    "docs_per_s_server": size / statistics.median(server),
                    "server_s_median": statistics.median(server), "health": call("/health")}
            cells.append(cell)
            print(f"size {size:3d} conc {conc}  docs/s client {cell['docs_per_s_client']:7.1f}  "
                  f"server {cell['docs_per_s_server']:7.1f}  gpu peak {cell['health'].get('gpu_peak_allocated_mib')} MiB",
                  flush=True)
    date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H%M%SZ")
    out = os.path.join("results", f"sweep-{a.variant}-{date}.json")
    json.dump({"variant": a.variant, "date": date, "url_region": a.url, "cells": cells}, open(out, "w"), indent=2)
    print(out)


if __name__ == "__main__":
    main()
