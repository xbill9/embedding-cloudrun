"""Write prices.json: the Cloud Run rates this project uses, read from the Cloud Billing Catalog API.

    python3 scripts/fetch_prices.py [--region us-central1] [--out prices.json]

Uses the gcloud user's access token. Every rate in the output carries the SKU description it came
from, so cost.py never relies on a number typed by hand.
"""

import argparse
import datetime
import json
import subprocess
import urllib.request

SERVICE = "services/152E-C115-5142"  # Cloud Run
WANTED = {
    "request_cpu_vcpu_s": ("Services CPU (Request-based billing)", None),
    "request_mem_gib_s": ("Services Memory (Request-based billing)", None),
    "instance_cpu_vcpu_s": ("Services CPU (Instance-based billing) in {region}", None),
    "instance_mem_gib_s": ("Services Memory (Instance-based billing) in {region}", None),
    "l4_no_zonal_redundancy_s": ("NVIDIA L4 GPU with no zonal redundancy in {region}", None),
    "l4_zonal_redundancy_s": ("NVIDIA L4 GPU with zonal redundancy in {region}", None),
    "requests": ("Requests", "global"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--region", default="us-central1")
    p.add_argument("--out", default="prices.json")
    a = p.parse_args()
    token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    skus, page = [], ""
    while True:
        req = urllib.request.Request(
            f"https://cloudbilling.googleapis.com/v1/{SERVICE}/skus?pageSize=5000&pageToken={page}",
            headers={"Authorization": f"Bearer {token}"})
        d = json.load(urllib.request.urlopen(req))
        skus += d.get("skus", [])
        page = d.get("nextPageToken", "")
        if not page:
            break
    out = {"fetched": datetime.date.today().isoformat(), "region": a.region, "source": SERVICE, "rates": {}}
    for key, (desc, region) in WANTED.items():
        desc = desc.format(region=a.region)
        match = [s for s in skus if s["description"] == desc
                 and (region or a.region) in s.get("serviceRegions", [])]
        if len(match) != 1:
            raise SystemExit(f"{key}: expected one SKU '{desc}', found {len(match)}")
        expr = match[0]["pricingInfo"][0]["pricingExpression"]
        tiers = [{"from": t["startUsageAmount"],
                  "usd": int(t["unitPrice"].get("units", "0")) + t["unitPrice"].get("nanos", 0) / 1e9}
                 for t in expr["tieredRates"]]
        out["rates"][key] = {"sku": desc, "unit": expr["usageUnit"], "tiers": tiers}
    json.dump(out, open(a.out, "w"), indent=2)
    for k, v in out["rates"].items():
        print(f"{k:28s} {v['tiers'][-1]['usd']:.4g} USD per {v['unit']}  ({v['sku']})")


if __name__ == "__main__":
    main()
