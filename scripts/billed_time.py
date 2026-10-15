"""Billed instance seconds and instance counts for one Cloud Run service, from Cloud Monitoring.

    python3 scripts/billed_time.py SERVICE REGION START END     (RFC 3339 times, UTC)

run.googleapis.com/container/billable_instance_time is the instance time Cloud Run bills; for an
instance-billed service (the L4) that includes idle time before it scales to zero. Prints it per
minute with run.googleapis.com/container/instance_count, and the total in seconds.
"""

import json
import subprocess
import sys
import urllib.parse
import urllib.request

service, region, start, end = sys.argv[1:5]
project = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True).stdout.strip()
token = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True).stdout.strip()


def series(metric):
    q = urllib.parse.urlencode({
        "filter": f'metric.type="{metric}" AND resource.labels.service_name="{service}" '
                  f'AND resource.labels.location="{region}"',
        "interval.startTime": start, "interval.endTime": end, "pageSize": 100000})
    req = urllib.request.Request(f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries?{q}",
                                 headers={"Authorization": f"Bearer {token}"})
    return json.load(urllib.request.urlopen(req)).get("timeSeries", [])


def value(p):
    v = p["value"]
    return float(v.get("doubleValue", v.get("int64Value", 0)))


billed, count = {}, {}
for s in series("run.googleapis.com/container/billable_instance_time"):
    for p in s["points"]:
        t = p["interval"]["endTime"]
        billed[t] = billed.get(t, 0.0) + value(p)
for s in series("run.googleapis.com/container/instance_count"):
    state = s["metric"]["labels"].get("state", "?")
    for p in s["points"]:
        count.setdefault(p["interval"]["endTime"], {})[state] = value(p)

print(f"{'minute ending (UTC)':22s} {'billed s':>9s}  instances")
for t in sorted(set(billed) | set(count)):
    if billed.get(t, 0) or any(count.get(t, {}).values()):
        print(f"{t[:19]:22s} {billed.get(t, 0):9.1f}  {count.get(t, {})}")
print(f"total billed instance seconds {sum(billed.values()):.1f}")
