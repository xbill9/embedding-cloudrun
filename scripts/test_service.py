"""Check a deployed service end to end: health, shape, unit length, Matryoshka prefixes, ranking.

    python3 scripts/test_service.py URL [--against URL2]

--against embeds the same texts on a second service (the float32 or bf16 original) and reports the
cosine similarity between the two services' vectors for each text. Exits 1 if any check fails.
"""

import argparse
import json
import math
import sys
import urllib.error
import urllib.request

QUERY = "How do plants make food from sunlight?"
DOCS = [
    "Photosynthesis converts light energy, water and carbon dioxide into glucose and oxygen in chloroplasts.",
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War in the Holy Roman Empire.",
    "A sourdough starter is a mixture of flour and water fermented by wild yeast and lactic acid bacteria.",
]


def post(url, body):
    req = urllib.request.Request(url.rstrip("/") + "/embed", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        return {"http_status": e.code, "body": e.read().decode()[:200]}


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (norm(a) * norm(b))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("url")
    p.add_argument("--against")
    a = p.parse_args()
    results = []

    def check(name, ok, detail):
        results.append(ok)
        print(f"{'✅' if ok else '❌'} {name}: {detail}")

    health = json.load(urllib.request.urlopen(a.url.rstrip("/") + "/health", timeout=300))
    check("health", "variant" in health, json.dumps(health))

    for dim in (128, 256, 512, 768):
        e = post(a.url, {"texts": [QUERY], "prompt_name": "SearchQuery", "dim": dim})["embeddings"][0]
        check(f"dim {dim}", len(e) == dim and abs(norm(e) - 1) < 1e-3, f"length {len(e)}, norm {norm(e):.4f}")

    full = post(a.url, {"texts": [QUERY], "prompt_name": "SearchQuery", "dim": 768})["embeddings"][0]
    short = post(a.url, {"texts": [QUERY], "prompt_name": "SearchQuery", "dim": 128})["embeddings"][0]
    c = cos(full[:128], short)
    check("Matryoshka prefix", c > 0.999, f"cosine(first 128 of 768, dim 128) = {c:.5f}")

    bad = post(a.url, {"texts": [QUERY], "dim": 100})
    check("rejects dim 100", "error" in bad, json.dumps(bad))
    empty = post(a.url, {"texts": []})
    check("rejects empty texts", empty.get("http_status") == 422, f"HTTP {empty.get('http_status')}")

    q = post(a.url, {"texts": [QUERY], "prompt_name": "SearchQuery"})["embeddings"][0]
    d = post(a.url, {"texts": DOCS, "prompt_name": "Document"})["embeddings"]
    scores = [cos(q, x) for x in d]
    check("ranking", scores[0] == max(scores), "scores " + ", ".join(f"{s:.3f}" for s in scores))

    if a.against:
        texts = [QUERY] + DOCS
        mine = post(a.url, {"texts": texts, "prompt_name": "Document"})["embeddings"]
        theirs = post(a.against, {"texts": texts, "prompt_name": "Document"})["embeddings"]
        sims = [cos(x, y) for x, y in zip(mine, theirs)]
        check("agrees with --against", min(sims) > 0.95,
              f"cosine per text min {min(sims):.4f} max {max(sims):.4f}")

    print(f"{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
