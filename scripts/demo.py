"""Semantic search over a small corpus with a deployed service.

    python3 scripts/demo.py URL "your question" [--dim 768] [--top 3]

Embeds docs/corpus.txt (one passage per line) as documents and the question as a search query,
then prints the passages ranked by cosine similarity. The vectors are unit length, so the dot
product is the cosine.
"""

import argparse
import json
import os
import time
import urllib.request

CORPUS = os.path.join(os.path.dirname(__file__), "..", "docs", "corpus.txt")


def embed(url, texts, prompt_name, dim):
    req = urllib.request.Request(url.rstrip("/") + "/embed", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"texts": texts, "prompt_name": prompt_name, "dim": dim}).encode())
    return json.load(urllib.request.urlopen(req, timeout=600))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("url")
    p.add_argument("question")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--top", type=int, default=3)
    a = p.parse_args()

    docs = [line.strip() for line in open(CORPUS) if line.strip()]
    t = time.perf_counter()
    d = embed(a.url, docs, "Document", a.dim)
    q = embed(a.url, [a.question], "SearchQuery", a.dim)
    wall = time.perf_counter() - t
    scores = [sum(x * y for x, y in zip(q["embeddings"][0], v)) for v in d["embeddings"]]
    ranked = sorted(zip(scores, docs), reverse=True)[:a.top]
    print(f"{len(docs)} passages, dim {a.dim}, server {d['seconds'] + q['seconds']:.2f} s, wall {wall:.2f} s")
    for s, doc in ranked:
        print(f"{s:.3f}  {doc}")


if __name__ == "__main__":
    main()
