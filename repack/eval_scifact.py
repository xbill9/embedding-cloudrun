"""Retrieval quality of this repack against google/embeddinggemma-2, on SciFact (BEIR test split).

    python3 eval_scifact.py DATA_DIR REPACK_DIR --original ORIGINAL_DIR [--variants a,b] [--out results.json]

DATA_DIR holds corpus.jsonl, queries.jsonl and test.tsv from huggingface.co/datasets/mteb/scifact.
Every variant embeds the same 300 queries ("SearchQuery" prompt) and 5,183 documents
("title: {title} | text: {text}"), max 512 tokens, 768 dimensions, and ranks by cosine.

Variants (names for --variants): float32 (the original), repack (cpu_load defaults: int8
Linears except the float-input ones, int4 table), repack-all-int8 (every Linear int8 per-tensor
activations, the first loader), int8-linears (repack-all-int8 with the original table), int4-table
(the original with only the int4 table). nDCG@10 and recall@10 are computed here, and each
variant's top 10 is kept in the output; once float32 is in the file, every variant also gets how
many of each query's top 10 match float32's. Results merge into an existing --out file.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from cpu_load import Int4Embedding, build_model, load  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


def read(data):
    corpus = [json.loads(line) for line in open(os.path.join(data, "corpus.jsonl"))]
    queries = {q["_id"]: q["text"] for q in map(json.loads, open(os.path.join(data, "queries.jsonl")))}
    qrels = {}
    for line in list(open(os.path.join(data, "test.tsv")))[1:]:
        qid, did, score = line.split("\t")
        if int(score) > 0:
            qrels.setdefault(qid, set()).add(did)
    qids = sorted(qrels)
    return corpus, [queries[q] for q in qids], [qrels[q] for q in qids]


def run(model, corpus, queries):
    t = time.perf_counter()
    d = model.encode([f"title: {c['title'] or 'none'} | text: {c['text']}" for c in corpus],
                     batch_size=32, convert_to_tensor=True)
    q = model.encode(queries, prompt_name="SearchQuery", batch_size=32, convert_to_tensor=True)
    top = (q @ d.T).topk(10, dim=1).indices.tolist()
    return top, time.perf_counter() - t


def score(top, corpus, rel):
    ndcg = recall = 0.0
    for ranks, r in zip(top, rel):
        ids = [corpus[i]["_id"] for i in ranks]
        dcg = sum(1 / math.log2(k + 2) for k, x in enumerate(ids) if x in r)
        ideal = sum(1 / math.log2(k + 2) for k in range(min(len(r), 10)))
        ndcg += dcg / ideal
        recall += len(set(ids) & r) / len(r)
    return ndcg / len(top), recall / len(top)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("data")
    p.add_argument("repack")
    p.add_argument("--original", required=True)
    p.add_argument("--variants", default="float32,repack,int4-table")
    p.add_argument("--out", default="eval_scifact.json")
    a = p.parse_args()
    corpus, queries, rel = read(a.data)

    def original():
        m = SentenceTransformer(a.original, device="cpu",
                                config_kwargs={"vision_config": None, "audio_config": None},
                                model_kwargs={"dtype": torch.float32})
        return m

    def int8_linears():
        m = load(a.repack, float_inputs=())
        m[0].model.language_model.embed_tokens = original()[0].model.language_model.embed_tokens
        return m

    def int4_table():
        m = original()
        sd = load_file(os.path.join(a.repack, "model.safetensors"))
        e = m[0].model.language_model.embed_tokens
        m[0].model.language_model.embed_tokens = Int4Embedding(
            sd["language_model.embed_tokens.weight_packed"], sd["language_model.embed_tokens.weight_scale"],
            sd["language_model.embed_tokens.weight_shape"], e.embed_scale.item())
        return m

    variants = {"float32": original, "repack": lambda: load(a.repack),
                "repack-all-int8": lambda: load(a.repack, float_inputs=()),
                "int8-linears": int8_linears, "int4-table": int4_table}
    out = json.load(open(a.out)) if os.path.exists(a.out) else {"queries": len(queries), "documents": len(corpus),
                                                                 "variants": {}}
    for name in a.variants.split(","):
        m = variants[name]()
        m.max_seq_length = 512
        top, secs = run(m, corpus, queries)
        ndcg, recall = score(top, corpus, rel)
        out["variants"][name] = {"ndcg@10": ndcg, "recall@10": recall, "seconds": secs,
                                 "threads": torch.get_num_threads(), "top10": top}
        print(f"{name:16s} nDCG@10 {ndcg:.4f}  recall@10 {recall:.4f}  {secs:.0f}s", flush=True)
        del m
        ref = out["variants"].get("float32", {}).get("top10")
        for v in out["variants"].values():
            if ref:
                v["top10_overlap_with_float32"] = sum(
                    len(set(t) & set(r)) for t, r in zip(v["top10"], ref)) / (10 * len(ref))
        json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
