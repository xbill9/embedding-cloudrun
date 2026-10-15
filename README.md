# EmbeddingGemma 2 on Cloud Run: CPU vs L4

**Question:** what does it cost to serve `google/embeddinggemma-2` text embeddings on Cloud Run, on a 1-vCPU instance running an int8/int4 repack, against the same model on an NVIDIA L4?

An L4 instance costs about ten times as much per hour as a 1-vCPU instance (table below). Two numbers decide the comparison: whether the L4 embeds documents more than ten times as fast, and how much of the hour a GPU instance sits idle. Request-based CPU billing charges only while a request is in flight; a GPU service uses instance-based billing, which charges for the whole time the instance is up.

Article: [EmbeddingGemma 2 on Cloud Run: an int8 CPU Service Against an NVIDIA L4](https://dev.to/gde/embeddinggemma-2-on-cloud-run-an-int8-cpu-service-against-an-nvidia-l4-15ac) (dev.to).

---

#### The three services

| service | model | instance | billing |
|---|---|---|---|
| `emb-cpu-repack` | text-only repack: int8 Linears, int4 embedding table, 203 MiB | 1 vCPU, 2 GiB | request-based |
| `emb-cpu-original` | original, text towers only, float32 | 1 vCPU, 4 GiB | request-based |
| `emb-l4-original` | original, text towers only, bf16 | 4 vCPU, 16 GiB, 1 L4, no zonal redundancy | instance-based |

4 vCPU and 16 GiB is the minimum Cloud Run allows with an L4 (`gcloud run deploy --help`). The repack runs only on CPU: its int8 layers use PyTorch's fbgemm kernels, so the L4 serves the original weights.

The repack script and its retrieval eval are in `repack/`; `app/cpu_load.py` loads the result.

#### W4A16 repack

`repack/repack_text_w4a16_emb4.py` builds the same text-only model with every Linear at int4 (group 32 along the input dimension, bf16 scales, float activations), in the `-w4a16-ct` layout of the Gemma 4 builds; the embedding table is the same int4 table. `app/cpu_load.py` loads it too, dequantizing the Linears to float32 at load. The model has no QAT export, so every group is rounded from bf16.

| build | `model.safetensors` | SciFact nDCG@10 | recall@10 | top 10 shared with float32 |
|---|---|---|---|---|
| float32 original (text towers) | 1.4 GB (all towers) | 0.8542 | 0.9437 | 100% |
| w8a8 int8, int4 table | 203 MiB | 0.8358 | 0.9259 | 85.5% |
| w4a16, int4 table | 146 MiB | 0.8336 | 0.9342 | 80.7% |

From `results/scifact.json` (300 queries, 5,183 documents, 12 threads). Validation, each in `results/`:

- `w4a16-verify.txt`: `repack_text_w4a16_emb4.py verify` rereads both checkpoints; every source text tensor is present, copied tensors are byte-identical, and each packed tensor's error recomputed from disk matches the report (int4 Linears 8.5–10.5% relative error against bf16; int8 Linears are 0.8–1.1%).
- `w4a16-compressed-tensors.txt`: `repack/check_compressed_tensors.py` decodes all 218 packed tensors with compressed-tensors 0.19.0's `PackedQuantizationCompressor`; they match `cpu_load.unpack_int4` exactly.
- `w4a16-cosine.txt`: mean cosine to float32 over `docs/corpus.txt`, 0.984 for documents and 0.986 for queries (w8a8: 0.993, 0.995).

Not tested: loading in vLLM. Transformers' `from_pretrained` with the compressed-tensors integration fails on this build at the quantized embedding table (`EmbeddingGemma2TextScaledWordEmbedding` has no `weight` during weight init); the w8a8 build was not tried.

---

#### Prices

From the Cloud Billing Catalog API on 2026-10-10, us-central1 (`scripts/fetch_prices.py` writes `prices.json`):

| rate | USD |
|---|---|
| CPU, request-based | 0.000024 per vCPU-second |
| memory, request-based | 0.0000025 per GiB-second |
| CPU, instance-based | 0.000018 per vCPU-second |
| memory, instance-based | 0.000002 per GiB-second |
| L4, no zonal redundancy | 0.0001867 per second |
| L4, zonal redundancy | 0.0002909 per second |
| requests | 0.40 per million after the first 2 million each month |

`python3 scripts/cost.py`:

```
variant        billing    USD/hour up
cpu-repack     request         0.1044
cpu-original   request         0.1224
l4-original    instance        1.0465
```

---

#### Step 1 — Stage the models

```
scripts/prepare_models.sh
```

Copies the repack build and a snapshot of `google/embeddinggemma-2` into `models/` (git-ignored).

---

#### Step 2 — Deploy one variant

```
scripts/deploy.sh cpu-repack
```

Builds the image (`Dockerfile`, CPU or CUDA PyTorch by build arg), pushes it to Artifact Registry `embedding-cloudrun` in us-central1, and deploys a private service with `--min-instances 0 --max-instances 1`. Deploying creates billable resources.

---

#### Step 3 — Benchmark it

```
python3 scripts/bench.py cpu-repack https://emb-cpu-repack-....run.app
```

Writes `results/cpu-repack.json`: first-request latency, median latency of 50 single queries, and documents per second over 10 requests of 32 documents (~135 tokens each). Latencies come from the server's own timer as well as the client's.

---

#### Step 4 — Cost per million

```
python3 scripts/cost.py results/*.json
```

Prints USD per million queries and per million documents for each variant, computed from `prices.json` and the bench results. For the L4 these figures assume a fully busy instance; idle time is extra.

---

#### Step 5 — Take it down

```
scripts/down.sh
```

---

#### Status

🟢 Server runs locally on the repack (`/health`, `/embed` with Matryoshka dimensions 128–768, unit-length output).
⚠️ Image not yet built. `torch==2.14.1` from the `cu130` index is assumed for the L4 image and still has to be confirmed by the first build.
⚠️ Nothing deployed or benchmarked on Cloud Run yet. Local speed numbers come from an i7-1360P laptop; Cloud Run hosts are a different CPU, and the int8 speedup depends on which int8 instructions the host has.

#### Scope

One region, one instance shape per variant, a fixed document length, and one service at a time with at most one instance. Retrieval quality of the repack is measured in the repack's own directory (SciFact), not here.
