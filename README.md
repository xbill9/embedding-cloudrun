# EmbeddingGemma 2 on Cloud Run: CPU vs L4

**Question:** what does it cost to serve `google/embeddinggemma-2` text embeddings on Cloud Run, on a 1-vCPU instance running an int8/int4 repack, against the same model on an NVIDIA L4?

An L4 instance costs about ten times as much per hour as a 1-vCPU instance (table below). Two numbers decide the comparison: whether the L4 embeds documents more than ten times as fast, and how much of the hour a GPU instance sits idle. Request-based CPU billing charges only while a request is in flight; a GPU service uses instance-based billing, which charges for the whole time the instance is up.

Article: [EmbeddingGemma 2 on Cloud Run: an int8 CPU Service Against an NVIDIA L4](https://dev.to/gde/embeddinggemma-2-on-cloud-run-an-int8-cpu-service-against-an-nvidia-l4-15ac) (dev.to).

---

#### The three services

| service | model | instance | billing |
|---|---|---|---|
| `emb-cpu-repack` | text-only repack: int8 Linears, int4 embedding table, 203 MiB | 1 vCPU, 2 GiB | request-based |
| `emb-cpu-w4a16` | text-only repack: int4 Linears (dequantized to float32 at load), int4 embedding table, 146 MiB | 1 vCPU, 3 GiB | request-based |
| `emb-cpu-original` | original, text towers only, float32 | 1 vCPU, 4 GiB | request-based |
| `emb-l4-original` | original, text towers only, bf16 | 4 vCPU, 16 GiB, 1 L4, no zonal redundancy | instance-based |

4 vCPU and 16 GiB is the minimum Cloud Run allows with an L4 (`gcloud run deploy --help`). The repack runs only on CPU: its int8 layers use PyTorch's fbgemm kernels, so the L4 serves the original weights.

The repack script and its retrieval eval are in `repack/`; `app/cpu_load.py` loads the result.

Both builds are on Hugging Face, each with `cpu_load.py`: [`xbill9/embeddinggemma-2-w8a8-ct-text-emb4`](https://huggingface.co/xbill9/embeddinggemma-2-w8a8-ct-text-emb4) (the one the CPU service runs) and [`xbill9/embeddinggemma-2-w4a16-ct-text-emb4`](https://huggingface.co/xbill9/embeddinggemma-2-w4a16-ct-text-emb4).

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

On Cloud Run (`scripts/deploy.sh cpu-w4a16`), against the int8 repack redeployed the same day, three runs each, alternating (`results/runs/*2026-10-15*`, `results/cost-w4a16-2026-10-15.txt`):

```
variant        runs          query ms            docs/s  USD / 1M queries       USD / 1M docs
cpu-repack        3     133 (132-135)     1.0 (1.0-1.1)              4.25 27.67 (26.16-28.55)
cpu-w4a16         3     201 (190-216)     0.8 (0.8-0.8)              6.74 38.09 (37.20-39.43)
```

Queries take 1.52x as long and the int8 repack embeds documents 1.27x as fast (`results/ratios.txt`): the int4 Linears run in float32. The service needs 3 GiB, against 2 GiB for the int8 repack: loading plus one 32-document batch peaks at 2,127 MiB locally (int8: 1,856 MiB). The image is 580 MB compressed against 636 MB (`results/images.txt`). One cold start each on 2026-10-15: 164.9 s for w4a16, 134.8 s for the int8 repack, whose earlier cold starts took 58.4 s and 83.5 s (`results/coldstart.txt`). It passes `scripts/test_service.py` against the int8 service (cosine per text 0.971 to 0.993, `results/test-cpu-w4a16-vs-repack.txt`).

Not tested: loading in vLLM. Transformers' `from_pretrained` with the compressed-tensors integration fails on this build at the quantized embedding table (`EmbeddingGemma2TextScaledWordEmbedding` has no `weight` during weight init); the w8a8 build was not tried.

---

#### CPU or L4?

The L4 runs in us-east4 (us-central1 had no quota for an L4 without zonal redundancy; the prices are the same). It loads the float32 model in 4.7–6.4 s, holds 516 MiB of its 22,563 MiB of GPU memory and passes `scripts/test_service.py` against the int8 service (cosine 0.9887–0.9977).

Throughput, `scripts/sweep.py` (`results/sweep-l4-original-2026-10-15T181046Z.json`): the best client rate is 171.8 documents per second (64-document requests, two at a time). Timed inside the server, the GPU does about 328 documents per second at batch 32. Four 256-document requests at once peak at 10.3 GiB of GPU memory.

Five cold starts after 20 minutes idle (`results/coldstart-l4-series/`): first request 17.1–38.9 s, warm request 0.09–0.11 s. Each of those five single-request bursts was billed 265–621 s of L4 time, mean 474 s (`results/billed-l4-series.txt`, Cloud Monitoring), because an instance-billed service keeps charging while idle until it scales to zero.

`python3 scripts/tradeoff.py` (`results/tradeoff-cpu-l4.txt`):

| | CPU int8 (`cpu-repack`) | L4 |
|---|---|---|
| USD per 1M documents while busy | 27.67 | 1.69 |
| one isolated burst of 1,000 documents | USD 0.028 | USD 0.140 |
| one isolated burst of 100,000 documents | USD 2.77 | USD 0.31 |
| break-even burst size | 5,307 documents | |
| kept warm all month | | USD 764, the CPU price of 27.6M documents |

🟢 The 1 vCPU int8 service is the cheaper choice for queries and for bursts under about 5,300 documents, and costs nothing while idle.
🟢 The L4 is about 16x cheaper per document once a burst is large enough to pay for its idle tail.
⚠️ The CPU service needs its memory sized to document length: the int8 model alone holds about 1.35 GiB, and a 32-document batch of 512-token documents peaks at 3,551 MiB locally against 1,871 MiB at 135 tokens (`results/cpu-repack-memory.txt`). At 2 GiB, 512-token documents need batches of 8 or fewer.
⚠️ The L4 client rate is probably limited by the 768-number JSON responses and the network rather than the GPU; untested.

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
