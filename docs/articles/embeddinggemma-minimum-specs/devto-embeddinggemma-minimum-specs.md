---
title: "EmbeddingGemma 2 on Cloud Run: Minimum Specs, and When an NVIDIA L4 Pays for Itself"
published: false
description: "How small a Cloud Run instance EmbeddingGemma 2 fits in, whether an int4 repack shrinks it further, and how many documents a burst needs before an NVIDIA L4 costs less than a 1 vCPU int8 service."
tags: googlecloud, gemma, cloudrun, machinelearning
cover_image: https://raw.githubusercontent.com/xbill9/embedding-cloudrun/main/docs/articles/embeddinggemma-minimum-specs/devto-cover.d94c2bc5.jpg
---

This article provides a step by step sizing guide for EmbeddingGemma 2 text embeddings on Google Cloud Run. It measures the smallest CPU instance the model fits in, tests an int4 repack of the weights, and prices a burst of work on a 1 vCPU service against the same burst on an NVIDIA L4 GPU.

https://github.com/xbill9/embedding-cloudrun

---

#### What is this project trying to Do?

The first article in this series deployed EmbeddingGemma 2 to Cloud Run three ways and priced each per million documents: https://dev.to/gde/embeddinggemma-2-on-cloud-run-an-int8-cpu-service-against-an-nvidia-l4-15ac

EmbeddingGemma 2 is a compact model. With the vision and audio encoders dropped and the Linears stored as int8, the text model is a 203.4 MiB file. This follow-up asks the minimum-spec question: is a basic 1 vCPU instance good enough, and when does the L4 cost less?

Two numbers decide it. How much memory the CPU service needs, and how much an L4 bills for a short burst of work, idle time included.

---

#### Where do I start?

The strategy is an incremental step by step approach, smallest instance first.

The int8 repack is measured for memory on a laptop at one thread, the same thread count Cloud Run gives it. An int4 repack is built and deployed beside it. Then the L4 is swept for throughput, cold started repeatedly, and its billed seconds are read back from Cloud Monitoring. One script does the cost arithmetic from those files.

---

#### At This Point You Should Have…

- The repository cloned and `models/` staged with `scripts/prepare_models.sh`, as in the first article
- The `emb-cpu-repack` service in `us-central1` and `emb-l4-original` in `us-east4`, from `scripts/deploy.sh`
- The `gcloud` CLI signed in, for the Cloud Monitoring query in Step 6

---

#### Step 1 — How Little Memory Does the CPU Service Need?

`scripts/cpu_memory.py` loads the int8 repack at one thread, embeds 32 copies of a document at a given batch size, and prints the process's peak memory. A fresh process each time, documents of 135 tokens and of 512 tokens (the model's limit):

```shell
for n in short long; do for b in 1 4 8 16 32; do python3 scripts/cpu_memory.py $b $n; done; done
```

```plaintext
batch  1 doc 135 tokens: peak RSS 1382 MiB
batch  4 doc 135 tokens: peak RSS 1369 MiB
batch  8 doc 135 tokens: peak RSS 1345 MiB
batch 16 doc 135 tokens: peak RSS 1468 MiB
batch 32 doc 135 tokens: peak RSS 1871 MiB
batch  1 doc 512 tokens: peak RSS 1385 MiB
batch  4 doc 512 tokens: peak RSS 1572 MiB
batch  8 doc 512 tokens: peak RSS 1893 MiB
batch 16 doc 512 tokens: peak RSS 2356 MiB
batch 32 doc 512 tokens: peak RSS 3551 MiB
```

The loaded model and PyTorch take about 1.35 GiB before any batch, so 1 GiB is too small at any setting. The service from the first article runs at 2 GiB with a batch of 32, which holds for 135-token documents at 1,871 MiB. A batch of 32 documents at 512 tokens peaks at 3,551 MiB.

So the 2 GiB service needs batches of 8 or fewer for long documents, or 4 GiB at a batch of 32.

---

#### 🔎 Tip: Make Batch Size a Request Field

The server now takes `batch_size` per request, 1 to 256, default 32:

```python
class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)
    prompt_name: str | None = None
    dim: int = 768
    batch_size: int = Field(32, ge=1, le=256)  # texts per forward pass
```

A client sending long documents to a 2 GiB instance asks for `"batch_size": 8` and the instance stays inside its memory limit without a redeploy. `/health` also reports `rss_mib` and `peak_rss_mib`, and on a GPU the GPU's total, allocated and peak memory, so the numbers in this article can be read from a live service.

---

#### Step 2 — Does an int4 Repack Shrink It Further?

`repack/repack_text_w4a16_emb4.py` stores every Linear in the text model as int4, in groups of 32 with a bf16 scale, in the compressed-tensors `pack-quantized` layout. Activations stay in float. The file is 146.1 MiB against the int8 repack's 203.4 MiB.

Quality on SciFact (300 queries over 5,183 abstracts, 768 dimensions) with `repack/eval_scifact.py`:

| weights | nDCG@10 | recall@10 | top 10 shared with float32 |
|---|---|---|---|
| 🥇 float32 original | 0.854 | 0.944 | 100% |
| 🥈 int8 repack | 0.836 | 0.926 | 85.5% |
| 🥉 int4 repack | 0.834 | 0.934 | 80.7% |

Both repacks keep about 98% of the original's nDCG@10: 97.9% for int8 and 97.6% for int4.

Deployed as `cpu-w4a16` on the same 1 vCPU shape, with three runs of each variant alternating on the same day:

```shell
python3 scripts/cost.py results/runs/cpu-repack-2026-10-15*.json results/runs/cpu-w4a16-*.json
```

```plaintext
variant        runs          query ms            docs/s  USD / 1M queries       USD / 1M docs
cpu-repack        3     133 (132-135)     1.0 (1.0-1.1)              4.25 27.67 (26.16-28.55)
cpu-w4a16         3     201 (190-216)     0.8 (0.8-0.8)              6.74 38.09 (37.20-39.43)
```

The server unpacks the int4 weights to float32 once at load, so every matrix multiply runs in float32, while the int8 weights run on PyTorch's int8 CPU kernels. Queries take 1.52x as long on int4 and documents cost 38% more. The int4 service also needs 3 GiB: loading plus one 32-document batch peaks at 2,127 MiB locally, against 1,856 MiB for int8.

The smaller file buys a smaller image, 580 MB compressed against 636 MB, and nothing else on a CPU. The int8 repack stays the minimum spec.

---

#### Step 3 — How Fast Is the L4 From a Client?

The first article timed the L4 inside the server: 331 documents per second, median of four runs. `scripts/sweep.py` measures what a client receives, across request sizes and concurrent requests, with the request size also used as the batch size:

```shell
python3 scripts/sweep.py l4-original $L4_URL --sizes 32,64,128,256 --concurrency 1,2,4 --requests 8
```

Documents per second as the client saw them, network and JSON included:

| documents per request | 1 at a time | 2 at a time | 4 at a time |
|---|---|---|---|
| 32 | 81.4 | 148.1 | 162.4 |
| 64 | 102.6 | 🥇 171.8 | 167.4 |
| 128 | 111.9 | 166.1 | 152.6 |
| 256 | 115.9 | 155.8 | 151.9 |

The best cell is 171.8 documents per second, at 64 documents per request and two requests at a time. The server's own timer on one request at a time gives 328.3 documents per second at a batch of 32 and falls to 192.5 at 256, so smaller batches suit this GPU.

Peak GPU memory reached 14,956 MiB reserved by PyTorch at four requests of 256, out of 22,563 MiB on the L4. The model itself holds 516 MiB.

Every response carries 768 numbers per document as JSON. That transfer is the likely limit on the client rate; this was not tested.

---

#### Step 4 — Cold Starts on the L4

`scripts/coldstart_series.sh` waits 20 minutes with no traffic, sends one request, then a second, and repeats:

```shell
scripts/coldstart_series.sh $L4_URL 8 1200 results/coldstart-l4-series/run
python3 scripts/coldstart_summary.py results/coldstart-l4-series/run-*.txt
```

```plaintext
file                                   variant       first request s  model load s  the rest s  warm s
run-1.txt                              original                 38.9           6.4        32.4    0.09
run-2.txt                              original                 17.1           4.8        12.3    0.11
run-3.txt                              original                 27.4           5.9        21.5    0.11
run-4.txt                              original                 17.3           4.7        12.7    0.10
run-5.txt                              original                 27.9           6.3        21.5    0.10
```

Five samples completed. The first request took 17.1 to 38.9 seconds, of which the model load was 4.7 to 6.4. A warm request took about 0.1 seconds.

The 1 vCPU services took longer: 58.4 and 83.5 seconds in the first article, and 134.8 seconds for the int8 repack and 164.9 for int4 on 2026-10-15.

---

#### 🔎 Tip: A Cold Start Series Needs a Quiet Service

Each sample is cold only if nothing else reaches the service during the wait. A health check, a test, or a sweep in that window starts an instance, and the next sample measures a warm request. Run the series alone, and leave the wait longer than the time Cloud Run takes to scale an idle instance to zero.

---

#### Step 5 — What Does One Burst Cost on the L4?

A GPU service on Cloud Run is billed for its instance: every second from start until it scales to zero, idle seconds included. Cloud Monitoring records that time as `run.googleapis.com/container/billable_instance_time`. `scripts/billed_time.py` reads it per minute for one service:

```shell
python3 scripts/billed_time.py emb-l4-original us-east4 2026-10-15T18:15:00Z 2026-10-15T20:10:00Z
```

```plaintext
minute ending (UTC)     billed s  instances
2026-10-15T18:32:00         20.8  {'active': 1.0, 'idle': 0.0}
2026-10-15T18:33:00         60.0  {'active': 1.0, 'idle': 0.0}
...
2026-10-15T18:42:00         60.0  {'active': 1.0, 'idle': 0.0}
2026-10-15T18:53:00         24.7  {'active': 1.0, 'idle': 0.0}
...
total billed instance seconds 2731.5
```

Each cold cycle in Step 4 served two health requests and was billed 265 to 621 seconds, a mean of 474. At USD 1.0465 per hour, one isolated burst costs about USD 0.14 before it embeds anything.

The 1 vCPU service uses request-based billing: it is charged only while a request is in flight, and an idle instance costs nothing.

---

#### Step 6 — How Big Must a Burst Be?

`scripts/tradeoff.py` takes the billed seconds per burst from Step 5, the client rate from Step 3 and the int8 service's cost per document from Step 2, and does the arithmetic:

```shell
python3 scripts/tradeoff.py
```

```plaintext
L4 USD/hour 1.0465; billed s per isolated burst 265-621, mean 474
  one isolated burst costs USD 0.1379 (mean), the price of 4,982 documents on the CPU
L4 best end-to-end 171.8 docs/s -> USD 1.69 per 1M docs while busy; CPU USD 27.67

burst of N documents (one cold L4 cycle vs CPU)
         N   CPU USD    L4 USD
       100    0.0028    0.1380
     1,000    0.0277    0.1396
    10,000    0.2767    0.1548
   100,000    2.7670    0.3071
 1,000,000   27.6700    1.8302
break-even burst size 5,307 documents

L4 kept warm all month (traffic at least every ~8 min): USD 764/month, equal to 27.6M documents/month on the CPU
```

Below about 5,300 documents in a burst, the 1 vCPU service is cheaper. Above it, the L4 is cheaper, and at a million documents it costs USD 1.83 against USD 27.67.

An L4 that receives traffic at least every few minutes never scales to zero and costs USD 764 a month. The 1 vCPU service embeds 27.6 million documents a month for the same money.

---

#### Compare and Contrast

| | 1 vCPU int8 | 1 vCPU int4 | NVIDIA L4 |
|---|---|---|---|
| Weights | 🥈 203.4 MiB | 🥇 146.1 MiB | 🥉 float32 original, run in bf16 |
| Memory needed | 🥇 2 GiB, or 4 GiB for long documents at batch 32 | 🥈 3 GiB | 🥉 16 GiB and a GPU |
| Query, median | 🥈 133 ms | 🥉 201 ms | 🥇 57 ms |
| USD per 1M documents, busy | 🥈 27.67 | 🥉 38.09 | 🥇 1.69 from a client |
| One isolated burst of 1,000 documents | 🥇 USD 0.03 | 🥈 USD 0.04 | 🥉 USD 0.14 |
| Cold start | 🥈 58–135 s | 🥉 165 s | 🥇 17–39 s |
| SciFact nDCG@10 | 🥈 0.836 | 🥉 0.834 | 🥇 0.854 for the float32 weights |

---

#### So, Which One?

For queries and for indexing in batches under about 5,300 documents, the 1 vCPU int8 service is good enough and is the cheapest. It answers a query in a median 133 ms, costs nothing between requests and needs no GPU quota. Give it 2 GiB, with `batch_size` at 8 for documents near 512 tokens, or 4 GiB.

For a batch job of tens of thousands of documents or more, deploy the L4, run the job, and let it scale to zero. One burst bills about eight minutes of L4 time, and past 5,307 documents that is cheaper than the CPU.

The int4 repack is the smallest file and the slowest CPU service here, because this server runs it in float32. A runtime with int4 kernels may change that; none was tested.

---

#### Step 7 — Teardown

```shell
scripts/down.sh
```

It deletes the services in both regions. All of them were deployed with `--min-instances 0`, so an idle service costs nothing while it stays deployed.

---

#### Summary

The goal of this article was to find the minimum Cloud Run spec for EmbeddingGemma 2 and the point where an NVIDIA L4 costs less. The key to the solution was measuring the L4's billed instance time per burst from Cloud Monitoring, idle time included, and pricing a burst on each service from that. The sizing results were:

- 🟢 The int8 repack runs on 1 vCPU and 2 GiB, at a median 133 ms per query and USD 27.67 per million documents
- 🟢 The L4 costs USD 1.69 per million documents at its best client rate of 171.8 documents per second
- 🟢 The L4 is cheaper for bursts above 5,307 documents; the 1 vCPU service is cheaper below
- 🟢 L4 cold starts took 17.1 to 38.9 seconds over five samples, with the model load under 7 seconds
- ⚠️ Each isolated L4 burst was billed 265 to 621 seconds, a mean of 474, about USD 0.14
- ⚠️ 512-token documents at a batch of 32 need 3,551 MiB; at 2 GiB they need a batch of 8 or fewer
- ⚠️ An L4 kept warm all month costs USD 764, the 1 vCPU price of 27.6 million documents
- ❌ The int4 repack is 28% smaller and 1.52x slower per query on CPU, and needs 3 GiB
- ❌ 1 GiB is too small for any batch size

Scope: one Google Cloud project, CPU services in `us-central1` and the L4 in `us-east4`, Cloud Run prices from the Billing Catalog on 2026-10-10, measured 2026-10-15. Memory was measured locally on a laptop at one thread, not inside Cloud Run. The int8 and int4 services ran three benchmark runs each, alternating. The L4 sweep ran once, with eight requests per cell, from one client outside Google Cloud. The cold start series stopped after five of eight samples, each cycle a single request after 20 minutes idle; the billed time per burst comes from those five cycles. The break-even assumes each burst runs at the sweep's best client rate.

The strategy for sizing EmbeddingGemma 2 on Cloud Run was validated with an incremental step by step approach.

---

#### References

- Repository: https://github.com/xbill9/embedding-cloudrun
- First article in this series: https://dev.to/gde/embeddinggemma-2-on-cloud-run-an-int8-cpu-service-against-an-nvidia-l4-15ac
- EmbeddingGemma 2 on Hugging Face: https://huggingface.co/google/embeddinggemma-2
- int8 repack: https://huggingface.co/xbill9/embeddinggemma-2-w8a8-ct-text-emb4
- int4 repack: https://huggingface.co/xbill9/embeddinggemma-2-w4a16-ct-text-emb4
- Cloud Run GPU support: https://cloud.google.com/run/docs/configuring/services/gpu
- Cloud Run billing settings: https://cloud.google.com/run/docs/configuring/billing-settings
- Cloud Run pricing: https://cloud.google.com/run/pricing
- Cloud Run metrics: https://cloud.google.com/monitoring/api/metrics_gcp_p_z#gcp-run
