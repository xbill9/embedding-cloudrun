---
title: "EmbeddingGemma 2 on Cloud Run: an int8 CPU Service Against an NVIDIA L4"
published: false
description: "Step by step deployment of Google's EmbeddingGemma 2 text embeddings to Cloud Run, on a 1 vCPU instance with an int8 repack and on an NVIDIA L4, with tests, a semantic search demo, benchmarks and a cost per million documents."
tags: googlecloud, gemma, cloudrun, machinelearning
cover_image: https://raw.githubusercontent.com/xbill9/embedding-cloudrun/main/docs/articles/embeddinggemma-cloudrun/devto-cover.a2390bee.jpg
---

This article provides a step by step deployment guide for EmbeddingGemma 2 text embeddings to Google Cloud Run, on a CPU-only instance and on an NVIDIA L4 GPU. Each service is tested, used in a small semantic search demo, benchmarked and priced per million queries and per million documents.

https://github.com/xbill9/embedding-cloudrun

---

#### What is this project trying to Do?

This project serves `google/embeddinggemma-2` behind one HTTP endpoint on Cloud Run and answers a cost question: what does it take to embed text with it, on a 1 vCPU instance or on an L4?

Cloud Run runs a container image as an HTTPS service. It starts instances when requests arrive, stops them when traffic stops, and needs no cluster, VM or load balancer. A CPU service can be billed per request: CPU and memory are charged only while a request is in flight. A GPU service is billed per instance: the L4 is charged for every second the instance is up.

An L4 instance costs about ten times as much per hour as a 1 vCPU instance. So the comparison turns on two numbers: how much faster the L4 embeds, and how much of its time it spends busy.

---

#### Why Run Your Own Embeddings?

Hosted decision APIs now sell work that embedding models have done for years. TypeSafe's Jev, launched on September 15, 2026, answers typed questions about a piece of text: pick one label from a list, place the text on a scale, or return the probability that a statement is true. It is priced at $0.042 per million input tokens, with access through a waitlist, OpenRouter and Vercel's AI Gateway.

Ranking passages against a query and labelling text by its closest description are what an embedding model is built for. EmbeddingGemma 2 is an open model, so it can run in your own Google Cloud project, at Cloud Run's rates, with the text never leaving the project.

This article prices that option per million input tokens, with the same unit Jev uses, and measures its speed and what it takes to deploy. Accuracy on the same items as Jev is not measured here; that head-to-head is the subject of a follow-up.

---

#### What is EmbeddingGemma 2?

EmbeddingGemma 2 turns text into a 768-number vector, so that texts with similar meaning land close together. It also carries vision and audio encoders, which this project drops: only the text towers are loaded.

The vectors are Matryoshka vectors. The first 128, 256 or 512 numbers of the 768 are themselves usable embeddings, so a search index can trade a little quality for a smaller footprint. The model also takes a prompt name: `SearchQuery` for a question and `Document` for a passage to be searched.

---

#### Three Services, One Image

| service | weights | instance | billing |
|---|---|---|---|
| `emb-cpu-repack` | text only, int8 Linears, int4 token table, 203 MiB | 1 vCPU, 2 GiB | per request |
| `emb-cpu-original` | text only, float32 | 1 vCPU, 4 GiB | per request |
| `emb-l4-original` | text only, bf16 | 4 vCPU, 16 GiB, 1 L4 | per instance |

The repack is built by `repack/repack_text_w8a8_emb4.py` in the repository: every Linear in the text model stored as int8 with one scale per output channel, and the 262,144-row token table as int4. On CPU the int8 Linears run on PyTorch's fbgemm kernels, so the L4 serves the original weights in bf16.

4 vCPU and 16 GiB is the smallest instance Cloud Run attaches an L4 to.

---

#### Where do I start?

The strategy for starting the deployment is an incremental step by step approach.

First, the model files are staged and the server is checked on the local machine. Then each variant is built into an image, deployed, tested against its own contract and against the original weights, used for a search demo, timed, and priced from the Cloud Billing catalog.

---

#### At This Point You Should Have…

- A Google Cloud project with billing, and the `gcloud` CLI signed in
- The Cloud Run, Artifact Registry and Cloud Build APIs enabled
- Docker, for the CPU images
- Python 3.12 or newer with `torch`, `transformers` and `sentence-transformers` for the local check and the repack
- For the L4: Cloud Run GPU quota, without zonal redundancy, in the region you deploy to

---

#### Setup the Basic Environment

```shell
git clone https://github.com/xbill9/embedding-cloudrun
cd embedding-cloudrun
python3 -m pip install torch==2.14.1 transformers==5.19.0 sentence-transformers==6.1.0 huggingface_hub
```

The scripts that talk to a deployed service (`test_service.py`, `demo.py`, `bench.py`, `cost.py`) use only the Python standard library.

---

#### Step 1 — Stage the Models

```shell
scripts/prepare_models.sh
```

The script downloads a snapshot of `google/embeddinggemma-2` into `models/original`, then builds the repack from it into `models/repack`:

```plaintext
dropped 473.4M tower parameters; checkpoint 203.4 MiB
```

```plaintext
1.5G	models/original
235M	models/repack
```

The repack takes about 8 seconds on a laptop CPU and is deterministic: a second build writes a `model.safetensors` that is byte for byte the same. `models/` is gitignored; the Dockerfile copies one of the two directories into the image.

---

#### Step 2 — The Server

`app/server.py` is one FastAPI file with two routes. `MODEL_VARIANT` picks the weights, and the model loads before uvicorn binds the port, so Cloud Run's startup time includes it:

```python
@app.post("/embed")
def embed(r: EmbedRequest):
    if r.dim not in (128, 256, 512, 768):
        return {"error": "dim must be 128, 256, 512 or 768"}
    t = time.perf_counter()
    e = model.encode(r.texts, prompt_name=r.prompt_name, truncate_dim=r.dim,
                     normalize_embeddings=True, batch_size=32)
    return {"embeddings": e.tolist(), "seconds": time.perf_counter() - t}
```

Every reply carries `seconds`, the server's own time spent embedding. The benchmark and the cost model use that figure, so network distance between the client and the region does not enter the price.

One request holds 1 to 256 texts. Vectors come back unit length, so a dot product is the cosine similarity.

---

#### 🔎 Tip: Cloud Run Reserves Paths Ending in z

The conventional health route is `/healthz`. On Cloud Run a request for it never reaches the container: paths ending in `z` are reserved, and the front end answers 404. The health route here is `/health`.

---

#### Step 3 — Build and Deploy a CPU Service

```shell
scripts/deploy.sh cpu-repack
```

The script builds the image locally, pushes it to Artifact Registry and deploys:

```shell
gcloud run deploy emb-cpu-repack --image $IMAGE --region us-central1 \
  --allow-unauthenticated --min-instances 0 --max-instances 1 --cpu-boost \
  --cpu 1 --memory 2Gi --concurrency 1 --cpu-throttling \
  --set-env-vars TORCH_THREADS=1
```

```plaintext
Service [emb-cpu-original] revision [emb-cpu-original-00001-r96] has been deployed and is serving 100 percent of traffic.
Service URL: https://emb-cpu-original-1056842563084.us-central1.run.app

real	6m17.699s
```

The output above is the float32 service, `scripts/deploy.sh cpu-original`, from build to serving. The flags that matter:

- `--cpu-throttling` selects request-based billing: CPU is allocated, and charged, only during a request.
- `--min-instances 0` lets the service scale to zero, so an idle service costs nothing.
- `--max-instances 1` caps a runaway client at one instance.
- `--concurrency 1` sends one request at a time to the single vCPU.
- `TORCH_THREADS=1` matters because the container sees every core on the host while Cloud Run limits it to one vCPU by quota. PyTorch would otherwise start a thread per host core.

Compressed in Artifact Registry, model included, the repack image is 636 MB and the float32 image 1270 MB.

---

#### Step 4 — Deploy the L4 Service

The CUDA image is 4092 MB compressed in Artifact Registry, so `deploy.sh` builds it on Cloud Build instead of on local disk:

```shell
SERVICE_REGION=us-east4 scripts/deploy.sh l4-original
```

```shell
gcloud run deploy emb-l4-original --image $IMAGE --region us-east4 \
  --allow-unauthenticated --min-instances 0 --max-instances 1 --cpu-boost \
  --cpu 4 --memory 16Gi --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
  --no-cpu-throttling --concurrency 4 --set-env-vars TORCH_THREADS=4
```

```plaintext
Service [emb-l4-original] revision [emb-l4-original-00001-shk] has been deployed and is serving 100 percent of traffic.
Service URL: https://emb-l4-original-1056842563084.us-east4.run.app

real	2m30.320s
```

`--no-gpu-zonal-redundancy` gives up capacity reserved in a second zone for failover, at 0.0001867 USD per L4-second against 0.0002909 with it.

---

#### 🔎 Tip: GPU Quota Is Per Region

The image lives in `us-central1`, and the same deploy there was refused:

```plaintext
ERROR: (gcloud.run.deploy) spec.template.metadata.annotations[autoscaling.knative.dev/maxScale]: You do not have quota for using GPUs without zonal redundancy. Learn more about GPU zonal redundancy: g.co/cloudrun/gpu-redundancy-help
```

The quota is set per region and per redundancy mode. List it before choosing a region:

```shell
gcloud beta quotas info describe NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion \
  --service=run.googleapis.com --format=json
```

```plaintext
{'region': 'europe-west1'} {'value': '3'}
{'region': 'europe-west4'} {'value': '3'}
{'region': 'us-central1'} {}
{'region': 'us-east4'} {'value': '3'}
```

A service in one region pulls an image from Artifact Registry in another, so only `SERVICE_REGION` changes. The Cloud Billing catalog lists the same Cloud Run CPU, memory and L4 rates for `us-central1` and `us-east4`.

---

#### Step 5 — Test the Deployment

`scripts/test_service.py` checks one service against the contract the server promises: health, vector length and unit norm at every Matryoshka size, the 128-number vector being the first 128 of the 768, bad input rejected, and a relevant passage ranked above two unrelated ones.

```shell
python3 scripts/test_service.py https://emb-cpu-repack-fgasxpwzoq-uc.a.run.app
```

```plaintext
✅ health: {"variant": "repack", "device": "cpu", "threads": 1, "load_s": 9.03, "max_seq_length": 512}
✅ dim 128: length 128, norm 1.0000
✅ dim 256: length 256, norm 1.0000
✅ dim 512: length 512, norm 1.0000
✅ dim 768: length 768, norm 1.0000
✅ Matryoshka prefix: cosine(first 128 of 768, dim 128) = 1.00000
✅ rejects dim 100: {"error": "dim must be 128, 256, 512 or 768"}
✅ rejects empty texts: HTTP 422
✅ ranking: scores 0.776, 0.511, 0.642
9/9 checks passed
```

The ranking check asks "How do plants make food from sunlight?" against a passage on photosynthesis, one on the Treaty of Westphalia and one on sourdough. The float32 and L4 services pass the same nine checks:

```plaintext
✅ health: {"variant": "original", "device": "cuda:0", "threads": 4, "load_s": 6.87, "max_seq_length": 512}
✅ dim 128: length 128, norm 1.0000
...
9/9 checks passed
```

---

#### 🔎 Tip: Normalize bf16 Embeddings in float32

Sentence Transformers' `normalize_embeddings=True` normalizes in the model's own precision. On the L4 the model runs in bf16, and vectors that should be unit length came back with norms of 0.9975 at 128 dimensions and 1.0015 at 512, so a dot product used as a cosine is off by the same amount. Loading the same weights in bf16 on a laptop CPU gives the same spread, 0.9977 to 1.0012.

The server asks for the raw vectors and normalizes them in float32:

```python
e = model.encode(r.texts, prompt_name=r.prompt_name, truncate_dim=r.dim,
                 convert_to_tensor=True, batch_size=32)
e = torch.nn.functional.normalize(e.float(), dim=-1)
```

With that change every norm prints as 1.0000 on all three services.

---

#### Step 6 — Does the Repack Match the Original?

`--against` embeds the same four texts on a second service and compares the vectors:

```shell
python3 scripts/test_service.py $REPACK_URL --against $ORIGINAL_URL
```

```plaintext
✅ agrees with --against: cosine per text min 0.9888 max 0.9978
10/10 checks passed
```

The L4 in bf16 and the CPU in float32 run the same original weights, and agree almost exactly:

```shell
python3 scripts/test_service.py $L4_URL --against $ORIGINAL_URL
```

```plaintext
✅ agrees with --against: cosine per text min 0.9999 max 1.0000
10/10 checks passed
```

The repack against the L4 measured 0.9858 to 0.9981. The repack's vectors shift slightly from one Cloud Run host to the next, because its int8 kernels depend on the host CPU, so the same check gives 0.986 to 0.998 across runs.

A cosine near 0.99 says the vectors point the same way. Retrieval is the use that matters, so the repack was also scored on SciFact (300 queries over 5,183 scientific abstracts, 768 dimensions) with `repack/eval_scifact.py`:

| weights | nDCG@10 | recall@10 | top 10 shared with float32 |
|---|---|---|---|
| 🥇 float32 original | 0.854 | 0.944 | 100% |
| 🥈 int4 token table only | 0.850 | 0.939 | 92.7% |
| 🥉 repack (int8 + int4) | 0.836 | 0.926 | 85.5% |

The repack keeps 97.9% of the original's nDCG@10.

---

#### Step 7 — A Simple Demo: Semantic Search

`scripts/demo.py` embeds the sixteen passages in `docs/corpus.txt` as documents and the question as a search query, then prints the closest three:

```shell
python3 scripts/demo.py $REPACK_URL "Why does bread rise?"
```

```plaintext
16 passages, dim 768, server 2.59 s, wall 2.99 s
0.853  Bread dough rises because yeast releases carbon dioxide that the gluten network traps.
0.698  A sourdough starter is a mixture of flour and water fermented by wild yeast and lactic acid bacteria.
0.638  The northern lights appear when charged particles from the sun excite gases in the upper atmosphere.
```

```shell
python3 scripts/demo.py $REPACK_URL "Which old document helped decode Egyptian writing?"
```

```plaintext
16 passages, dim 768, server 2.63 s, wall 3.02 s
0.745  The Rosetta Stone carries the same decree in hieroglyphic, Demotic and Ancient Greek scripts.
0.631  Matryoshka embeddings can be truncated to a shorter prefix and still be used for retrieval.
0.595  A hash map stores key-value pairs and finds a key in constant time on average.
```

The second question shares no keywords with its answer: "Rosetta Stone" and "hieroglyphic" appear only in the passage. The L4 returns the same ranking:

```shell
python3 scripts/demo.py $L4_URL "Which old document helped decode Egyptian writing?"
```

```plaintext
16 passages, dim 768, server 0.13 s, wall 0.42 s
0.749  The Rosetta Stone carries the same decree in hieroglyphic, Demotic and Ancient Greek scripts.
0.634  Matryoshka embeddings can be truncated to a shorter prefix and still be used for retrieval.
0.592  A hash map stores key-value pairs and finds a key in constant time on average.
```

All three services put the same passage first for all three demo questions.

---

#### Step 8 — Cold Start

With `--min-instances 0`, the first request after an idle period starts an instance: Cloud Run places it, pulls the image and starts the container, and the model loads before the port opens. `scripts/coldstart.sh` times that first request, then a second one, after 25 minutes with no traffic:

```shell
scripts/coldstart.sh $L4_URL
```

```plaintext
2026-10-12T17:46:53Z
{"variant":"original","device":"cuda:0","threads":4,"load_s":4.78,"max_seq_length":512}
http=200 total_s=17.271792
second http=200 total_s=0.098748
```

`scripts/coldstart_summary.py` splits each first request into the server's model load and the rest:

```plaintext
file                                   variant    first request s  model load s  the rest s  warm s
coldstart-cpu-original.txt             original              48.0          15.5        32.5    0.13
coldstart-cpu-repack-2.txt             repack                58.4           9.5        48.9    0.11
coldstart-cpu-repack.txt               repack                83.5          13.8        69.7    0.13
coldstart-l4-original.txt              original              17.3           4.8        12.5    0.10
```

The model load is the smaller part on every service. The CPU services took 48 to 84 seconds to answer a first request, and the L4 took 17, with the largest image of the three. Cloud Run's request log for the float32 cold start shows the order: instance started at 17:08:08, first line from the container at 17:08:49, startup check passed at 17:08:56.

A service that cannot wait that long can keep one instance warm with `--min-instances 1`; Cloud Run pricing lists how idle instances are billed.

---

#### Step 9 — Benchmark

`scripts/bench.py` sends 50 single-query requests, then 10 requests of 32 documents of 141 tokens each (`scripts/count_tokens.py`, prompt included), and records the server's own time for each:

```shell
python3 scripts/bench.py cpu-repack $REPACK_URL
```

Each run is kept in `results/runs/`. A 1 vCPU instance lands on whichever host Cloud Run picks, so each variant was run three times, each on a new revision and so a new instance. `cost.py` reports the median run and the range:

```plaintext
variant        runs          query ms            docs/s  USD / 1M queries       USD / 1M docs
cpu-original      3     251 (227-251)     0.9 (0.7-0.9)              8.93 39.05 (38.37-47.72)
cpu-repack        3      121 (88-128)     1.1 (1.0-1.6)              3.91 25.87 (18.17-27.69)
l4-original       3        56 (54-59) 333.1 (322.7-339.1)             16.27    0.87 (0.86-0.90)
```

`scripts/ratios.py` turns the medians into ratios:

```plaintext
L4 docs/s over cpu-repack docs/s: 297x
cpu-original query time over cpu-repack query time: 2.08x
cpu-repack docs/s over cpu-original docs/s: 1.29x
SciFact nDCG@10, repack over float32: 97.9%
```

On the same 1 vCPU, the repack answers a query in half the time of float32 and embeds documents 1.29x as fast, in half the memory. The L4 embeds 297 times as many documents per second as the repack.

---

#### Step 10 — Cost Analysis

`scripts/fetch_prices.py` reads the Cloud Run rates from the Cloud Billing Catalog API into `prices.json`, and `scripts/cost.py` does all the arithmetic from those rates and the benchmark runs:

```shell
python3 scripts/cost.py results/runs/*.json
```

```plaintext
prices fetched 2026-10-10, us-central1

variant        billing    USD/hour up
cpu-repack     request         0.1044
cpu-original   request         0.1224
l4-original    instance        1.0465
```

A per-request CPU service is charged its server seconds times its per-second rate, plus 0.40 USD per million requests after the first two million each month. An idle CPU service costs nothing. The L4 is charged for every second its instance is up, so its per-document figures in Step 9 hold only while it is kept busy.

---

#### And Price/Performance?

An instance-billed L4 that is busy part of the time costs more per document in proportion. `cost.py` prints the curve against the repack:

```plaintext
L4 cost per 1M docs when the instance is busy part of the time it is up (vs cpu-repack, median runs):
  busy  L4 USD / 1M docs   cpu-repack USD / 1M docs
  100%              0.87                      25.87
   50%              1.75                      25.87
   25%              3.49                      25.87
   10%              8.73                      25.87
    5%             17.46                      25.87
    1%             87.28                      25.87
break-even: the L4 is cheaper per document while busy more than 3.4% of the time it is up
```

Per million input tokens, the unit Jev is priced in:

```plaintext
USD per 1M input tokens, documents of 141 tokens (median runs):
cpu-original     0.2770
cpu-repack       0.1835
l4-original      0.0062  at 100% busy
l4-original      0.0619  at 10% busy
jev (hosted)     0.0420  list price
l4-original matches Jev's price per input token while busy 14.7% of the time it is up; at 100% busy Jev costs 6.8x as much
cpu-original costs 6.6x Jev's price per input token
cpu-repack costs 4.4x Jev's price per input token
```

A busy L4 embeds text for a sixth of Jev's list price per input token. The 1 vCPU repack costs 4.4 times Jev's price per token, and nothing at all while idle.

---

#### Compare and Contrast

| | cpu-repack | cpu-original | l4-original |
|---|---|---|---|
| Query, median | 🥈 121 ms | 🥉 251 ms | 🥇 56 ms |
| Documents per second | 🥈 1.1 | 🥉 0.9 | 🥇 333 |
| USD per 1M queries | 🥇 3.91 | 🥈 8.93 | 🥉 16.27 |
| USD per 1M documents, busy | 🥈 25.87 | 🥉 39.05 | 🥇 0.87 |
| Cost while idle | 🥇 none | 🥇 none | 🥉 1.05 USD per hour up |
| Cold start | 🥉 58–84 s | 🥈 48 s | 🥇 17 s |
| GPU quota needed | 🥇 no | 🥇 no | 🥉 yes |
| SciFact nDCG@10 | 🥈 0.836 | 🥇 0.854 | not run; cosine 0.9999 to float32 |

---

#### So, Which One?

For search queries arriving through the day, use the CPU repack. It is the cheapest per query, costs nothing between requests, needs no GPU quota, and keeps 97.9% of the original's retrieval score.

For indexing a corpus, use the L4. It embeds 297 times as fast and costs 0.87 USD per million documents while busy; it pays for itself over the repack once it is busy more than 3.4% of the time it is up. Deploy it for the job and let it scale to zero afterwards.

The float32 CPU service is the reference that the other two are checked against.

---

#### Step 11 — Teardown

```shell
scripts/down.sh
```

It deletes the three services in both regions. The images stay in Artifact Registry until the repository is deleted:

```shell
gcloud artifacts repositories delete embedding-cloudrun --location us-central1
```

---

#### Summary

The goal of this article was to deploy EmbeddingGemma 2 text embeddings to Cloud Run and price them against a hosted decision API. The key to the solution was one server and one Dockerfile for three services: an int8 repack on 1 vCPU with per-request billing, the float32 original on 1 vCPU as the reference, and the original in bf16 on an NVIDIA L4. The deployment results were:

- 🟢 All three services pass the same nine checks, and the demo puts the same passage first on all three
- 🟢 The repack matches float32 at a cosine of 0.986 to 0.998 per text and keeps 97.9% of its SciFact nDCG@10
- 🟢 The repack answers a query in a median 121 ms on 1 vCPU, at 3.91 USD per million queries and nothing while idle
- 🟢 The L4 embeds 333 documents per second, 0.87 USD per million documents while busy, 6.8 times below Jev's list price per input token
- ⚠️ The L4 bills for every second it is up and is cheaper per document than the repack only while busy more than 3.4% of that time
- ⚠️ Cold starts took 48 to 84 seconds on CPU and 17 on the L4
- ⚠️ bf16 vectors need float32 normalization to come back unit length
- ❌ No L4 quota without zonal redundancy in `us-central1`; the L4 ran in `us-east4`

Scope: one Google Cloud project, CPU services in `us-central1` and the L4 in `us-east4`, one instance per service, Cloud Run prices from the Billing Catalog on 2026-10-10, measured 2026-10-11 and 2026-10-12. Three benchmark runs per variant, each on a new instance, with server-side timing, one document length (141 tokens) and one request at a time; the L4 was not tested with concurrent requests. Cold starts are single measurements. Jev's figure is its list price per input token; its accuracy against EmbeddingGemma 2 on the same items is not measured here.

The strategy for deploying EmbeddingGemma 2 on Cloud Run was validated with an incremental step by step approach.

---

#### References

- Repository: https://github.com/xbill9/embedding-cloudrun
- EmbeddingGemma 2 on Hugging Face: https://huggingface.co/google/embeddinggemma-2
- Cloud Run GPU support: https://cloud.google.com/run/docs/configuring/services/gpu
- Cloud Run billing settings: https://cloud.google.com/run/docs/configuring/billing-settings
- Cloud Run pricing: https://cloud.google.com/run/pricing
- Sentence Transformers: https://sbert.net/
- SciFact in BEIR: https://huggingface.co/datasets/mteb/scifact
- TypeSafe Jev: https://typesafe.ai
- Jev after eight days of independent tests: https://dev.to/aws-builders/jev-after-eight-days-of-independent-tests-level-with-mid-price-llms-behind-the-frontier-1c60
