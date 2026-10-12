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

Compressed in Artifact Registry, model included, the repack image is 636 MB and the float32 image 1.27 GB.

---

#### Step 4 — Deploy the L4 Service

The CUDA image is 4.09 GB compressed in Artifact Registry, so `deploy.sh` builds it on Cloud Build instead of on local disk:

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
✅ health: {"variant": "repack", "device": "cpu", "threads": 1, "load_s": 13.83, "max_seq_length": 512}
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

The ranking check asks "How do plants make food from sunlight?" against a passage on photosynthesis, one on the Treaty of Westphalia and one on sourdough. The float32 and L4 services pass the same nine checks.

L4_TEST_BLOCK

---

#### Step 6 — Does the Repack Match the Original?

`--against` embeds the same four texts on a second service and compares the vectors:

```shell
python3 scripts/test_service.py $REPACK_URL --against $ORIGINAL_URL
```

```plaintext
✅ agrees with --against: cosine per text min 0.9858 max 0.9981
10/10 checks passed
```

L4_AGREE_BLOCK

A cosine of 0.986 to 0.998 per text says the vectors point the same way. Retrieval is the use that matters, so the repack was also scored on SciFact (300 queries over 5,183 scientific abstracts, 768 dimensions), with `repack/eval_scifact.py`:

| weights | nDCG@10 | recall@10 | top 10 shared with float32 |
|---|---|---|---|
| float32 original | 0.854 | 0.944 | 100% |
| int4 token table only | 0.850 | 0.939 | 92.7% |
| repack (int8 + int4) | 0.836 | 0.926 | 85.5% |

The repack keeps 98% of the original's nDCG@10.

---

#### Step 7 — A Simple Demo: Semantic Search

`scripts/demo.py` embeds the sixteen passages in `docs/corpus.txt` as documents, the question as a search query, and prints the closest three:

```shell
python3 scripts/demo.py $REPACK_URL "Why does bread rise?"
```

```plaintext
16 passages, dim 768, server 2.32 s, wall 2.69 s
0.859  Bread dough rises because yeast releases carbon dioxide that the gluten network traps.
0.694  A sourdough starter is a mixture of flour and water fermented by wild yeast and lactic acid bacteria.
0.636  The northern lights appear when charged particles from the sun excite gases in the upper atmosphere.
```

```shell
python3 scripts/demo.py $REPACK_URL "Which old document helped decode Egyptian writing?"
```

```plaintext
16 passages, dim 768, server 2.36 s, wall 2.74 s
0.750  The Rosetta Stone carries the same decree in hieroglyphic, Demotic and Ancient Greek scripts.
0.641  Matryoshka embeddings can be truncated to a shorter prefix and still be used for retrieval.
0.589  A hash map stores key-value pairs and finds a key in constant time on average.
```

The second question shares no keywords with its answer: "Rosetta Stone" and "hieroglyphic" appear only in the passage. On the L4 the same search comes back in L4_DEMO_SERVER of server time:

L4_DEMO_BLOCK

---

#### Step 8 — Cold Start

With `--min-instances 0`, the first request after an idle period starts an instance: Cloud Run pulls the image, starts the container, and the model loads before the port opens.

```shell
curl -s -w '\nhttp=%{http_code} total_s=%{time_total}\n' $REPACK_URL/health
```

COLD_START_BLOCK

---

#### Step 9 — Benchmark

`scripts/bench.py` sends 50 single-query requests, then 10 requests of 32 documents of about 135 tokens each, and records the server's own time for each:

```shell
python3 scripts/bench.py cpu-repack $REPACK_URL
```

BENCH_BLOCK

---

#### Step 10 — Cost Analysis

`scripts/fetch_prices.py` reads the Cloud Run rates from the Cloud Billing Catalog API into `prices.json`, and `scripts/cost.py` does all the arithmetic from those rates and the benchmark files:

```shell
python3 scripts/cost.py results/cpu-repack.json results/cpu-original.json results/l4-original.json
```

COST_BLOCK

A per-request CPU service is charged its server seconds times its per-second rate, plus 0.40 USD per million requests after the first two million each month. The L4 rows assume an instance kept busy for every second it is up.

---

#### And Price/Performance?

An instance-billed L4 that is busy a fraction of its time costs that much more per document. `cost.py` prints the curve:

BREAKEVEN_BLOCK

---

#### Compare and Contrast

COMPARE_BLOCK

---

#### So, Which One?

SO_WHICH_BLOCK

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

SUMMARY_BLOCK

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
