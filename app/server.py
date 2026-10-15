"""EmbeddingGemma 2 behind one HTTP endpoint, for Cloud Run on CPU or an L4.

MODEL_VARIANT=repack    the int8/int4 text-only repack through cpu_load.py (CPU only)
MODEL_VARIANT=repack-w4a16  the int4 (W4A16) text-only repack through cpu_load.py (CPU only)
MODEL_VARIANT=original  google/embeddinggemma-2, text only: bf16 on a GPU, float32 on CPU

POST /embed    {"texts": [...], "prompt_name": "SearchQuery" | "Document" | null, "dim": 768}
GET  /health   variant, device, threads, model load seconds (Cloud Run reserves paths ending in "z")

The model loads before uvicorn binds the port, so Cloud Run's startup time includes it.
"""

import os
import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

VARIANT = os.environ.get("MODEL_VARIANT", "repack")
MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
if os.environ.get("TORCH_THREADS"):
    # Cloud Run limits CPU by quota, but the container still sees every host core.
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))

t0 = time.perf_counter()
if VARIANT in ("repack", "repack-w4a16"):
    from cpu_load import load
    model = load(MODEL_DIR)
elif VARIANT == "original":
    cuda = torch.cuda.is_available()
    model = SentenceTransformer(MODEL_DIR, device="cuda" if cuda else "cpu",
                                config_kwargs={"vision_config": None, "audio_config": None},
                                model_kwargs={"dtype": torch.bfloat16 if cuda else torch.float32})
else:
    raise SystemExit(f"MODEL_VARIANT must be repack, repack-w4a16 or original, not {VARIANT!r}")
model.max_seq_length = int(os.environ.get("MAX_SEQ_LENGTH", "512"))
LOAD_S = time.perf_counter() - t0

app = FastAPI()


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)
    prompt_name: str | None = None
    dim: int = 768


@app.post("/embed")
def embed(r: EmbedRequest):
    if r.dim not in (128, 256, 512, 768):
        return {"error": "dim must be 128, 256, 512 or 768"}
    t = time.perf_counter()
    e = model.encode(r.texts, prompt_name=r.prompt_name, truncate_dim=r.dim,
                     convert_to_tensor=True, batch_size=32)
    # Normalize in float32: a bf16 model normalizes in bf16, leaving norms up to 0.25% off 1.
    e = torch.nn.functional.normalize(e.float(), dim=-1)
    return {"embeddings": e.cpu().tolist(), "seconds": time.perf_counter() - t}


@app.get("/health")
def health():
    return {"variant": VARIANT, "device": str(model.device), "threads": torch.get_num_threads(),
            "load_s": round(LOAD_S, 2), "max_seq_length": model.max_seq_length}
