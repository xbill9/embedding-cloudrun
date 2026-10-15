"""Repack google/embeddinggemma-2 as a text-only compressed-tensors checkpoint for CPU serving.

    python3 repack_text_w8a8_emb4.py SRC_DIR OUT_DIR

SRC_DIR is a snapshot of google/embeddinggemma-2 (bf16, one model.safetensors).

- Dropped: the vision and audio encoders (`vision_tower`, `audio_tower`, `embed_vision`,
  `embed_audio`), 470M of the 740M parameters. `vision_config` and `audio_config` become null.
- int8, one bf16 scale per output channel (`weight_scale` [out, 1]): every Linear in the text
  model (attention, MLP, the PLE block gates and projections, `ple.per_layer_model_projection`),
  each row scaled by max|w| / 127 and rounded to nearest. Activations are declared int8,
  asymmetric, dynamic, one scale per tensor: what PyTorch's fbgemm dynamic Linear does on x86.
- int4, group 32 along each row, fp16 scales: `embed_tokens` [262144, 512], in the same
  `weight_packed` / `weight_scale` / `weight_shape` layout as the Gemma 4 `-emb4` builds.
  This model has no QAT export, so the table is rounded, not recovered from a trained grid:
  each group's step is the candidate amax / m, m in 7.0..8.0, with the least squared error.
- Kept bf16, on the ignore list: `embedding_projection` (512 -> 768, the last layer).
- Copied byte for byte: norms, scalars.

Prints the relative error of each kind of quantized tensor against the bf16 source and writes
it to repack_report.json.
"""

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_q4_0 import GROUP, bf16_to_f32, f32_to_bf16, open_checkpoint, pack, write_safetensors  # noqa: E402

TOWERS = re.compile(r"^(vision_tower|audio_tower|embed_vision|embed_audio)\.")
EMBED = "language_model.embed_tokens.weight"
KEEP = {"language_model.embedding_projection"}
STEPS = np.linspace(7.0, 8.0, 9, dtype=np.float32)


def kind(name):
    return re.sub(r"\.\d+\.", ".", name.removeprefix("language_model.").removesuffix(".weight"))


def quantize_int8(w_u16):
    """bf16 bits [out, in] -> (int8 [out, in], bf16-bits scale [out, 1], relative error)."""
    w = bf16_to_f32(w_u16)
    amax = np.abs(w).max(axis=1, keepdims=True)
    scale = f32_to_bf16(np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32))
    s = bf16_to_f32(scale)
    q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
    err = np.linalg.norm(q.astype(np.float32) * s - w) / max(np.linalg.norm(w), 1e-30)
    return q, scale, float(err)


def _round_scale(s, tag):
    if tag == "F16":
        return s.astype(np.float16).astype(np.float32)
    return bf16_to_f32(f32_to_bf16(s.astype(np.float32)))


def quantize_int4(w_u16, chunk=8192, scale_tag="F16"):
    """bf16 bits [rows, cols] -> (packed int32 [rows, cols/8], scales [rows, cols/32], rel error).

    Scales are fp16 (scale_tag "F16") or bf16 bits ("BF16"); each candidate step is rounded to the
    stored dtype before its error is measured, so the levels match the scale that is written."""
    rows, cols = w_u16.shape
    packed = np.empty((rows, cols // 8), dtype=np.int32)
    scales = np.empty((rows, cols // GROUP), dtype=np.float16 if scale_tag == "F16" else np.uint16)
    se = sw = 0.0
    for r0 in range(0, rows, chunk):
        g = bf16_to_f32(np.asarray(w_u16[r0:r0 + chunk])).reshape(-1, cols // GROUP, GROUP)
        amax = np.abs(g).max(axis=-1, keepdims=True)
        best_s = best_e = None
        for m in STEPS:
            s = _round_scale(np.where(amax > 0, amax / m, 1.0), scale_tag)
            e = ((np.clip(np.rint(g / s), -8, 7) * s - g) ** 2).sum(axis=-1, keepdims=True)
            if best_s is None:
                best_s, best_e = s, e
            else:
                better = e < best_e
                best_s, best_e = np.where(better, s, best_s), np.where(better, e, best_e)
        q = np.clip(np.rint(g / best_s), -8, 7)
        se += float(((q * best_s - g) ** 2).sum())
        sw += float((g ** 2).sum())
        n = g.shape[0]
        packed[r0:r0 + n] = pack(q.astype(np.int8).reshape(n, cols))
        best_s = best_s.reshape(n, cols // GROUP)
        scales[r0:r0 + n] = best_s.astype(np.float16) if scale_tag == "F16" else f32_to_bf16(best_s)
    return packed, scales, (se / sw) ** 0.5


def copy_text_files(src, out):
    """Sentence Transformers files, text modality only; the source model card as ORIGINAL_README.md."""
    for f in os.listdir(src):
        path = os.path.join(src, f)
        if f in ("config.json", "README.md") \
                or f.endswith(".safetensors"):
            continue
        if os.path.isdir(path):
            shutil.copytree(path, os.path.join(out, f), dirs_exist_ok=True)
        else:
            shutil.copy2(path, out)
    sb = json.load(open(os.path.join(src, "sentence_bert_config.json")))
    sb["modality_config"] = {"text": sb["modality_config"]["text"]}
    json.dump(sb, open(os.path.join(out, "sentence_bert_config.json"), "w"), indent=2)
    shutil.copy2(os.path.join(src, "README.md"), os.path.join(out, "ORIGINAL_README.md"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src")
    p.add_argument("out")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = open_checkpoint(a.src)

    tensors, errs, dropped = [], {}, 0
    for name in sorted(ck):
        raw = np.asarray(ck[name].raw(name))
        tag = ck[name].header[name]["dtype"]
        if TOWERS.match(name):
            dropped += raw.size
            continue
        module = name.removesuffix(".weight")
        if name == EMBED:
            packed, scale, err = quantize_int4(raw)
            errs["embed_tokens (int4)"] = [err]
            tensors += [(module + ".weight_packed", "I32", packed), (module + ".weight_scale", "F16", scale),
                        (module + ".weight_shape", "I64", np.array(raw.shape, dtype=np.int64))]
        elif name.endswith(".weight") and raw.ndim == 2 and tag == "BF16" and module not in KEEP:
            q, scale, err = quantize_int8(raw)
            errs.setdefault(kind(name) + " (int8)", []).append(err)
            tensors += [(name, "I8", q), (module + ".weight_scale", "BF16", scale)]
        else:
            tensors.append((name, tag, raw))
    if not any(n.startswith("language_model.layers.") for n, _, _ in tensors):
        sys.exit("no language_model.layers.* tensors found; is SRC_DIR an embeddinggemma-2 snapshot?")

    write_safetensors(os.path.join(a.out, "model.safetensors"), tensors)
    total = os.path.getsize(os.path.join(a.out, "model.safetensors"))

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    cfg["vision_config"] = None
    cfg["audio_config"] = None
    base = {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel", "group_size": None,
            "dynamic": False, "actorder": None, "block_structure": None, "observer": None, "observer_kwargs": {}}
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {"targets": ["Linear"], "format": "int-quantized", "weights": base,
                        "input_activations": dict(base, symmetric=False, strategy="tensor", dynamic=True),
                        "output_activations": None},
            "group_embed_tokens": {"targets": ["re:.*embed_tokens$"], "format": "pack-quantized",
                                   "weights": dict(base, num_bits=4, strategy="group", group_size=GROUP),
                                   "input_activations": None, "output_activations": None},
        },
        "ignore": sorted(KEEP),
        "kv_cache_scheme": None,
        "sparsity_config": {},
    }
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)

    copy_text_files(a.src, a.out)

    report = {k: {"tensors": len(v), "rel_err_min": min(v), "rel_err_max": max(v), "rel_err_mean": sum(v) / len(v)}
              for k, v in sorted(errs.items())}
    json.dump({"source": os.path.abspath(a.src), "dropped_params": dropped, "error_vs_bf16": report,
               "total_bytes": total}, open(os.path.join(a.out, "repack_report.json"), "w"), indent=2)
    for k, r in report.items():
        print(f"{k:42s} {r['tensors']:3d} tensors, rel err {r['rel_err_min']:.4f}-{r['rel_err_max']:.4f} "
              f"(mean {r['rel_err_mean']:.4f})")
    print(f"dropped {dropped / 1e6:.1f}M tower parameters; checkpoint {total / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
