"""Repack google/embeddinggemma-2 as a text-only compressed-tensors W4A16 checkpoint.

    python3 repack_text_w4a16_emb4.py repack SRC_DIR OUT_DIR
    python3 repack_text_w4a16_emb4.py verify SRC_DIR OUT_DIR

SRC_DIR is a snapshot of google/embeddinggemma-2 (bf16, one model.safetensors).

- Dropped: the vision and audio encoders, as in repack_text_w8a8_emb4.py.
- int4, group 32 along the input dimension, bf16 scales: every Linear in the text model, in the
  `weight_packed` / `weight_scale` / `weight_shape` layout of the Gemma 4 `-w4a16-ct` builds.
  Activations stay float. This model has no QAT export, so every group is rounded, not recovered
  from a trained grid: the step is the candidate amax / m, m in 7.0..8.0, with the least squared
  error after rounding it to bf16.
- int4, group 32, fp16 scales: `embed_tokens`, exactly as in repack_text_w8a8_emb4.py.
- Kept bf16, on the ignore list: `embedding_projection` (512 -> 768, the last layer).
- Copied byte for byte: norms, scalars.

`verify` rereads both checkpoints from disk: every source text tensor must be present, copied
tensors byte-identical, packed tensors the right shape and dtype, and the relative error of each
packed tensor, recomputed from the unpacked levels and stored scales, must match the report.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_q4_0 import GROUP, bf16_to_f32, open_checkpoint, unpack, write_safetensors  # noqa: E402
from repack_text_w8a8_emb4 import EMBED, KEEP, TOWERS, copy_text_files, kind, quantize_int4  # noqa: E402

PACKED = {"F16": np.float16, "BF16": np.uint16}


def is_linear(name, raw, tag):
    return name.endswith(".weight") and raw.ndim == 2 and tag == "BF16" and name != EMBED \
        and name.removesuffix(".weight") not in KEEP


def repack(src, out):
    os.makedirs(out, exist_ok=True)
    ck = open_checkpoint(src)
    tensors, errs, dropped = [], {}, 0
    for name in sorted(ck):
        raw = np.asarray(ck[name].raw(name))
        tag = ck[name].header[name]["dtype"]
        if TOWERS.match(name):
            dropped += raw.size
            continue
        module = name.removesuffix(".weight")
        if name == EMBED or is_linear(name, raw, tag):
            scale_tag = "F16" if name == EMBED else "BF16"
            packed, scale, err = quantize_int4(raw, scale_tag=scale_tag)
            errs.setdefault(kind(name) + " (int4)", []).append(err)
            tensors += [(module + ".weight_packed", "I32", packed), (module + ".weight_scale", scale_tag, scale),
                        (module + ".weight_shape", "I64", np.array(raw.shape, dtype=np.int64))]
        else:
            tensors.append((name, tag, raw))
    if not any(n.startswith("language_model.layers.") for n, _, _ in tensors):
        sys.exit("no language_model.layers.* tensors found; is SRC_DIR an embeddinggemma-2 snapshot?")

    write_safetensors(os.path.join(out, "model.safetensors"), tensors)
    total = os.path.getsize(os.path.join(out, "model.safetensors"))

    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["vision_config"] = None
    cfg["audio_config"] = None
    w4 = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": GROUP,
          "dynamic": False, "actorder": None, "block_structure": None, "observer": None, "observer_kwargs": {}}
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {"targets": ["Linear"], "format": "pack-quantized", "weights": w4,
                        "input_activations": None, "output_activations": None},
            "group_embed_tokens": {"targets": ["re:.*embed_tokens$"], "format": "pack-quantized",
                                   "weights": w4, "input_activations": None, "output_activations": None},
        },
        "ignore": sorted(KEEP),
        "kv_cache_scheme": None,
        "sparsity_config": {},
    }
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    copy_text_files(src, out)

    report = {k: {"tensors": len(v), "rel_err_min": min(v), "rel_err_max": max(v), "rel_err_mean": sum(v) / len(v)}
              for k, v in sorted(errs.items())}
    json.dump({"source": os.path.abspath(src), "dropped_params": dropped, "error_vs_bf16": report,
               "total_bytes": total}, open(os.path.join(out, "repack_report.json"), "w"), indent=2)
    for k, r in report.items():
        print(f"{k:46s} {r['tensors']:3d} tensors, rel err {r['rel_err_min']:.4f}-{r['rel_err_max']:.4f} "
              f"(mean {r['rel_err_mean']:.4f})")
    print(f"dropped {dropped / 1e6:.1f}M tower parameters; checkpoint {total / 2**20:.1f} MiB")


def verify(src, out):
    s, o = open_checkpoint(src), open_checkpoint(out)
    report = json.load(open(os.path.join(out, "repack_report.json")))["error_vs_bf16"]
    problems, errs, packed_n, copied_n = [], {}, 0, 0
    for name in sorted(s):
        if TOWERS.match(name):
            if name in o:
                problems.append(f"{name}: tower tensor present")
            continue
        raw = np.asarray(s[name].raw(name))
        module = name.removesuffix(".weight")
        if module + ".weight_packed" in o:
            want = "F16" if name == EMBED else "BF16"
            p = np.asarray(o[module + ".weight_packed"].raw(module + ".weight_packed"))
            sh = o[module + ".weight_scale"].header[module + ".weight_scale"]
            shape = np.asarray(o[module + ".weight_shape"].raw(module + ".weight_shape")).tolist()
            rows, cols = raw.shape
            if shape != [rows, cols] or p.shape != (rows, cols // 8) or sh["dtype"] != want \
                    or sh["shape"] != [rows, cols // GROUP]:
                problems.append(f"{name}: shape/dtype {shape} {p.shape} {sh['dtype']} {sh['shape']}")
                continue
            sc = np.asarray(o[module + ".weight_scale"].raw(module + ".weight_scale"))
            sc = sc.astype(np.float32) if want == "F16" else bf16_to_f32(sc)
            se = sw = 0.0
            for r0 in range(0, rows, 8192):
                q = unpack(np.asarray(p[r0:r0 + 8192])).astype(np.float32)
                w = bf16_to_f32(np.asarray(raw[r0:r0 + 8192]))
                rec = q * np.repeat(sc[r0:r0 + 8192], GROUP, axis=1)
                se += float(((rec - w) ** 2).sum())
                sw += float((w ** 2).sum())
            errs.setdefault(kind(name) + " (int4)", []).append((se / sw) ** 0.5)
            packed_n += 1
        elif name in o:
            if o[name].header[name]["dtype"] != s[name].header[name]["dtype"] \
                    or not np.array_equal(np.asarray(o[name].raw(name)), raw):
                problems.append(f"{name}: copied tensor differs")
            copied_n += 1
        else:
            problems.append(f"{name}: missing")
    for k, v in errs.items():
        r = report.get(k)
        if r is None or r["tensors"] != len(v) or abs(max(v) - r["rel_err_max"]) > 1e-4 \
                or abs(sum(v) / len(v) - r["rel_err_mean"]) > 1e-4:
            problems.append(f"{k}: recomputed error {sum(v) / len(v):.5f} (max {max(v):.5f}) vs report {r}")
    print(f"{packed_n} packed tensors and {copied_n} copied tensors checked against the source")
    for k, v in sorted(errs.items()):
        print(f"{k:46s} {len(v):3d} tensors, rel err from disk {min(v):.4f}-{max(v):.4f} (mean {sum(v) / len(v):.4f})")
    print("\n".join(problems) if problems else "OK")
    sys.exit(1 if problems else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=["repack", "verify"])
    p.add_argument("src")
    p.add_argument("out")
    a = p.parse_args()
    (repack if a.cmd == "repack" else verify)(a.src, a.out)


if __name__ == "__main__":
    main()
