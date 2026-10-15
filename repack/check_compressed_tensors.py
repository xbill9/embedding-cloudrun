"""Decode every packed tensor of a repack with compressed-tensors' own PackedQuantizationCompressor.

    python3 check_compressed_tensors.py REPACK_DIR ORIGINAL_DIR

Checks the layout against the library rather than against our own unpacking: each tensor is
decompressed by compressed-tensors under the scheme in config.json, compared with
cpu_load.unpack_int4 rounded to the library's output dtype, and its relative error against the
bf16 source is reported.
"""

import json
import os
import sys

import compressed_tensors
import torch
from compressed_tensors.compressors import PackedQuantizationCompressor
from compressed_tensors.quantization import QuantizationScheme
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from cpu_load import unpack_int4  # noqa: E402


def main():
    repack, original = sys.argv[1:3]
    groups = json.load(open(os.path.join(repack, "config.json")))["quantization_config"]["config_groups"]
    sd = load_file(os.path.join(repack, "model.safetensors"))
    src = load_file(os.path.join(original, "model.safetensors"))
    worst, errs, dtypes = 0.0, [], set()
    modules = sorted(k.removesuffix(".weight_packed") for k in sd if k.endswith(".weight_packed"))
    for m in modules:
        g = groups["group_embed_tokens" if m.endswith("embed_tokens") else "group_0"]
        scheme = QuantizationScheme(targets=g["targets"], weights=g["weights"])
        parts = {k: sd[f"{m}.{k}"] for k in ("weight_packed", "weight_scale", "weight_shape")}
        w = PackedQuantizationCompressor.decompress(parts, scheme)["weight"]
        ours = unpack_int4(parts["weight_packed"], parts["weight_scale"], parts["weight_shape"])
        worst = max(worst, (w - ours.to(w.dtype)).abs().max().item())
        dtypes.add(str(w.dtype).removeprefix("torch."))
        ref = src[f"{m}.weight"].float()
        errs.append((torch.linalg.norm(w.float() - ref) / torch.linalg.norm(ref)).item())
    print(f"{len(modules)} packed tensors decompressed by compressed-tensors {compressed_tensors.__version__}")
    print(f"max |compressed-tensors - cpu_load| in {sorted(dtypes)}: {worst}")
    print(f"relative error against the bf16 source: {min(errs):.4f}-{max(errs):.4f}")
    sys.exit(0 if worst == 0.0 else 1)


if __name__ == "__main__":
    main()
