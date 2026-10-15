"""Load this checkpoint for CPU inference with Sentence Transformers, no GPU and no compressed-tensors.

    from cpu_load import load
    model = load("path/to/embeddinggemma-2-text-w8a8-int8-emb4")
    model.encode("What causes the northern lights?", prompt_name="SearchQuery")

Most int8 Linears run as PyTorch fbgemm dynamic Linears built from the stored int8 weights and
per-channel scales, unchanged: activations are quantized per tensor at run time, the int8 matmul
runs in fbgemm, and the result comes back as float32. `reduce_range=True` (the PyTorch default
on x86) quantizes activations to 7 bits so AVX2 kernels cannot overflow.

The Linears named in FLOAT_INPUTS keep float32 activations: their int8 weights are dequantized
once at load. A `down_proj` input has channels up to ~700x its RMS, which one 7-bit scale per
tensor cannot hold; quantizing it alone takes the mean cosine to the float32 original from 0.9999
to 0.960. With `down_proj` and `per_layer_projection` in float the repack measures 0.995. `embed_tokens` stays
packed int4 in memory and only the looked-up rows are unpacked. Everything else is float32.
"""

import json
import os

import torch
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules.transformer import Transformer
from sentence_transformers.sentence_transformer.modules import Normalize, Pooling
from transformers import AutoModel
from transformers.initialization import no_init_weights

FLOAT_INPUTS = ("down_proj", "per_layer_projection")
_SHIFTS = torch.arange(0, 32, 4, dtype=torch.int32)


class Int4Embedding(torch.nn.Module):
    """Group-32 int4 table, `weight_packed` int32 [rows, cols / 8] (nibble i of word j = level + 8
    for column 8j + i) and fp16 `weight_scale` [rows, cols / 32], times the model's embed_scale."""

    def __init__(self, packed, scale, shape, embed_scale):
        super().__init__()
        self.register_buffer("weight_packed", packed)
        self.register_buffer("weight_scale", scale)
        self.num_embeddings, self.embedding_dim = (int(x) for x in shape)
        self.register_buffer("embed_scale", torch.as_tensor(embed_scale, dtype=torch.float32))

    def forward(self, input_ids):
        p = self.weight_packed[input_ids]  # [..., cols / 8]
        q = ((p.unsqueeze(-1) >> _SHIFTS) & 0xF).flatten(-2).to(torch.float32) - 8.0
        s = self.weight_scale[input_ids].to(torch.float32).repeat_interleave(32, dim=-1)
        return q * s * self.embed_scale


def unpack_int4(packed, scale, shape):
    """Group-32 int4 `weight_packed` / `weight_scale` / `weight_shape` -> float32 [rows, cols]."""
    q = ((packed.unsqueeze(-1) >> _SHIFTS) & 0xF).flatten(-2).to(torch.float32) - 8.0
    return q * scale.to(torch.float32).repeat_interleave(32, dim=-1)


class Int8Linear(torch.nn.Module):
    def __init__(self, q, scale, reduce_range=True):
        super().__init__()
        w = torch._make_per_channel_quantized_tensor(
            q, scale.to(torch.float32).flatten().to(torch.float64), torch.zeros(q.shape[0], dtype=torch.int64), 0)
        self._packed = torch.ops.quantized.linear_prepack(w, None)
        self.in_features, self.out_features = q.shape[1], q.shape[0]
        self.reduce_range = reduce_range

    def forward(self, x):
        return torch.ops.quantized.linear_dynamic(x, self._packed, reduce_range=self.reduce_range)


class DequantLinear(torch.nn.Module):
    """int8 weight times its per-channel scale (or an int4 weight, see unpack_int4), dequantized
    once; float32 activations."""

    def __init__(self, q, scale, weight=None):
        super().__init__()
        if weight is None:
            weight = q.to(torch.float32) * scale.to(torch.float32)
        self.register_buffer("weight", weight, persistent=False)
        self.out_features, self.in_features = weight.shape

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


def _set(root, name, module):
    parent, _, leaf = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, leaf, module)


def build_model(path, config, reduce_range=True, float_inputs=FLOAT_INPUTS):
    config.quantization_config = None
    del config.quantization_config
    sd = load_file(os.path.join(path, "model.safetensors"))
    with no_init_weights():
        model = AutoModel.from_config(config, dtype=torch.bfloat16)
    for name in [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]:
        if name + ".weight_packed" in sd:  # W4A16
            w = unpack_int4(sd.pop(name + ".weight_packed"), sd.pop(name + ".weight_scale"),
                            sd.pop(name + ".weight_shape"))
            _set(model, name, DequantLinear(None, None, w))
        elif name + ".weight_scale" in sd:
            q, scale = sd.pop(name + ".weight"), sd.pop(name + ".weight_scale")
            float_in = name.rsplit(".", 1)[-1] in float_inputs
            _set(model, name, DequantLinear(q, scale) if float_in else Int8Linear(q, scale, reduce_range))
    emb = model.get_input_embeddings()
    _set(model, "language_model.embed_tokens", Int4Embedding(
        sd.pop("language_model.embed_tokens.weight_packed"), sd.pop("language_model.embed_tokens.weight_scale"),
        sd.pop("language_model.embed_tokens.weight_shape"), emb.embed_scale.item()))
    model.to(torch.float32)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("language_model.embed_tokens.")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.eval()


class CpuInt8Transformer(Transformer):
    reduce_range = True
    float_inputs = FLOAT_INPUTS

    def _load_model(self, model_name_or_path, transformer_task, config, backend, is_peft_model, **model_kwargs):
        return build_model(model_name_or_path, config, self.reduce_range, self.float_inputs)


def load(path, reduce_range=True, float_inputs=FLOAT_INPUTS, **kwargs):
    CpuInt8Transformer.reduce_range = reduce_range
    CpuInt8Transformer.float_inputs = tuple(float_inputs)
    st_cfg = json.load(open(os.path.join(path, "config_sentence_transformers.json")))
    modules = [CpuInt8Transformer(path), Pooling.load(os.path.join(path, "1_Pooling")), Normalize()]
    return SentenceTransformer(modules=modules, device="cpu", prompts=st_cfg["prompts"],
                               similarity_fn_name=st_cfg.get("similarity_fn_name"), **kwargs)
