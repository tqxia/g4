"""Phase A1 — weights tour.

Reads the downloaded Gemma 4 26B A4B checkpoint and prints a structured map of
every tensor: which category it belongs to, its shape, its dtype, and its
byte size.  The checkpoint ships as a multimodal (``gemma4``) bundle, so the
tour also flags tensors that belong to the vision tower or other non-text
components — those are dropped at load time when we only need the language
model.

Sources of information:

* ``config.json``              — hyperparameters (has a ``text_config`` sub-
  dict for the language model and a ``vision_config`` sub-dict).
* ``model.safetensors.index.json`` — full manifest of tensor names with the
  shard each lives in. This is enough to know the *logical* weight
  layout without loading any of the big shard files.
* The safetensors headers    — opened individually to read per-tensor shape
  and dtype. Headers are small (first few KB of each shard) so this tour
  is cheap even with the shards not yet fully downloaded; we gracefully
  skip tensors whose shard isn't on disk yet.

Run:

    python -m mlx_ref.tour_weights
    python -m mlx_ref.tour_weights --full        # list every tensor

The default output is a summarized view, counting tensors by group. ``--full``
prints every tensor. Either mode also prints the Gemma-4-specific knobs we
care about later (two head dims, p-RoPE on global layers, 5-sliding/1-global
interleave pattern, final logit softcapping, etc.).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .paths import require_model_dir


# ---------------------------------------------------------------------------
# safetensors header reader.
#
# A safetensors file starts with an 8-byte little-endian u64 header length,
# followed by that many bytes of UTF-8 JSON. We only read the header, never
# the tensor data. See https://github.com/huggingface/safetensors for the
# format spec.
# ---------------------------------------------------------------------------


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise RuntimeError(f"{path} is truncated (no header length)")
        (hdr_len,) = struct.unpack("<Q", raw)
        blob = f.read(hdr_len)
        if len(blob) < hdr_len:
            raise RuntimeError(
                f"{path} is truncated (header says {hdr_len} bytes, got {len(blob)})"
            )
        return json.loads(blob)


# Sizes in bytes for each safetensors dtype string we expect to see.
DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1,
    "BOOL": 1,
}


def tensor_nbytes(shape: list[int], dtype: str) -> int:
    n = 1
    for d in shape:
        n *= d
    return n * DTYPE_BYTES.get(dtype, 0)


# MLX quantizes linear weights by packing N 4-bit (or 8-bit) values into a
# single ``uint32`` along the *input* axis. The ``.weight`` tensor is stored
# with that last axis divided by the pack factor, so to get a shape a human
# can reason about we undo the packing and annotate the logical bit-width.
#
# The checkpoint config pins the bit-width per tensor (default 4, overrides
# for dense MLP / router at 8). That information is read from ``config.json``
# by the caller and passed in via ``bits_lookup``.
PACK_PER_U32 = {4: 8, 8: 4}


def logical_weight_shape(
    name: str, packed_shape: list[int], dtype: str, bits_lookup: dict[str, int] | None
) -> tuple[list[int], int | None]:
    """Return ``(logical_shape, bits)`` for a ``.weight`` tensor, or
    ``(packed_shape, None)`` if the tensor is not quantized."""
    if dtype != "U32" or not name.endswith(".weight") or not packed_shape:
        return packed_shape, None
    bits = (bits_lookup or {}).get(_quant_key(name), 4)
    pack = PACK_PER_U32.get(bits)
    if pack is None:
        return packed_shape, bits
    return packed_shape[:-1] + [packed_shape[-1] * pack], bits


def _quant_key(name: str) -> str:
    # Config keys name layers directly, e.g.
    #   "language_model.model.layers.0.mlp.gate_proj"
    # which matches the tensor name without the ``.weight`` / ``.biases`` /
    # ``.scales`` suffix.
    for suffix in (".weight", ".scales", ".biases"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_bits_lookup(cfg: dict[str, Any]) -> tuple[dict[str, int], int]:
    """Collect ``{quant_key: bits}`` overrides from the config.

    Returns ``(overrides, default_bits)``. Tensors without an entry use
    ``default_bits``.
    """
    q = cfg.get("quantization") or {}
    default_bits = int(q.get("bits", 4))
    overrides: dict[str, int] = {}
    for key, val in q.items():
        if isinstance(val, dict) and "bits" in val:
            overrides[key] = int(val["bits"])
    return overrides, default_bits


# ---------------------------------------------------------------------------
# Classifier: map a tensor name to a coarse category and a Gemma-4-aware
# description. Categories are chosen so the printed summary tells a story
# about what goes where in the model; they are not used by any downstream
# code.
# ---------------------------------------------------------------------------


LAYER_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.(.*)$")


@dataclasses.dataclass(frozen=True)
class Classified:
    group: str           # high level bucket for the summary printout
    description: str     # one-line human description
    layer: int | None    # layer index if applicable, else None
    language: bool       # True if part of the text model we'll run


def classify(name: str) -> Classified:
    # Non-text components we ignore for v1.
    if name.startswith("embed_vision") or name.startswith("vision_tower"):
        return Classified("vision", "vision tower (ignored)", None, False)
    if name.startswith("embed_audio") or name.startswith("audio_tower"):
        return Classified("audio", "audio tower (ignored)", None, False)
    if name.startswith("multi_modal_projector"):
        return Classified("mm_projector", "multimodal projector (ignored)", None, False)

    # Top-level text model.
    if name == "language_model.model.embed_tokens.weight":
        return Classified("embed", "token embedding table (vocab x hidden)", None, True)
    if name.startswith("language_model.model.embed_tokens."):
        return Classified("embed", "token embedding quant metadata", None, True)
    if name == "language_model.model.norm.weight":
        return Classified("final_norm", "pre-LM-head RMSNorm scale", None, True)
    if name.startswith("language_model.lm_head."):
        # tied to embed_tokens in this checkpoint; included for completeness.
        return Classified("lm_head", "LM head (usually tied)", None, True)

    m = LAYER_RE.match(name)
    if not m:
        return Classified("other", "unrecognized top-level tensor", None, True)

    layer = int(m.group(1))
    sub = m.group(2)

    if sub == "input_layernorm.weight":
        return Classified("layer_norm", "pre-attention RMSNorm scale", layer, True)
    if sub == "post_attention_layernorm.weight":
        return Classified("layer_norm", "post-attention RMSNorm scale", layer, True)
    if sub == "pre_feedforward_layernorm.weight":
        return Classified("layer_norm", "pre-FFN RMSNorm (dense branch)", layer, True)
    if sub == "post_feedforward_layernorm.weight":
        return Classified("layer_norm", "post-FFN RMSNorm (combined)", layer, True)
    if sub == "pre_feedforward_layernorm_2.weight":
        return Classified("layer_norm", "pre-FFN RMSNorm (MoE branch)", layer, True)
    if sub == "post_feedforward_layernorm_1.weight":
        return Classified("layer_norm", "post-FFN RMSNorm (dense branch)", layer, True)
    if sub == "post_feedforward_layernorm_2.weight":
        return Classified("layer_norm", "post-FFN RMSNorm (MoE branch)", layer, True)
    if sub == "layer_scalar":
        return Classified("layer_norm", "per-layer output scalar", layer, True)

    if sub.startswith("self_attn."):
        tail = sub[len("self_attn.") :]
        return Classified("attention", f"attention {tail}", layer, True)

    if sub.startswith("mlp."):
        tail = sub[len("mlp.") :]
        return Classified("dense_mlp", f"dense (shared) MLP {tail}", layer, True)

    if sub.startswith("experts.switch_glu."):
        tail = sub[len("experts.switch_glu.") :]
        return Classified("moe_experts", f"routed experts {tail}", layer, True)

    if sub.startswith("router."):
        tail = sub[len("router.") :]
        return Classified("moe_router", f"router {tail}", layer, True)

    return Classified("other", sub, layer, True)


# ---------------------------------------------------------------------------
# Config extraction.
# ---------------------------------------------------------------------------


def summarize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pull the fields we care about out of the multimodal config.

    The model ships with ``model_type == "gemma4"`` and stashes the language
    model hyperparameters under ``text_config``.  We ignore the vision config.
    """
    tc = cfg.get("text_config", {})
    layer_types = tc.get("layer_types", [])
    pattern_counts = Counter(layer_types)

    # Detect the repeating stride, e.g. "sliding sliding sliding sliding
    # sliding full ..." -> stride 6.
    stride = None
    for k in range(1, min(len(layer_types), 12) + 1):
        if layer_types and all(layer_types[i] == layer_types[i % k] for i in range(len(layer_types))):
            stride = k
            break

    return {
        "model_type": cfg.get("model_type"),
        "text_model_type": tc.get("model_type"),
        "hidden_size": tc.get("hidden_size"),
        "num_layers": tc.get("num_hidden_layers"),
        "num_attention_heads": tc.get("num_attention_heads"),
        "num_key_value_heads": tc.get("num_key_value_heads"),
        "num_global_key_value_heads": tc.get("num_global_key_value_heads"),
        "head_dim": tc.get("head_dim"),
        "global_head_dim": tc.get("global_head_dim"),
        "attention_k_eq_v": tc.get("attention_k_eq_v"),
        "sliding_window": tc.get("sliding_window"),
        "layer_pattern_stride": stride,
        "layer_counts": dict(pattern_counts),
        "vocab_size": tc.get("vocab_size"),
        "tie_word_embeddings": tc.get("tie_word_embeddings"),
        "rms_norm_eps": tc.get("rms_norm_eps"),
        "max_position_embeddings": tc.get("max_position_embeddings"),
        "final_logit_softcapping": tc.get("final_logit_softcapping"),
        "enable_moe_block": tc.get("enable_moe_block"),
        "num_experts": tc.get("num_experts"),
        "top_k_experts": tc.get("top_k_experts"),
        "moe_intermediate_size": tc.get("moe_intermediate_size"),
        "intermediate_size": tc.get("intermediate_size"),
        "rope_parameters": tc.get("rope_parameters"),
        "dtype": tc.get("dtype"),
        "has_vision": "vision_config" in cfg and cfg["vision_config"] is not None,
        "has_audio": cfg.get("audio_config") is not None,
    }


# ---------------------------------------------------------------------------
# Pretty printing helpers.
# ---------------------------------------------------------------------------


def human_bytes(n: int) -> str:
    for unit, step in (("B", 1), ("KiB", 1024), ("MiB", 1024**2), ("GiB", 1024**3), ("TiB", 1024**4)):
        if n < step * 1024:
            return f"{n / step:7.2f} {unit}"
    return f"{n:d} B"


def print_config(cfg_summary: dict[str, Any]) -> None:
    print("=" * 72)
    print(" Config summary")
    print("=" * 72)
    rope = cfg_summary.get("rope_parameters") or {}
    rope_full = rope.get("full_attention", {})
    rope_slide = rope.get("sliding_attention", {})

    lines = [
        ("model_type",           cfg_summary["model_type"]),
        ("text model_type",      cfg_summary["text_model_type"]),
        ("dtype (on-disk)",      cfg_summary["dtype"]),
        ("hidden_size",          cfg_summary["hidden_size"]),
        ("num_hidden_layers",    cfg_summary["num_layers"]),
        ("num_attention_heads",  cfg_summary["num_attention_heads"]),
        ("num_key_value_heads",  cfg_summary["num_key_value_heads"]),
        ("num_global_kv_heads",  cfg_summary["num_global_key_value_heads"]),
        ("head_dim (sliding)",   cfg_summary["head_dim"]),
        ("head_dim (global)",    cfg_summary["global_head_dim"]),
        ("attention_k_eq_v",     cfg_summary["attention_k_eq_v"]),
        ("sliding_window",       cfg_summary["sliding_window"]),
        ("layer pattern",        f"{cfg_summary['layer_counts']} "
                                 f"(stride={cfg_summary['layer_pattern_stride']})"),
        ("vocab_size",           cfg_summary["vocab_size"]),
        ("tie_word_embeddings",  cfg_summary["tie_word_embeddings"]),
        ("rms_norm_eps",         cfg_summary["rms_norm_eps"]),
        ("max_position_embeds",  cfg_summary["max_position_embeddings"]),
        ("final_logit_softcap",  cfg_summary["final_logit_softcapping"]),
        ("MoE enabled",          cfg_summary["enable_moe_block"]),
        ("num_experts",          cfg_summary["num_experts"]),
        ("top_k_experts",        cfg_summary["top_k_experts"]),
        ("dense MLP inter.",     cfg_summary["intermediate_size"]),
        ("MoE expert inter.",    cfg_summary["moe_intermediate_size"]),
        ("RoPE (global)",        f"type={rope_full.get('rope_type')} "
                                 f"theta={rope_full.get('rope_theta')} "
                                 f"partial={rope_full.get('partial_rotary_factor')}"),
        ("RoPE (sliding)",       f"type={rope_slide.get('rope_type')} "
                                 f"theta={rope_slide.get('rope_theta')} "
                                 f"partial={rope_slide.get('partial_rotary_factor', 1.0)}"),
        ("has vision config",    cfg_summary["has_vision"]),
        ("has audio config",     cfg_summary["has_audio"]),
    ]

    for k, v in lines:
        print(f"  {k:22s}  {v}")
    print()


def print_group_summary(records: list[dict[str, Any]]) -> None:
    print("=" * 72)
    print(" Tensor counts and bytes by group")
    print("=" * 72)
    bytes_by_group: dict[str, int] = defaultdict(int)
    count_by_group: Counter[str] = Counter()
    missing_by_group: Counter[str] = Counter()
    for rec in records:
        bytes_by_group[rec["group"]] += rec["nbytes"]
        count_by_group[rec["group"]] += 1
        if rec["dtype"] is None:
            missing_by_group[rec["group"]] += 1

    total_bytes = sum(bytes_by_group.values())
    total_count = sum(count_by_group.values())
    print(f"  {'group':14s} {'count':>8s} {'missing':>8s} {'size':>14s} {'%':>6s}")
    print(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 14} {'-' * 6}")
    for group in sorted(bytes_by_group, key=bytes_by_group.get, reverse=True):
        print(
            f"  {group:14s} {count_by_group[group]:8d} "
            f"{missing_by_group[group]:8d} "
            f"{human_bytes(bytes_by_group[group]):>14s} "
            f"{100 * bytes_by_group[group] / max(total_bytes, 1):5.1f}%"
        )
    print(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 14} {'-' * 6}")
    print(
        f"  {'TOTAL':14s} {total_count:8d} {sum(missing_by_group.values()):8d} "
        f"{human_bytes(total_bytes):>14s} {100.0:5.1f}%"
    )
    if sum(missing_by_group.values()):
        print()
        print("  Note: 'missing' tensors have entries in the index but whose shard")
        print("  has not been (fully) downloaded yet. Their byte size is computed")
        print("  from the manifest dtype/shape, not from the on-disk header.")
    print()


def print_layer_0(records: list[dict[str, Any]]) -> None:
    """Dump every tensor for layer 0. This is the single best page of output
    for anyone trying to learn the Gemma 4 weight layout, so print it by
    default instead of hiding it behind --full."""
    print("=" * 72)
    print(" Layer 0 tensors (representative of every layer)")
    print("=" * 72)
    layer0 = [r for r in records if r["layer"] == 0]
    layer0.sort(key=lambda r: r["name"])
    print_tensor_rows(layer0)


def print_full(records: list[dict[str, Any]]) -> None:
    print("=" * 72)
    print(" All tensors")
    print("=" * 72)
    print_tensor_rows(sorted(records, key=lambda r: r["name"]))


def print_tensor_rows(records: Iterable[dict[str, Any]]) -> None:
    print(f"  {'name':72s}  {'dtype':>5s}  {'shape':>22s}  {'size':>12s}  group")
    print(f"  {'-' * 72}  {'-' * 5}  {'-' * 22}  {'-' * 12}  -----")
    for r in records:
        shape = "x".join(str(d) for d in r["shape"]) if r["shape"] else "?"
        # Show quantized weights as Q4/Q8 so the logical shape matches the
        # bit-width the reader sees. Non-weight siblings (.scales/.biases)
        # keep their underlying BF16 dtype.
        if r.get("bits") is not None:
            dtype = f"Q{r['bits']}"
        else:
            dtype = r["dtype"] or "?"
        print(
            f"  {r['name']:72s}  {dtype:>5s}  {shape:>22s}  "
            f"{human_bytes(r['nbytes']):>12s}  {r['group']}"
        )


# ---------------------------------------------------------------------------
# Main tour.
# ---------------------------------------------------------------------------


def build_records(
    model_path: Path, bits_lookup: dict[str, int], default_bits: int
) -> tuple[list[dict[str, Any]], int, int]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing {index_path}")
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]

    # Cache shard headers so we only read each shard's JSON once.
    shard_headers: dict[str, dict[str, Any] | None] = {}

    def header_for(shard: str) -> dict[str, Any] | None:
        if shard not in shard_headers:
            shard_path = model_path / shard
            if not shard_path.exists() or shard_path.stat().st_size < 16:
                shard_headers[shard] = None
            else:
                try:
                    shard_headers[shard] = read_safetensors_header(shard_path)
                except Exception as e:
                    print(f"! failed to read header for {shard}: {e}", file=sys.stderr)
                    shard_headers[shard] = None
        return shard_headers[shard]

    records: list[dict[str, Any]] = []
    missing_shards: set[str] = set()
    fallback_to_manifest = 0

    # Helper that prepends ``bits_lookup`` with the top-level default.
    def bits_for(name: str) -> int:
        return bits_lookup.get(_quant_key(name), default_bits)

    for name, shard in weight_map.items():
        classified = classify(name)
        entry: dict[str, Any] | None = None
        hdr = header_for(shard)
        if hdr is not None:
            entry = hdr.get(name)

        if entry is None:
            missing_shards.add(shard)
            fallback_to_manifest += 1
            packed_shape: list[int] = []
            dtype = None
            nbytes = 0
        else:
            packed_shape = list(entry.get("shape", []))
            dtype = entry.get("dtype")
            nbytes = tensor_nbytes(packed_shape, dtype or "")

        logical_shape, bits = logical_weight_shape(
            name, packed_shape, dtype or "", {name[: -len(".weight")]: bits_for(name)}
        )

        records.append(
            {
                "name": name,
                "shard": shard,
                "shape": logical_shape,
                "packed_shape": packed_shape,
                "dtype": dtype,
                "bits": bits,
                "nbytes": nbytes,
                "group": classified.group,
                "description": classified.description,
                "layer": classified.layer,
                "language": classified.language,
            }
        )

    return records, len(missing_shards), fallback_to_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="print every tensor instead of just the layer-0 dump",
    )
    args = parser.parse_args()

    model_path = require_model_dir()
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        print(f"! no config.json at {cfg_path}", file=sys.stderr)
        return 2

    cfg = json.loads(cfg_path.read_text())
    cfg_summary = summarize_config(cfg)
    bits_lookup, default_bits = load_bits_lookup(cfg)
    print(f"model path:  {model_path}")
    print(f"config:      {cfg_path.name}")
    print(f"quantization: default={default_bits} bit, {len(bits_lookup)} tensor overrides")
    print()

    print_config(cfg_summary)

    records, missing_shards, fallback_to_manifest = build_records(
        model_path, bits_lookup, default_bits
    )
    if missing_shards:
        print(
            f"(note: {missing_shards} shard(s) not yet on disk — "
            f"{fallback_to_manifest} tensor shapes/dtypes unknown)"
        )
        print()

    print_group_summary(records)

    # Separate language-model vs non-language tensors for clarity.
    lang = [r for r in records if r["language"]]
    nonlang = [r for r in records if not r["language"]]
    if nonlang:
        nbytes = sum(r["nbytes"] for r in nonlang)
        print(
            f"Non-text components present: {len(nonlang)} tensors, "
            f"{human_bytes(nbytes)} (dropped when loading language-only)."
        )
        print()

    print_layer_0(lang)

    if args.full:
        print_full(records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
