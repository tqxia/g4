"""Phase A3 — hand-rolled RMSNorm + embed lookup vs the MLX reference.

We load the raw safetensors shards directly (not through ``mlx_lm.load``) and
reconstruct two things by hand:

  1. The embedding lookup for a batch of token IDs, dequantizing the 4-bit
     ``embed_tokens`` weights ourselves. This is the *first* Gemma-4-specific
     dequant the project does; Phase B's CPU and Metal paths will both have
     to reproduce it.

  2. The RMSNorm op applied by the model right before the LM head
     (``language_model.model.norm``), which is the simplest use of the norm
     formula we'll see anywhere in the model. The per-layer RMSNorms use
     the exact same math.

After each hand-rolled output, we pull the same result out of
``mlx_lm.models.gemma4_text`` and compare. If both stages' max |diff| is
below ~1e-4 (tight, given everything lives in bfloat16), the chapter
passes.

Run:

    python -m mlx_ref.norm_check
    python -m mlx_ref.norm_check --verbose   # dump per-step tensors
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mlx_lm import load

from .paths import require_model_dir


# ---------------------------------------------------------------------------
# Raw-weight loaders.
#
# We want to do the embed lookup from the on-disk quantized weights, *without*
# using any of MLX's quant helpers. The hope is that by unpacking the bits by
# hand once, we understand exactly what MLX is doing, and can replicate it in
# C later.
# ---------------------------------------------------------------------------


def load_safetensors_for(names: list[str], model_path: Path) -> dict[str, mx.array]:
    """Load just the named tensors from whichever shards they live in.

    ``mx.load`` on a safetensors file reads the header and mmaps tensor data;
    we filter by name afterward. This keeps memory use low even though our
    model spans 15 GiB.
    """
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    wanted = set(names)
    by_shard: dict[str, list[str]] = {}
    for n in wanted:
        shard = weight_map.get(n)
        if shard is None:
            raise KeyError(f"tensor {n!r} not in safetensors index")
        by_shard.setdefault(shard, []).append(n)
    out: dict[str, mx.array] = {}
    for shard, keys in by_shard.items():
        shard_contents = mx.load(str(model_path / shard))
        for k in keys:
            out[k] = shard_contents[k]
    return out


# ---------------------------------------------------------------------------
# Hand-rolled dequant.
#
# MLX affine quantization packs ``pack = 32 // bits`` values into each uint32
# along the *last* axis, then stores a per-group ``scale`` and ``bias`` in
# bfloat16. Group size for this checkpoint is 64, so every row has
# ``in_features / 64`` groups, each containing 64 consecutive scalar values.
#
# The formula to reconstruct a single scalar is:
#
#     dequantized = scale[group] * q + bias[group]
#
# where ``q`` is the raw small-integer read from the uint32. Note the sign:
# MLX's ``affine`` mode uses a bias that already accounts for the zero-point,
# so it's literally scale*q + bias -- no extra subtraction.
# ---------------------------------------------------------------------------


def dequantize_row(
    packed_row: mx.array,   # shape (in_features // pack,), uint32
    scales_row: mx.array,   # shape (in_features // group_size,), bfloat16
    biases_row: mx.array,   # shape (in_features // group_size,), bfloat16
    bits: int,
    group_size: int,
) -> mx.array:
    """Return a single dequantized row of ``in_features`` floats.

    Explicit, slow Python reference. The goal is to show every step so a
    reader can map Phase B's C code to this line-by-line.
    """
    pack = 32 // bits
    mask = (1 << bits) - 1
    in_features = int(packed_row.shape[0]) * pack

    # Unpack the uint32 words into (N_words, pack) of small ints.
    packed = packed_row.astype(mx.uint32)
    shifts = mx.arange(pack, dtype=mx.uint32) * bits
    # shape: (N_words, pack)
    q = (packed[:, None] >> shifts[None, :]) & mx.array(mask, dtype=mx.uint32)
    q = q.reshape(in_features).astype(mx.float32)

    # Broadcast scale/bias from per-group to per-scalar.
    groups = in_features // group_size
    scales = scales_row.astype(mx.float32).reshape(groups, 1)
    biases = biases_row.astype(mx.float32).reshape(groups, 1)
    # shape: (groups, group_size)
    q_g = q.reshape(groups, group_size)
    row = scales * q_g + biases
    return row.reshape(in_features)


def dequantize_matrix(
    packed: mx.array,   # (rows, in_features // pack)
    scales: mx.array,   # (rows, in_features // group_size)
    biases: mx.array,   # (rows, in_features // group_size)
    bits: int,
    group_size: int,
) -> mx.array:
    """Dequantize the whole matrix at once. Same math as row_dequantize,
    but vectorized so the check is cheap to run on 2816 embedding dims."""
    pack = 32 // bits
    mask = (1 << bits) - 1
    rows, n_words = packed.shape
    in_features = int(n_words) * pack

    shifts = mx.arange(pack, dtype=mx.uint32) * bits
    q = (packed.astype(mx.uint32)[:, :, None] >> shifts[None, None, :]) & mx.array(
        mask, dtype=mx.uint32
    )
    q = q.reshape(rows, in_features).astype(mx.float32)

    groups = in_features // group_size
    scales_f = scales.astype(mx.float32).reshape(rows, groups, 1)
    biases_f = biases.astype(mx.float32).reshape(rows, groups, 1)
    q_g = q.reshape(rows, groups, group_size)
    out = scales_f * q_g + biases_f
    return out.reshape(rows, in_features)


# ---------------------------------------------------------------------------
# Hand-rolled RMSNorm.
#
# The formula (per https://arxiv.org/abs/1910.07467):
#
#     y = x / sqrt(mean(x^2) + eps) * weight
#
# MLX accumulates the mean in float32 -- see nn.RMSNorm docstring. We match
# that by casting to float32 before the reduction. The weight is broadcast as
# a per-feature multiplier on the way out.
# ---------------------------------------------------------------------------


def rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Bit-for-bit match to ``mx.fast.rms_norm`` for bf16 inputs.

    Formula:   y = x / sqrt(mean(x^2) + eps) * weight

    Numerical order (matters for bf16!):
      1. cast x to float32 for the mean-of-squares reduction
      2. compute inverse sqrt in f32
      3. multiply the normalized activation in f32
      4. **downcast the activation to the weight dtype BEFORE** the weight
         multiply
      5. final result inherits the weight dtype

    The cast in step 4 is the non-obvious part. An "all-f32" variant matches
    RMSNorm's paper definition more literally but diverges from MLX's fused
    kernel at the one-bit-of-bf16 level (e.g. 1232 vs 1240 at magnitudes on
    the order of 1e3). Since our whole Phase B strategy is using MLX as a
    bit-exact oracle, we match the kernel, not the paper.
    """
    weight_dtype = weight.dtype
    x32 = x.astype(mx.float32)
    mean_sq = (x32 * x32).mean(axis=-1, keepdims=True)
    inv = 1.0 / mx.sqrt(mean_sq + eps)
    y32 = x32 * inv
    # step 4: drop to the weight dtype, *then* multiply.
    y = y32.astype(weight_dtype) * weight
    return y


# ---------------------------------------------------------------------------
# Checks.
# ---------------------------------------------------------------------------


def max_abs_diff(a: mx.array, b: mx.array) -> float:
    """Return the max absolute difference across matching shapes, as a
    Python float. Both inputs are promoted to float32 first so bf16 doesn't
    silently saturate the result."""
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    a32 = a.astype(mx.float32)
    b32 = b.astype(mx.float32)
    return float(mx.max(mx.abs(a32 - b32)))


def mean_abs(a: mx.array) -> float:
    return float(mx.mean(mx.abs(a.astype(mx.float32))))


def check_embedding(
    model_path: Path, verbose: bool
) -> tuple[bool, float, mx.array, mx.array, mx.array, mx.array]:
    """Compare our hand-rolled dequantized embedding against mlx_lm's.

    Returns (ok, max_diff, ids, ref_row, ours_row, scaled_row_ref).
    """
    cfg = json.loads((model_path / "config.json").read_text())
    hidden_size = cfg["text_config"]["hidden_size"]
    embed_scale = math.sqrt(hidden_size)

    # Load the embedding weights we need.
    tensors = load_safetensors_for(
        [
            "language_model.model.embed_tokens.weight",
            "language_model.model.embed_tokens.scales",
            "language_model.model.embed_tokens.biases",
        ],
        model_path,
    )
    w = tensors["language_model.model.embed_tokens.weight"]
    s = tensors["language_model.model.embed_tokens.scales"]
    b = tensors["language_model.model.embed_tokens.biases"]
    # Confirm the shapes we expect: (vocab, in_features / pack), (vocab, groups).
    # 4-bit with group_size=64 -> pack=8, groups = 2816 / 64 = 44.
    # But the embed weight prints as (262144, 352) = (vocab, 2816/8).
    group_size = 64
    bits = 4

    # Pick a handful of token IDs to check.
    ids = mx.array([0, 1, 2, 3, 10, 98, 100, 105, 106, 50000, 200000, 262143])
    ids_list = ids.tolist()

    # --- Ours ------------------------------------------------------
    rows_ours = mx.stack([
        dequantize_row(w[i], s[i], b[i], bits=bits, group_size=group_size)
        for i in ids_list
    ], axis=0)
    # Cast to bf16 to match mlx_lm's output dtype.
    rows_ours_bf16 = rows_ours.astype(mx.bfloat16)

    # --- Reference -------------------------------------------------
    model, _ = load(str(model_path))
    ref_embed = model.language_model.model.embed_tokens  # QuantizedEmbedding
    rows_ref = ref_embed(ids)   # shape (N, hidden), bfloat16

    max_diff = max_abs_diff(rows_ours_bf16, rows_ref)
    ok = max_diff == 0.0

    if verbose:
        print("  ids:", ids_list)
        print("  ref row[0][:6]:", rows_ref[0, :6].tolist())
        print("  ours row[0][:6]:", rows_ours_bf16[0, :6].tolist())

    # Also run the embed_scale multiplication (this is what the model does
    # before the first layer). Report max diff of the scaled version too.
    scaled_ref = rows_ref * embed_scale
    scaled_ours = rows_ours_bf16 * embed_scale
    scaled_diff = max_abs_diff(scaled_ours, scaled_ref)
    if verbose:
        print(f"  post-scale max |diff|: {scaled_diff:.6g}")

    return ok, max_diff, ids, rows_ref, rows_ours_bf16, scaled_ref


def check_final_norm(
    model_path: Path, scaled_ref: mx.array, verbose: bool
) -> tuple[bool, float]:
    """Compare hand-rolled RMSNorm against model.language_model.model.norm."""
    cfg = json.loads((model_path / "config.json").read_text())
    tc = cfg["text_config"]
    eps = tc["rms_norm_eps"]

    tensors = load_safetensors_for(
        ["language_model.model.norm.weight"], model_path
    )
    w = tensors["language_model.model.norm.weight"]

    # Ours.
    ours = rms_norm(scaled_ref, w, eps)

    # Reference.
    model, _ = load(str(model_path))
    ref = model.language_model.model.norm(scaled_ref)

    max_diff = max_abs_diff(ours, ref)
    ok = max_diff == 0.0

    if verbose:
        print(f"  norm weight[:6]: {w[:6].tolist()}")
        print(f"  ref[0, :6]:      {ref[0, :6].tolist()}")
        print(f"  ours[0, :6]:     {ours[0, :6].tolist()}")
        print(f"  max |diff|:      {max_diff:.6g}")

    return ok, max_diff


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_path = require_model_dir()

    print("=" * 72)
    print(" Hand-rolled Q4 embedding dequant vs mlx_lm QuantizedEmbedding")
    print("=" * 72)
    ok_embed, max_diff_embed, ids, rows_ref, rows_ours, scaled_ref = check_embedding(
        model_path, args.verbose
    )
    print(f"  checked {len(ids.tolist())} token IDs across the vocab")
    print(f"  max |diff|: {max_diff_embed:.6g}  "
          f"({'OK' if ok_embed else 'FAIL'}, threshold = exact bf16)")
    print()

    print("=" * 72)
    print(" Hand-rolled RMSNorm vs mlx.nn.RMSNorm (final-norm layer)")
    print("=" * 72)
    ok_norm, max_diff_norm = check_final_norm(model_path, scaled_ref, args.verbose)
    print(f"  max |diff|: {max_diff_norm:.6g}  "
          f"({'OK' if ok_norm else 'FAIL'}, threshold = exact bf16)")
    print()

    if ok_embed and ok_norm:
        print("All A3 checks passed.")
        return 0
    print("A3 checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
