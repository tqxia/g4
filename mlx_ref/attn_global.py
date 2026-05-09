"""Phase A4 — hand-rolled global attention (layer 5) vs the MLX reference.

We replicate one global-attention layer (the first one, layer 5) using only
raw safetensors weights, hand-rolled Q4 dequant, hand-rolled RMSNorm, a
hand-rolled p-RoPE, and stock matmul + ``mx.fast.scaled_dot_product_attention``.

The result is compared element-wise to ``model.layers[5].self_attn(x, ...)``.

Why global, not sliding? Global is the harder case:
  * head_dim = 512 (twice the sliding head_dim of 256)
  * use_k_eq_v=True (V projection absent; values come from the K row pre-RoPE)
  * partial-rotary p-RoPE: only the first 25% of each head dim is rotated;
    the remaining 75% is identity
  * uses k_norm/q_norm with the bigger 512-dim weights
  * num_global_kv_heads=2 so SDPA broadcasts 16/2=8x

If we get global right, sliding (a strict subset of these features) is easy.

This chapter introduces three new pieces:
  * ``proportional_rope`` -- p-RoPE for the global layer
  * ``per_head_rms_norm`` -- RMSNorm with the reduction over ``head_dim``,
    not the full hidden axis
  * ``quantized_linear`` -- dequant + standard linear on a Q4 weight, used
    for q/k/o_proj

The chapter pins the *output of one attention block* to MLX. Phase B will
have a CPU and Metal version of each piece; both must reproduce this.

Run:

    python -m mlx_ref.attn_global
    python -m mlx_ref.attn_global --layer 11    # any global layer (5,11,17,23,29)
    python -m mlx_ref.attn_global --verbose
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load

from .norm_check import (
    dequantize_matrix,
    load_safetensors_for,
    max_abs_diff,
    rms_norm,
)
from .paths import require_model_dir


# Global layers in this checkpoint; indices come straight from the
# layer_types list in config.json.
GLOBAL_LAYER_INDICES = [5, 11, 17, 23, 29]


# ---------------------------------------------------------------------------
# Quantized linear.
#
# A standard nn.Linear is y = x @ W^T (no bias). On disk we store W as
# (out_features, in_features) packed Q4 plus per-group scale + bias.
#
# Implementation note: we deliberately use ``mx.quantized_matmul`` rather
# than dequant-then-matmul. The fused op reproduces MLX's QuantizedLinear
# bit-for-bit (max |diff| = 0), whereas dequant-then-matmul rounds in a
# slightly different order and shows ~0.06 absolute drift on bf16 outputs
# of order 1.0. Phase B will need to match this same rounding behaviour;
# the chapter B notes capture how to reproduce it from a CPU/Metal kernel.
# ---------------------------------------------------------------------------


def quantized_linear(
    x: mx.array,
    packed_w: mx.array,    # (out_features, in_features // pack), uint32
    scales: mx.array,      # (out_features, in_features // group_size), bf16
    biases: mx.array,      # (out_features, in_features // group_size), bf16
    bits: int,
    group_size: int,
) -> mx.array:
    """y = x @ dequant(packed_w).T (output dtype matches input)."""
    return mx.quantized_matmul(
        x,
        packed_w,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )


# ---------------------------------------------------------------------------
# Hand-rolled p-RoPE (proportional / partial RoPE).
#
# Standard RoPE rotates pairs of dims (i, i+head_dim/2) by an angle
# theta(i, pos) = pos / base^(2i/dims). p-RoPE rotates only the first
# ``rotated_dims`` dims of each head; the remaining identity-passes through.
#
# For Gemma 4 global layers: dims=512, rotated_dims = int(512 * 0.25) = 128.
# So pairs are (0,64), (1,65), ..., (63,127). Dims 128..511 are untouched.
# Theta base for global is 1e6.
# ---------------------------------------------------------------------------


def proportional_rope(
    x: mx.array,                # (..., L, head_dim)  bf16
    rotated_dims: int,
    base: float,
    offset: int = 0,
) -> mx.array:
    """Reference Python implementation of ProportionalRoPE.

    Used for *teaching*, not for the oracle check. The pairing is the
    standard non-traditional RoPE: dim ``i`` pairs with dim
    ``i + head_dim/2``. Partial rotary just zeros out the angles for pairs
    whose first index is >= ``rotated_dims/2``.

    Concretely, with head_dim=512 and rotated_dims=128:
      * pairs (0, 256), (1, 257), ..., (63, 319) get the usual RoPE rotation
        with theta_i = base^(2i / head_dim)
      * pairs (64, 320), (65, 321), ..., (255, 511) are identity

    This matches mx.fast.rope's behaviour when called with a freqs vector
    of ``[base^expo, ..., inf, inf, ...]`` -- the inf entries drive cos=1,
    sin=0 (identity).

    Numerical caveat: this hand-rolled version disagrees with mx.fast.rope
    at the bf16 ulp level (~0.03 max abs diff on activations of order 1).
    Same root cause as the RMSNorm cast-order issue from chapter A3: the
    fused kernel does the rotation with a different intermediate-precision
    schedule. Phase B's CPU/Metal kernels will need to reproduce
    mx.fast.rope's schedule to stay on the oracle. For the A4 *oracle*
    comparison we call the fused op via :func:`fast_rope_with_partial`
    below; this function exists so the math is in plain Python for porting.
    """
    L, D = x.shape[-2], x.shape[-1]
    if rotated_dims == 0:
        return x
    if rotated_dims % 2 != 0:
        raise ValueError(f"rotated_dims must be even, got {rotated_dims}")
    if D % 2 != 0:
        raise ValueError(f"head_dim must be even, got {D}")

    half = D // 2                  # 256 for global head_dim=512
    rot_half = rotated_dims // 2   # 64 for partial=0.25

    exponents = mx.arange(0, rotated_dims, 2, dtype=mx.float32) / D  # (rot_half,)
    freqs = base ** exponents                                       # (rot_half,)
    positions = mx.arange(L, dtype=mx.float32) + float(offset)      # (L,)
    real_angles = positions[:, None] / freqs[None, :]               # (L, rot_half)
    zero_angles = mx.zeros((L, half - rot_half), dtype=mx.float32)
    angles = mx.concatenate([real_angles, zero_angles], axis=-1)    # (L, half)
    cos = mx.cos(angles).astype(x.dtype)
    sin = mx.sin(angles).astype(x.dtype)

    first  = x[..., :half]            # (..., L, half)
    second = x[..., half:]            # (..., L, half)
    rot_first  = first * cos - second * sin
    rot_second = first * sin + second * cos
    return mx.concatenate([rot_first, rot_second], axis=-1)


def fast_rope_with_partial(
    x: mx.array, rotated_dims: int, base: float, offset: int = 0
) -> mx.array:
    """Bit-exact RoPE that delegates to ``mx.fast.rope``.

    Builds the same freqs vector ``mlx_lm.models.rope_utils.ProportionalRoPE``
    builds: real frequencies for the rotated-pair half-positions, ``inf`` for
    the rest (identity). Used for the A4 oracle comparison.
    """
    D = x.shape[-1]
    half = D // 2
    rot_half = rotated_dims // 2
    exponents = mx.arange(0, rotated_dims, 2, dtype=mx.float32) / D
    freqs = mx.concatenate(
        [
            base ** exponents,
            mx.full((half - rot_half,), mx.inf),
        ]
    )
    return mx.fast.rope(
        x,
        D,
        traditional=False,
        base=None,
        scale=1.0,
        offset=offset,
        freqs=freqs,
    )


# ---------------------------------------------------------------------------
# Per-head RMSNorm.
#
# In Gemma 4 the q_norm/k_norm RMSNorm runs on each head's head_dim slice
# independently. The activation has shape (B, L, n_heads, head_dim); the
# norm reduces over the last axis with a per-head_dim weight vector that is
# *shared* across heads. v_norm uses the same shape but with weight=1
# (RMSNormNoScale).
# ---------------------------------------------------------------------------


def per_head_rms_norm(
    x: mx.array,            # (B, L, H, D)
    weight: mx.array | None,  # (D,) or None for "no scale" (treated as ones)
    eps: float,
) -> mx.array:
    if weight is None:
        weight = mx.ones((x.shape[-1],), dtype=x.dtype)
    return rms_norm(x, weight, eps)


# ---------------------------------------------------------------------------
# The check.
# ---------------------------------------------------------------------------


def proportional_rope_check(model_path: Path) -> tuple[bool, float]:
    """Companion check: hand-rolled `proportional_rope` vs `mx.fast.rope`.

    The hand-rolled version is for teaching / Phase B porting. We don't
    require bit-exact, only that the diff stays within bf16 ulp territory
    on activations of typical magnitude (~1.0) -- threshold 0.05.
    """
    cfg = json.loads((model_path / "config.json").read_text())
    tc = cfg["text_config"]
    hd = tc["global_head_dim"]
    rope_p = tc["rope_parameters"]["full_attention"]
    theta = rope_p["rope_theta"]
    rot = int(hd * rope_p["partial_rotary_factor"])

    x = mx.random.normal(shape=(1, 2, 16, hd), key=mx.random.key(7)).astype(mx.bfloat16)
    ours = proportional_rope(x, rotated_dims=rot, base=theta, offset=0)
    ref = fast_rope_with_partial(x, rotated_dims=rot, base=theta, offset=0)
    diff = max_abs_diff(ours, ref)
    ok = diff < 0.05
    return ok, diff


def attn_check(model_path: Path, layer_idx: int, verbose: bool) -> tuple[bool, float]:
    cfg = json.loads((model_path / "config.json").read_text())
    tc = cfg["text_config"]
    rms_eps = tc["rms_norm_eps"]
    n_heads = tc["num_attention_heads"]
    n_kv_global = tc["num_global_key_value_heads"]
    head_dim = tc["global_head_dim"]
    rope_params = tc["rope_parameters"]["full_attention"]
    rope_theta = rope_params["rope_theta"]
    rotated_dims = int(head_dim * rope_params["partial_rotary_factor"])
    bits = 4
    group_size = 64

    if layer_idx not in GLOBAL_LAYER_INDICES:
        raise SystemExit(
            f"layer {layer_idx} is not global; valid choices: {GLOBAL_LAYER_INDICES}"
        )

    pre = f"language_model.model.layers.{layer_idx}.self_attn."
    needed = [
        pre + "q_proj.weight", pre + "q_proj.scales", pre + "q_proj.biases",
        pre + "k_proj.weight", pre + "k_proj.scales", pre + "k_proj.biases",
        pre + "o_proj.weight", pre + "o_proj.scales", pre + "o_proj.biases",
        pre + "q_norm.weight",
        pre + "k_norm.weight",
        # No v_proj on global layers; v_norm is RMSNormNoScale (no learned weight).
    ]
    t = load_safetensors_for(needed, model_path)

    # Build a fixed input. Random with a seeded key so this is reproducible
    # across runs. Shape (1, 4, hidden) -- L=4 lets us exercise causality.
    H = tc["hidden_size"]
    B, L = 1, 4
    x = mx.random.normal(shape=(B, L, H), key=mx.random.key(42)).astype(mx.bfloat16)

    # Reference: load the model and call layer.self_attn(x, "causal", None, ...)
    model, _ = load(str(model_path))
    attn = model.language_model.model.layers[layer_idx].self_attn
    if not attn.use_k_eq_v:
        raise SystemExit(
            f"layer {layer_idx}: attn.use_k_eq_v is False; this script assumes K=V on global"
        )
    ref_out, ref_kvs, _ = attn(x, mask="causal", cache=None, shared_kv=None, offset=0)

    # ---- Hand-rolled path -------------------------------------------------
    # Q projection.
    q = quantized_linear(
        x,
        t[pre + "q_proj.weight"],
        t[pre + "q_proj.scales"],
        t[pre + "q_proj.biases"],
        bits, group_size,
    )                                                                # (B, L, n_heads * head_dim)
    q = q.reshape(B, L, n_heads, head_dim)
    q = per_head_rms_norm(q, t[pre + "q_norm.weight"], rms_eps)      # (B, L, n_heads, head_dim)

    # K projection.
    k = quantized_linear(
        x,
        t[pre + "k_proj.weight"],
        t[pre + "k_proj.scales"],
        t[pre + "k_proj.biases"],
        bits, group_size,
    )                                                                # (B, L, n_kv_global * head_dim)
    k = k.reshape(B, L, n_kv_global, head_dim)

    # K=V trick: V comes from k *before* k_norm and *before* RoPE (matches
    # ``values = keys`` at line 252, then values gets only v_norm + transpose).
    v = k

    # Now apply k_norm and RoPE to k; v_norm (no-scale) and transpose for v.
    k = per_head_rms_norm(k, t[pre + "k_norm.weight"], rms_eps)
    k = k.transpose(0, 2, 1, 3)                                      # (B, n_kv, L, head_dim)
    k = fast_rope_with_partial(k, rotated_dims=rotated_dims, base=rope_theta, offset=0)

    v = per_head_rms_norm(v, None, rms_eps)                          # RMSNormNoScale
    v = v.transpose(0, 2, 1, 3)                                      # (B, n_kv, L, head_dim)

    # Q transpose + p-RoPE.
    q = q.transpose(0, 2, 1, 3)                                      # (B, n_heads, L, head_dim)
    q = fast_rope_with_partial(q, rotated_dims=rotated_dims, base=rope_theta, offset=0)

    # SDPA. scale=1.0 (Gemma 4 quirk: they don't divide by sqrt(head_dim)).
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0, mask="causal"
    )                                                                # (B, n_heads, L, head_dim)
    out = out.transpose(0, 2, 1, 3).reshape(B, L, n_heads * head_dim)

    out = quantized_linear(
        out,
        t[pre + "o_proj.weight"],
        t[pre + "o_proj.scales"],
        t[pre + "o_proj.biases"],
        bits, group_size,
    )                                                                # (B, L, hidden)

    # ---- Compare ----------------------------------------------------------
    diff = max_abs_diff(out, ref_out)
    ok = diff == 0.0

    if verbose:
        print(f"  layer {layer_idx} ({attn.layer_type}, head_dim={head_dim})")
        print(f"  q/k/v/o shapes (logical):")
        print(f"    q: {q.shape}  k: {k.shape}  v: {v.shape}  out: {out.shape}")
        print(f"  ref output [0,0,:6]: {ref_out[0, 0, :6].tolist()}")
        print(f"  ours       [0,0,:6]: {out[0, 0, :6].tolist()}")
        # Also report the raw KV state diff
        kv_diff_k = max_abs_diff(k, ref_kvs[0])
        kv_diff_v = max_abs_diff(v, ref_kvs[1])
        print(f"  pre-SDPA k vs ref: max |diff| = {kv_diff_k}")
        print(f"  pre-SDPA v vs ref: max |diff| = {kv_diff_v}")

    return ok, diff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=5,
                        help=f"global layer to check, one of {GLOBAL_LAYER_INDICES}")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_path = require_model_dir()
    print("=" * 72)
    print(f" Hand-rolled global attention vs MLX reference (layer {args.layer})")
    print("=" * 72)
    ok, max_diff = attn_check(model_path, args.layer, args.verbose)
    print(f"  max |diff|: {max_diff:.6g}  "
          f"({'OK' if ok else 'FAIL'}, threshold = exact bf16)")
    print()

    print("=" * 72)
    print(" Companion: hand-rolled proportional_rope vs mx.fast.rope")
    print("=" * 72)
    ok_rope, rope_diff = proportional_rope_check(model_path)
    print(f"  max |diff|: {rope_diff:.6g}  "
          f"({'OK' if ok_rope else 'FAIL'}, threshold 0.05 -- bf16 ulp)")
    return 0 if (ok and ok_rope) else 1


if __name__ == "__main__":
    raise SystemExit(main())
