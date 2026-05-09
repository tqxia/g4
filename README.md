# g4 — Gemma 4 inference engine for Mac

`g4` is a small, Mac-native inference engine for
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it).
It is intentionally narrow: not a generic GGUF runner, not a wrapper around
another runtime, and not a framework. The main path will be a Gemma-4-specific
Metal graph executor with Gemma-4-specific loading, prompt rendering, KV state,
and server API glue.

The project is modelled after
[`antirez/ds4`](https://github.com/antirez/ds4) — a DeepSeek V4 Flash engine —
and follows the same philosophy: one model at a time, correctness gated by
official-reference logits, Metal-only fast path, and enough agent integration
to know whether it really works.

This project would not exist without **llama.cpp / GGML**, **Apple MLX / mlx-lm**,
and **ds4**. See the acknowledgements at the bottom.

## Status

**Alpha. Under construction.** We are building the engine chapter-by-chapter.
Each chapter is a git tag (`chapter-00-setup`, `chapter-01-weights-tour`, …)
snapshotting the minimal working state at that milestone. You can check out
any chapter tag and run it.

See [`AGENT.md`](./AGENT.md) for coding rules and
[`plan.md`](./plan.md) (once written) for the chapter roadmap.

## Why Gemma 4 26B A4B

- Mixture-of-Experts: **3.8 B active parameters out of 25.2 B total** — fast
  on an M-series Mac with more world knowledge than a ~4B dense model.
- **256K context window** with **p-RoPE** on global layers and interleaved
  **sliding-window attention** (window = 1024) on local layers. Long-context
  is a first-class design concern.
- **Native thinking mode** via `<|think|>`: reasoning is emitted on a
  separate channel and can be streamed independently, matching OpenAI-style
  `reasoning_content` and Anthropic-style `thinking` blocks.
- **Apache 2.0**, open weights, actively maintained by Google DeepMind.

## Target hardware

v1 targets a **64 GB Apple Silicon Mac** with Q4 weights (~15 GB on disk, fits
comfortably with room for a 32–64k-token KV cache). BF16 and asymmetric
MoE-only Q2 come later.

The CPU path is correctness-only. The release path is Metal.

## Download the model

v1 uses MLX-format 4-bit weights from the `mlx-community` mirror:

```sh
./scripts/download_model.sh
```

This runs `hf` (from the `huggingface_hub` Python package) to pull
[`mlx-community/gemma-4-26b-a4b-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-4bit)
into `./models/gemma-4-26b-a4b-4bit/`.

## Phase A: MLX reference

Before building the native C engine we implement Gemma 4 end-to-end in MLX.
This reference stays alive as the correctness oracle for everything that
follows.

```sh
pip install -r requirements.txt
./scripts/download_model.sh
python -m mlx_ref.run -p "Hello."
```

See the `mlx_ref/` directory.

## Phase B: C99 + Metal

Still to come. It will look a lot like `ds4/` in layout: `g4.c`, `g4_metal.m`,
`metal/*.metal`, `g4_cli.c`, `g4_server.c`.

## Acknowledgements

- **llama.cpp / GGML** — the GGUF format, the tensor layouts, the quant
  kernels, years of Metal engineering. This project will carry MIT-licensed
  material directly from there.
- **Apple MLX / `mlx-lm`** — the Phase A reference forward pass is derived
  from `mlx_lm/models/gemma4.py`.
- **ds4 (`antirez/ds4`)** — the model for how a small, narrow, one-model
  inference engine should be structured. Many patterns (GGUF cursor loader,
  mmap'd weights, fused Metal graph, disk KV cache keyed by token-ID SHA1,
  OpenAI + Anthropic server surface) are ported directly.
