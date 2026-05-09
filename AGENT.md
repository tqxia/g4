# Agent Notes

`g4` is a Gemma 4 26B A4B specific inference engine. It is not a generic GGUF
or safetensors runner. The goal is a small, readable, high-performance
codebase — MLX (Python) in Phase A, C99 with Objective-C-only-where-Metal-
requires-it and Metal kernels under `metal/` in Phase B onward.

The project is modelled after `ds4` (DeepSeek V4 Flash engine); many of the
rules below come straight from its `AGENT.md`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF / safetensors.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with
  unexplained attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Project structure

- **Hybrid chapters.** `main` grows a ds4-style monolithic codebase. Each
  chapter ends in a git tag (`chapter-00-setup`, `chapter-01-weights-tour`,
  …) snapshotting the minimal working state at that milestone. Any chapter
  tag must be checkout-and-runnable.
- **Phase A (`mlx_ref/`)** is the MLX Python reference. It stays alive
  forever as the correctness oracle.
- **Phase B (`g4.c`, `g4_metal.m`, `metal/`)** is the production native
  engine.
- **Phase C** is performance + server + disk KV cache on top of Phase B.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI / server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches
  are fine when they validate the one release path.
- Do not introduce C++ in Phase B / C.
- New chapters must ship a regression test (`g4_test --chapter N` in Phase B,
  a pytest or comparison script in Phase A) before they get tagged.

## Correctness

- The MLX Phase A reference is the **official oracle**. After chapter A9 we
  freeze a set of (prompt, token IDs, logits checksum) test vectors. Every
  Phase B / C change must reproduce them.
- Two-level diff harness (mirrors ds4): `./g4 --cpu` runs the C CPU
  reference path; `./g4 --metal` runs the release Metal path;
  `./g4 --dump-logprobs` captures logit distributions for golden-vector
  comparison.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has historically
  exposed kernel VM failures with very large mappings (noted in ds4).
- Do not run multiple huge model processes concurrently once the instance
  lock is in place.
- Prefer short Metal smoke tests for build verification.
