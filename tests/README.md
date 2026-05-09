# g4 test harness.

Phase A (MLX) test scripts live in `../mlx_ref/`.
Phase B (C) test harness will be `g4_test.c` here.

## test-vectors/

Golden outputs captured from the MLX reference (Phase A chapter 9). Each
vector is a `(prompt, token_ids, logits_sha1)` bundle — generation must
reproduce exactly the same token IDs, and the final logits must hash to the
same value, for the engine to be considered correct.

File format TBD; will be defined in chapter A9.

## prompts/

Canonical prompts used throughout the chapters. Keep them small and
committed; they make tests reproducible without downloading corpora.
