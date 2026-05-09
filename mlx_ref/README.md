# Phase A — MLX reference for Gemma 4 26B A4B

This directory holds the Python / MLX reference implementation. It is both an
educational artifact (we implement each piece of Gemma 4 by hand and compare
against `mlx_lm.models.gemma4_text`) and a correctness oracle (every Phase B
kernel is checked against output captured from here).

Chapters:

- A0 setup (this file).
- A1 weights tour — `tour_weights.py`.
- A2 tokenizer — `tokenizer_check.py`.
- A3 RMSNorm + embed — `norm_check.py`.
- A4 global attention — `attn_global.py`.
- A5 sliding-window attention — `attn_swa.py`.
- A6 MoE FFN — `moe.py`.
- A7 full forward — `forward.py`.
- A8 KV cache — `kv_cache.py`.
- A9 chat template + test-vector capture — `chat.py`, `capture.py`.
- A10 sampler + CLI — `run.py`.

Run:

```sh
pip install -r requirements.txt
./scripts/download_model.sh
python -m mlx_ref.tour_weights     # once A1 lands
```
