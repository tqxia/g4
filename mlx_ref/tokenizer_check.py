"""Phase A2 — tokenizer wrap + round-trip.

Wraps the HuggingFace fast tokenizer that ships with
``mlx-community/gemma-4-26b-a4b-4bit`` and prints:

  * vocab size, model type, pre-tokenizer, normalizer, decoder
  * the 24 added special tokens (with IDs and key categories: reasoning,
    chat structure, tool-calling, image/audio/video)
  * round-trip encode-then-decode for a canonical prompt corpus
  * expected-id spot-checks for ``<bos>``, ``<eos>``, ``<|think|>``,
    ``<|turn>``, ``<turn|>`` to make sure they match the IDs we will rely
    on in chapter A9 when rendering the chat template

Run:

    python -m mlx_ref.tokenizer_check
    python -m mlx_ref.tokenizer_check --file tests/prompts/short.txt
    python -m mlx_ref.tokenizer_check -p "Your custom text here."

Round-trip policy: we consider a prompt *lossless* if
``decode(encode(text)) == text`` byte-for-byte. Gemma's normalizer replaces
``' '`` with ``▁`` and its decoder reverses that, so this should hold for
any plain UTF-8 input. Leading-space handling and unicode edge cases are
the things this chapter exists to confirm.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer

from .paths import require_model_dir


# ---------------------------------------------------------------------------
# Token category heuristics.
#
# The tokenizer has 24 special tokens. Sorting them by purpose turns an
# opaque integer dump into a legend we can refer back to when we implement
# the chat template in chapter A9.
# ---------------------------------------------------------------------------


@dataclass
class SpecialToken:
    id: int
    content: str
    category: str


def categorize_special(content: str) -> str:
    c = content
    if c in ("<pad>", "<eos>", "<bos>", "<unk>", "<mask>"):
        return "core"
    if "think" in c:
        return "reasoning"
    if "channel" in c:
        return "reasoning"
    if "turn" in c:
        return "chat"
    if "tool_call" in c:
        return "tool_call"
    if "tool_response" in c:
        return "tool_response"
    if "tool" in c:
        return "tool"
    if "image" in c:
        return "image"
    if "audio" in c:
        return "audio"
    if "video" in c:
        return "video"
    return "other"


# Token IDs we will depend on in chapter A9 (chat template). If any of these
# drift the template will produce wrong bytes and every downstream test will
# fail, so catch it here.
EXPECTED_IDS: dict[str, int] = {
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    "<unk>": 3,
    "<|think|>": 98,
    "<|channel>": 100,
    "<channel|>": 101,
    "<|turn>": 105,
    "<turn|>": 106,
}


# Canonical round-trip corpus. Keep these committed and tiny so tests are
# reproducible without external data. Each entry pokes at something specific:
ROUND_TRIP_CASES: list[tuple[str, str]] = [
    ("ascii",          "The quick brown fox jumps over the lazy dog."),
    ("leading_space",  " leading space"),
    ("trailing_space", "trailing space "),
    ("double_space",   "two  spaces between"),
    ("newlines",       "first line\nsecond line\n"),
    ("tab",            "col1\tcol2\tcol3"),
    ("unicode_mixed",  "café résumé — naïve"),
    ("emoji",          "hello 👋 world 🌍"),
    ("cjk",            "你好,世界。こんにちは。안녕하세요."),
    ("code",           "def f(x: int) -> int:\n    return x + 1\n"),
    ("markdown",       "# Title\n\n- item 1\n- item 2\n\n```py\nprint('hi')\n```"),
    ("think_marker",   "Answer: <|think|>because ...<|channel>final<channel|>ok"),
    ("bos_eos",        "<bos>hello<eos>"),
]


# ---------------------------------------------------------------------------
# Loader.
# ---------------------------------------------------------------------------


def load_tokenizer(model_path: Path) -> Tokenizer:
    tok_path = model_path / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"no tokenizer.json at {tok_path}")
    return Tokenizer.from_file(str(tok_path))


def added_tokens(tokenizer: Tokenizer) -> list[SpecialToken]:
    # ``get_added_tokens_decoder`` returns {id: AddedToken}. AddedToken has
    # .content attribute. This is stable across tokenizers>=0.14.
    out: list[SpecialToken] = []
    for tid, atoken in sorted(tokenizer.get_added_tokens_decoder().items()):
        out.append(SpecialToken(id=tid, content=atoken.content, category=categorize_special(atoken.content)))
    return out


# ---------------------------------------------------------------------------
# Output sections.
# ---------------------------------------------------------------------------


def print_header(tokenizer: Tokenizer, model_path: Path) -> None:
    print("=" * 72)
    print(" Tokenizer summary")
    print("=" * 72)
    print(f"  model path           {model_path}")
    print(f"  file                 tokenizer.json")
    print(f"  vocab size           {tokenizer.get_vocab_size(with_added_tokens=True)}")
    print(f"  base vocab size      {tokenizer.get_vocab_size(with_added_tokens=False)}")

    # tokenizers does not expose model/pre_tokenizer/decoder types as plain
    # Python objects with stable attributes, so we just read the JSON config
    # strings via to_str() and pull out the interesting bits by substring.
    import json as _json
    config = _json.loads(tokenizer.to_str())
    model = config.get("model", {}) or {}
    pre = config.get("pre_tokenizer", {}) or {}
    norm = config.get("normalizer", {}) or {}
    dec = config.get("decoder", {}) or {}
    print(f"  model type           {model.get('type')}  (merges: {len(model.get('merges') or [])})")
    print(f"  pre_tokenizer        {pre.get('type')}")
    print(f"  normalizer           {norm.get('type')}  (replace space with '▁')")
    print(f"  decoder              {dec.get('type')}")
    print()


def print_special_tokens(specials: list[SpecialToken]) -> None:
    print("=" * 72)
    print(f" Special tokens ({len(specials)})")
    print("=" * 72)
    print(f"  {'id':>8s}  {'content':20s}  category")
    print(f"  {'-' * 8}  {'-' * 20}  --------")
    for st in specials:
        print(f"  {st.id:8d}  {st.content:20s}  {st.category}")
    print()

    # ID sanity-check.
    print("=" * 72)
    print(" ID spot-check")
    print("=" * 72)
    by_content = {st.content: st.id for st in specials}
    ok = True
    for content, expected in EXPECTED_IDS.items():
        got = by_content.get(content)
        status = "OK" if got == expected else "MISMATCH"
        if got != expected:
            ok = False
        print(f"  {status:9s} {content:16s}  expected={expected:>8d}  got={got!s}")
    print()
    if not ok:
        print("! One or more special-token IDs differ from the chapter A9 plan.")
        print("! The chat template will need to be regenerated; flag this run.")
        print()


def print_round_trip(tokenizer: Tokenizer, cases: Iterable[tuple[str, str]]) -> int:
    print("=" * 72)
    print(" Round-trip tests  (decode(encode(text)) == text)")
    print("=" * 72)
    failures = 0
    print(f"  {'name':16s} {'result':8s} {'n_tok':>6s}  preview")
    print(f"  {'-' * 16} {'-' * 8} {'-' * 6}  -------")
    for name, text in cases:
        # add_special_tokens=False: we want to see exactly what the model
        # gets, without automatic <bos>. That is the text we'll compare.
        enc = tokenizer.encode(text, add_special_tokens=False)
        ids = enc.ids
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        status = "OK" if decoded == text else "FAIL"
        if decoded != text:
            failures += 1
        preview = text.replace("\n", "\\n").replace("\t", "\\t")
        if len(preview) > 34:
            preview = preview[:31] + "..."
        print(f"  {name:16s} {status:8s} {len(ids):>6d}  {preview}")
        if decoded != text:
            print(f"    ! decode mismatch:")
            print(f"      in : {text!r}")
            print(f"      out: {decoded!r}")
    print()
    if failures:
        print(f"! {failures} round-trip failure(s)")
        print()
    return failures


def print_example_encoding(tokenizer: Tokenizer, text: str) -> None:
    print("=" * 72)
    print(" Example encoding")
    print("=" * 72)
    print(f"  input: {text!r}")
    enc = tokenizer.encode(text, add_special_tokens=False)
    print(f"  n_tokens: {len(enc.ids)}")
    print(f"  {'idx':>4s}  {'id':>8s}  piece")
    print(f"  {'-' * 4}  {'-' * 8}  -----")
    for i, (tid, piece) in enumerate(zip(enc.ids, enc.tokens)):
        # Escape the SentencePiece '▁' to make it visually obvious.
        show = piece.replace("▁", "_")
        print(f"  {i:>4d}  {tid:>8d}  {show!r}")
    print()


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p", "--prompt",
        help="tokenize this string at the end, in addition to the round-trip set",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="tokenize the contents of this file at the end",
    )
    args = parser.parse_args()

    model_path = require_model_dir()
    tokenizer = load_tokenizer(model_path)

    print_header(tokenizer, model_path)
    print_special_tokens(added_tokens(tokenizer))
    failures = print_round_trip(tokenizer, ROUND_TRIP_CASES)

    if args.prompt:
        print_example_encoding(tokenizer, args.prompt)
    if args.file:
        print_example_encoding(tokenizer, args.file.read_text())

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
