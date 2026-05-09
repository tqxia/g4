#!/bin/sh
set -e

# g4 model downloader (Phase A).
#
# Downloads the MLX-community 4-bit quantized Gemma 4 26B A4B weights into
# ./models/gemma-4-26b-a4b-4bit/. Uses the `hf` CLI that ships with the
# huggingface_hub Python package (installed via requirements.txt).
#
# Phase B will switch to a GGUF build converted once from these weights (or
# from the upstream BF16 weights), so we do not pull Google's gated repo here.

REPO="mlx-community/gemma-4-26b-a4b-4bit"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT_DIR="$ROOT/models/gemma-4-26b-a4b-4bit"
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<EOF
g4 Gemma 4 26B A4B MLX weights downloader

Usage:
  ./scripts/download_model.sh [--token TOKEN]

Pulls $REPO
into $OUT_DIR (about 15.6 GB).

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local
                 Hugging Face token cache (~/.cache/huggingface/token) is
                 used if present. Public mirror, token is optional.

Requirements:
  python3 -m pip install -r requirements.txt
  (provides the hf CLI)
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if ! command -v hf >/dev/null 2>&1; then
    echo "The 'hf' CLI is not on PATH." >&2
    echo "Install it with:  pip install -r requirements.txt" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "Downloading $REPO"
echo "into $OUT_DIR"

if [ -n "$TOKEN" ]; then
    hf download "$REPO" --local-dir "$OUT_DIR" --token "$TOKEN"
else
    hf download "$REPO" --local-dir "$OUT_DIR"
fi

cd "$ROOT"
ln -sfn "models/gemma-4-26b-a4b-4bit" g4-model
echo
echo "Linked ./g4-model -> models/gemma-4-26b-a4b-4bit"
echo "Done."
