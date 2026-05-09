"""Shared helpers for Phase A scripts.

Keep this module tiny. Its only job is locating the downloaded model on disk
and providing a single place to adjust the path.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "gemma-4-26b-a4b-4bit"
SYMLINK_MODEL_DIR = REPO_ROOT / "g4-model"


def model_dir() -> Path:
    """Return the directory with the downloaded MLX weights + tokenizer.

    Resolution order:
      1. ``G4_MODEL_DIR`` environment variable, if set.
      2. ``./g4-model`` symlink (created by ``scripts/download_model.sh``).
      3. ``./models/gemma-4-26b-a4b-4bit`` fallback.
    """
    env = os.environ.get("G4_MODEL_DIR")
    if env:
        return Path(env).expanduser().resolve()
    if SYMLINK_MODEL_DIR.exists():
        return SYMLINK_MODEL_DIR.resolve()
    return DEFAULT_MODEL_DIR


def require_model_dir() -> Path:
    """Same as :func:`model_dir`, but raises a friendly error if missing."""
    path = model_dir()
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}.\n"
            "Run ./scripts/download_model.sh first, or set G4_MODEL_DIR."
        )
    return path
