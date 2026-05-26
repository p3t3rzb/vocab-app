"""Project-wide filesystem layout.

Centralises the locations of ``storage/``, the model checkpoint directory,
and the settings file so that GUI code, settings, and the model package
all agree on where things live.
"""
from __future__ import annotations

from pathlib import Path

STORAGE_DIR: Path = Path(__file__).parent.parent / "storage"
MODELS_DIR: Path = STORAGE_DIR / "models"
SETTINGS_FILE: Path = STORAGE_DIR / "settings.json"


def model_path(src_lang: str, tgt_lang: str) -> Path:
    """Return the checkpoint path for a language pair (case-insensitive)."""
    return MODELS_DIR / f"{src_lang.lower()}_{tgt_lang.lower()}.pt"
