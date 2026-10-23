"""Project-wide filesystem layout.

Centralises the locations of ``storage/``, the database and model checkpoint
directories, and the settings file so that GUI code, settings, and the model package
all agree on where things live.
"""

from __future__ import annotations

from pathlib import Path

STORAGE_DIR: Path = Path(__file__).parent.parent / "storage"
DATABASES_DIR: Path = STORAGE_DIR / "databases"
MODELS_DIR: Path = STORAGE_DIR / "models"
SETTINGS_FILE: Path = STORAGE_DIR / "settings.json"


def model_path(src_lang: str, tgt_lang: str) -> Path:
    """Return the checkpoint path for a language pair (case-insensitive)."""
    return MODELS_DIR / f"{src_lang.lower()}_{tgt_lang.lower()}.pt"


def ensure_storage_dirs() -> None:
    """Create the ``storage/`` folder structure if it does not exist yet."""
    DATABASES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
