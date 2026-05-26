"""Value object carrying the database identity through the screen tree.

A :class:`DbContext` bundles the database path with the source/target
language names. Screens take a single ``ctx`` argument instead of three
separate fields and obtain the derived ``db_url`` and ``model_path`` from
the context itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .theme import Paths


@dataclass(frozen=True, slots=True)
class DbContext:
    """Immutable identity of the currently active language-pair database."""

    db_path: Path
    src_lang: str
    tgt_lang: str

    @property
    def db_url(self) -> str:
        """SQLAlchemy URL for this database."""
        return f"sqlite:///{self.db_path}"

    @property
    def model_path(self) -> Path:
        """Filesystem path of the matching trained-model checkpoint."""
        return Paths.model_path(self.src_lang, self.tgt_lang)

    @property
    def title(self) -> str:
        """``"French ↔ Polish"`` style label used in screen headers."""
        return f"{self.src_lang}  ↔  {self.tgt_lang}"
