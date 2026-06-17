"""Global, per-installation user settings.

Settings are stored as a single JSON file at ``storage/settings.json`` and are
shared across every language-pair database. Use :func:`load_settings` to read
the current values (returning defaults if the file is missing or invalid) and
:func:`save_settings` to persist changes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

from src.model.config import PredictConfig
from src.paths import SETTINGS_FILE


@dataclass
class AppSettings:
    """User-configurable application settings.

    Attributes:
        recall_threshold: P(recall) level below which a word is considered due.
            Lower values mean longer intervals between repetitions.
        max_delta_seconds: Hard cap on the predicted interval to the next
            repetition. Defaults to one year.
        appearance_mode: ``"Light"``, ``"Dark"``, or ``"System"`` — applied via
            ``ctk.set_appearance_mode`` at startup and when the user saves.
    """

    recall_threshold: float = 0.80
    max_delta_seconds: float = 365 * 86400.0
    appearance_mode: str = "System"  # "Light" | "Dark" | "System"

    def to_predict_config(self) -> PredictConfig:
        """Build a :class:`PredictConfig` with this settings' threshold/cap.

        Used wherever recall and due times are derived *live* from the stored
        curve params (the practice queue, the word list's due-time cache, and the
        :mod:`src.predict` CLI) so the user's threshold/cap are honoured.
        """
        return PredictConfig(
            recall_threshold=self.recall_threshold,
            max_delta_seconds=self.max_delta_seconds,
        )


def load_settings() -> AppSettings:
    """Load settings from disk.

    Returns the defaults if the settings file does not exist, is unreadable,
    or contains invalid JSON. Unknown keys in the file are ignored so that
    older installations remain forward-compatible.
    """
    if not SETTINGS_FILE.exists():
        return AppSettings()
    try:
        data = json.loads(SETTINGS_FILE.read_text())
    except Exception:
        return AppSettings()
    allowed = {f.name for f in fields(AppSettings)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    try:
        return AppSettings(**filtered)
    except Exception:
        return AppSettings()


def save_settings(s: AppSettings) -> None:
    """Write ``s`` to ``storage/settings.json``, creating parent dirs as needed."""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(asdict(s), indent=2))
