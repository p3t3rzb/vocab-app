from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from src.model.config import PredictConfig

SETTINGS_PATH = Path(__file__).parent.parent / "storage" / "settings.json"


@dataclass
class AppSettings:
    recall_threshold: float = 0.80
    max_delta_seconds: float = 365 * 86400.0
    appearance_mode: str = "System"  # "Light" | "Dark" | "System"

    def to_predict_config(self) -> PredictConfig:
        return PredictConfig(
            recall_threshold=self.recall_threshold,
            max_delta_seconds=self.max_delta_seconds,
        )


def load_settings() -> AppSettings:
    if not SETTINGS_PATH.exists():
        return AppSettings()
    try:
        data = json.loads(SETTINGS_PATH.read_text())
    except Exception:
        return AppSettings()
    allowed = {f.name for f in fields(AppSettings)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    try:
        return AppSettings(**filtered)
    except Exception:
        return AppSettings()


def save_settings(s: AppSettings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(asdict(s), indent=2))
