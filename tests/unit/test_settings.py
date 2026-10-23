"""Settings persistence. ``SETTINGS_FILE`` is redirected to a temp path by conftest."""
import json
import math

from src import settings
from src.settings import AppSettings, load_settings, save_settings


def test_missing_file_gives_defaults():
    assert load_settings() == AppSettings()


def test_round_trip():
    s = AppSettings(recall_threshold=0.7, appearance_mode="Dark", gain_only_ordering=True, use_threshold=False)
    save_settings(s)
    assert load_settings() == s


def test_infinite_cap_round_trips():
    save_settings(AppSettings(max_delta_seconds=math.inf))
    assert load_settings().max_delta_seconds == math.inf


def test_invalid_json_gives_defaults():
    settings.SETTINGS_FILE.write_text("{not json")
    assert load_settings() == AppSettings()


def test_unknown_keys_are_ignored():
    settings.SETTINGS_FILE.write_text(json.dumps({"recall_threshold": 0.6, "from_the_future": 1}))
    assert load_settings().recall_threshold == 0.6


def test_to_predict_config():
    cfg = AppSettings(recall_threshold=0.65, max_delta_seconds=100.0).to_predict_config()
    assert (cfg.recall_threshold, cfg.max_delta_seconds) == (0.65, 100.0)
