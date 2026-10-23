"""Shared fixtures for the whole suite.

Layout:

* ``unit/`` — pure functions and small classes, no database. Fast.
* ``integration/`` — code that goes through :func:`src.database.get_session`,
  each test against its own throwaway SQLite file (the ``db`` fixture).

Nothing here ever touches ``storage/``: databases, checkpoints and the settings
file are all redirected into pytest's ``tmp_path``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.database import init_db
from src.model.lstm import RecallLSTM


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Load checkpoints on CPU so results don't depend on MPS/CUDA availability."""
    monkeypatch.setattr("src.model.checkpoint.get_device", lambda: torch.device("cpu"))


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Point the settings file at a temp path so no test reads or writes the real one."""
    monkeypatch.setattr("src.settings.SETTINGS_FILE", tmp_path / "settings.json")


@pytest.fixture
def db(tmp_path) -> Path:
    """A fresh, initialised French→Polish database; returns its file path."""
    path = tmp_path / "test.db"
    init_db(f"sqlite:///{path}", "French", "Polish")
    return path


@pytest.fixture
def tiny_model() -> RecallLSTM:
    """A small, deterministic, untrained network in eval mode."""
    torch.manual_seed(0)
    model = RecallLSTM(hidden_size=8, num_layers=1, dropout=0.0)
    model.eval()
    return model
