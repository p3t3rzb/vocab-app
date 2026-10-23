"""Filename helpers for the database-selection screen."""
import os

import pytest

from src.gui import db_select_util
from src.gui.db_select_util import last_trained_label, last_trained_mtime, safe_db_basename, slugify


@pytest.mark.parametrize(
    "value, expected",
    [
        ("French", "french"),
        ("  Brazilian Portuguese ", "brazilian_portuguese"),
        ("Serbo-Croatian!!", "serbo_croatian"),
        ("日本語", ""),
    ],
)
def test_slugify(value, expected):
    assert slugify(value) == expected


def test_safe_db_basename():
    assert safe_db_basename("French", "Polish") == "french_polish"


@pytest.mark.parametrize("src, tgt", [("", "Polish"), ("French", "///"), ("..", "x")])
def test_safe_db_basename_rejects_empty_sides(src, tgt):
    assert safe_db_basename(src, tgt) == ""


def test_safe_db_basename_cannot_escape_directory():
    name = safe_db_basename("../../etc", "passwd")
    assert "/" not in name and ".." not in name


def test_last_trained_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(db_select_util.Paths, "model_path", staticmethod(lambda s, t: tmp_path / "none.pt"))
    assert last_trained_mtime("French", "Polish") == 0.0
    assert last_trained_label("French", "Polish") == "—"


def test_last_trained_with_checkpoint(tmp_path, monkeypatch):
    ckpt = tmp_path / "french_polish.pt"
    ckpt.write_bytes(b"")
    os.utime(ckpt, (1_700_000_000, 1_700_000_000))
    monkeypatch.setattr(db_select_util.Paths, "model_path", staticmethod(lambda s, t: ckpt))
    assert last_trained_mtime("French", "Polish") == 1_700_000_000
    assert last_trained_label("French", "Polish") != "—"
