from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import customtkinter as ctk

if TYPE_CHECKING:
    from .db_select import DatabaseSelectScreen
    from .word_list import WordListScreen
    from .word_detail import WordDetailScreen

STORAGE_DIR = Path(__file__).parent.parent.parent / "storage"


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Vocab Repetition")
        self.geometry("900x600")
        self.minsize(700, 450)

        from src.settings import load_settings
        ctk.set_appearance_mode(load_settings().appearance_mode)

        self._current_frame: ctk.CTkFrame | None = None
        self._db_path: Path | None = None
        self._src_lang: str = ""
        self._tgt_lang: str = ""

        self.show_db_select()

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _swap(self, frame: ctk.CTkFrame) -> None:
        if self._current_frame is not None:
            self._current_frame.destroy()
        self._current_frame = frame
        frame.pack(fill="both", expand=True)

    def show_db_select(self) -> None:
        from .db_select import DatabaseSelectScreen
        self._swap(DatabaseSelectScreen(self))

    def show_word_list(self, db_path: Path, src_lang: str, tgt_lang: str) -> None:
        self._db_path = db_path
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        from .word_list import WordListScreen
        self._swap(WordListScreen(self, db_path, src_lang, tgt_lang))

    def show_word_detail(self, word_id: int) -> None:
        assert self._db_path is not None
        from .word_detail import WordDetailScreen
        self._swap(WordDetailScreen(self, word_id, self._src_lang, self._tgt_lang))

    def back_to_word_list(self) -> None:
        assert self._db_path is not None
        self.show_word_list(self._db_path, self._src_lang, self._tgt_lang)

    def show_train_screen(self, db_path: Path, src_lang: str, tgt_lang: str) -> None:
        self._db_path = db_path
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        from .train_screen import TrainScreen
        self._swap(TrainScreen(self, db_path, src_lang, tgt_lang))

    def show_practice_screen(self, db_path: Path, src_lang: str, tgt_lang: str) -> None:
        self._db_path = db_path
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        from .practice_screen import PracticeScreen
        self._swap(PracticeScreen(self, db_path, src_lang, tgt_lang))

    def show_settings(self) -> None:
        from .settings_screen import SettingsScreen
        self._swap(SettingsScreen(self))
