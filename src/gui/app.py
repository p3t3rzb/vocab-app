"""Root window for the customtkinter desktop GUI.

The :class:`App` class is the top-level ``CTk`` window. It owns the
currently visible frame, the active :class:`DbContext`, and exposes one
``show_*`` method per screen — each one destroys the old frame and
instantiates the new one. Screen imports are deferred so the import graph
stays free of cycles and startup remains cheap.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import customtkinter as ctk

from src.settings import load_settings

from .db_context import DbContext
from .theme import WindowSizes

if TYPE_CHECKING:
    from .base_screen import BaseScreen


class App(ctk.CTk):
    """The main application window and navigation router."""

    def __init__(self) -> None:
        """Build the window and show the database-selection screen."""
        super().__init__()
        self.title("Vocab Repetition")
        self.geometry(WindowSizes.MAIN)
        self.minsize(*WindowSizes.MAIN_MIN)

        ctk.set_appearance_mode(load_settings().appearance_mode)

        self._current_frame: ctk.CTkFrame | None = None
        self._ctx: DbContext | None = None

        self.show_db_select()

    # ------------------------------------------------------------------
    # Active database context
    # ------------------------------------------------------------------

    @property
    def ctx(self) -> DbContext:
        """The currently active :class:`DbContext`.

        Asserts non-None so callers can rely on it without re-checking.
        """
        assert self._ctx is not None, "No database has been opened yet."
        return self._ctx

    def set_ctx(self, ctx: DbContext) -> None:
        """Remember ``ctx`` as the active database."""
        self._ctx = ctx

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _swap(self, frame: BaseScreen | ctk.CTkFrame) -> None:
        """Destroy the current frame (if any) and install ``frame`` in its place."""
        if self._current_frame is not None:
            self._current_frame.destroy()
        self._current_frame = frame
        frame.pack(fill="both", expand=True)
        on_show = getattr(frame, "on_show", None)
        if callable(on_show):
            on_show()

    def show_db_select(self) -> None:
        """Display the database-selection screen (the app's home view)."""
        from .db_select import DatabaseSelectScreen
        self._swap(DatabaseSelectScreen(self))

    def show_word_list(self, ctx: DbContext) -> None:
        """Open ``ctx`` and display its word list."""
        self.set_ctx(ctx)
        from .word_list import WordListScreen
        self._swap(WordListScreen(self))

    def show_word_detail(self, word_id: int) -> None:
        """Display the repetition history for a single word."""
        from .word_detail import WordDetailScreen
        self._swap(WordDetailScreen(self, word_id))

    def back_to_word_list(self) -> None:
        """Re-show the word list for the active database."""
        self.show_word_list(self.ctx)

    def show_train_screen(self, ctx: DbContext) -> None:
        """Display the model-training screen for ``ctx``."""
        self.set_ctx(ctx)
        from .train import TrainScreen
        self._swap(TrainScreen(self))

    def show_practice_screen(self, ctx: DbContext) -> None:
        """Enter the spaced-repetition practice session for ``ctx``."""
        self.set_ctx(ctx)
        from .practice import PracticeScreen
        self._swap(PracticeScreen(self))

    def show_settings(self) -> None:
        """Display the global settings screen."""
        from .settings import SettingsScreen
        self._swap(SettingsScreen(self))
