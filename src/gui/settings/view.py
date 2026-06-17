"""Settings screen view — composes the form with persistence + lifecycle.

The view holds a :class:`SettingsForm` (inputs + validation). On Save it just
persists the settings, applies the appearance mode, and invalidates the word
list's due-time cache — the recall threshold / max interval are applied *live*
from the stored curve params, so no model recompute is needed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import customtkinter as ctk

from src.settings import load_settings, save_settings

from ..base_screen import BaseScreen
from ..theme import Fonts, Spacing
from ..widgets import build_header
from .form import SettingsForm

if TYPE_CHECKING:
    from ..app import App


class SettingsScreen(BaseScreen):
    """Edit :class:`AppSettings`; changes apply live (no schedule recompute)."""

    def __init__(self, master: App) -> None:
        self._initial = load_settings()
        self._original_appearance = self._initial.appearance_mode

        super().__init__(master)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Construct header, form body, status row, and action buttons."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header = build_header(self, title="Settings", on_back=self._go_back)
        header.grid(
            row=0, column=0, sticky="ew",
            padx=Spacing.SCREEN_PAD_X, pady=Spacing.SCREEN_PAD_Y,
        )

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=Spacing.BODY_PAD_X, pady=(8, 8))

        self._form = SettingsForm(
            body,
            initial=self._initial,
            on_appearance_change=self._on_appearance_change,
        )

        self._status_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            self, textvariable=self._status_var, font=ctk.CTkFont(**Fonts.SMALL),
        ).grid(row=2, column=0, sticky="ew", padx=Spacing.SCREEN_PAD_X, pady=(2, 2))

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.grid(row=3, column=0, pady=(8, 16))

        self._btn_cancel = ctk.CTkButton(
            btns, text="Cancel", width=100, command=self._go_back,
        )
        self._btn_cancel.pack(side="left", padx=6)

        self._btn_save = ctk.CTkButton(
            btns, text="Save", width=100, command=self._on_save,
        )
        self._btn_save.pack(side="left", padx=6)

    # ------------------------------------------------------------------
    # Field event handler (passed into SettingsForm)
    # ------------------------------------------------------------------

    def _on_appearance_change(self, mode: str) -> None:
        """Live-preview the new appearance mode without persisting it."""
        ctk.set_appearance_mode(mode)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        """Persist the form and return home; threshold/max apply live."""
        new = self._form.read()
        if new is None:
            return

        save_settings(new)
        ctk.set_appearance_mode(new.appearance_mode)
        # Threshold / max interval feed the live due-time computation, so drop
        # the cached due times — they're rebuilt with the new values on next load.
        self._app.invalidate_due_cache()
        self._app.show_db_select()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_back(self) -> None:
        """Return to the home screen, reverting any unsaved appearance preview."""
        if self._form.appearance != self._original_appearance:
            ctk.set_appearance_mode(self._original_appearance)
        self._app.show_db_select()
