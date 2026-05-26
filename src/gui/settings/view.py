"""Settings screen view — composes form + worker + lifecycle.

The view holds a :class:`SettingsForm` (which encapsulates the inputs and
their validation) and a single :class:`BackgroundJob` for the
schedule-recalc worker. Saving without a scheduling-relevant change just
persists and goes home; otherwise the worker runs and the buttons are
disabled until it finishes.
"""
from __future__ import annotations

from dataclasses import replace
from tkinter import messagebox
from typing import TYPE_CHECKING

import customtkinter as ctk

from src.settings import AppSettings, load_settings, save_settings

from ..background import BackgroundJob
from ..base_screen import BaseScreen
from ..theme import Colors, Fonts, PollIntervals, Spacing
from ..widgets import build_header
from .form import SettingsForm
from .workers import RecalcTarget, iter_recalc_targets, recalc_worker

if TYPE_CHECKING:
    from ..app import App


class SettingsScreen(BaseScreen):
    """Edit :class:`AppSettings` and, on Save, recompute schedules across every DB."""

    def __init__(self, master: App) -> None:
        self._initial = load_settings()
        self._original_appearance = self._initial.appearance_mode

        super().__init__(master)

        self._recalc_job = BackgroundJob(
            self,
            handlers={
                "recalc_progress": self._on_progress,
                "recalc_done": lambda: self._on_finished("Done — schedules updated."),
                "recalc_cancelled": lambda: self._on_finished("Recalculation cancelled."),
                "recalc_error": lambda msg: self._on_finished(
                    f"Error: {msg}", error_title="Recalculation error",
                ),
            },
            poll_ms=PollIntervals.MS,
        )

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
        assert header.back_button is not None
        self._btn_back = header.back_button

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

        self._progress = ctk.CTkProgressBar(self)
        self._progress.grid(row=3, column=0, sticky="ew", padx=Spacing.SCREEN_PAD_X, pady=(0, 4))
        self._progress.set(0)
        self._progress.grid_remove()

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.grid(row=4, column=0, pady=(8, 16))

        self._btn_cancel = ctk.CTkButton(
            btns, text="Cancel", width=100, command=self._go_back,
        )
        self._btn_cancel.pack(side="left", padx=6)

        self._btn_save = ctk.CTkButton(
            btns, text="Save", width=100, command=self._on_save,
        )
        self._btn_save.pack(side="left", padx=6)

        self._btn_cancel_recalc = ctk.CTkButton(
            btns,
            text="Cancel recalc",
            width=140,
            fg_color=Colors.DANGER,
            hover_color=Colors.DANGER_HOVER,
            command=self._cancel_recalc,
        )
        self._btn_cancel_recalc.pack(side="left", padx=6)
        self._btn_cancel_recalc.pack_forget()

    # ------------------------------------------------------------------
    # Field event handler (passed into SettingsForm)
    # ------------------------------------------------------------------

    def _on_appearance_change(self, mode: str) -> None:
        """Live-preview the new appearance mode without persisting it."""
        ctk.set_appearance_mode(mode)

    # ------------------------------------------------------------------
    # Save / recalc
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        """Persist the form, then either navigate home or start a schedule recalc."""
        new = self._form.read()
        if new is None:
            return

        save_settings(new)
        ctk.set_appearance_mode(new.appearance_mode)

        needs_recalc = (
            new.recall_threshold != self._initial.recall_threshold
            or new.max_delta_seconds != self._initial.max_delta_seconds
        )
        if not needs_recalc:
            self._app.show_db_select()
            return

        targets = iter_recalc_targets()
        if not targets:
            messagebox.showinfo(
                "Saved",
                "Settings saved. No trained models exist yet, so no schedules need recalculation.",
                parent=self,
            )
            self._app.show_db_select()
            return

        self._start_recalc(new, targets)

    def _start_recalc(self, new_settings: AppSettings, targets: list[RecalcTarget]) -> None:
        """Disable the form and start the recalc worker."""
        self._initial = replace(new_settings)  # so re-saving without changes is a no-op

        self._btn_save.configure(state="disabled")
        self._btn_cancel.configure(state="disabled")
        self._btn_back.configure(state="disabled")
        self._form.set_enabled(False)
        self._btn_cancel_recalc.pack(side="left", padx=6)
        self._btn_cancel_recalc.configure(state="normal")

        self._progress.grid()
        self._progress.set(0)
        self._status_var.set("Starting recalculation…")

        self._recalc_job.start(
            recalc_worker,
            new_settings, targets, self._recalc_job.queue, self._recalc_job.stop_event,
        )

    # ------------------------------------------------------------------
    # BackgroundJob handlers
    # ------------------------------------------------------------------

    def _on_progress(self, pair: str, done: int, total: int, idx: int, count: int) -> None:
        """Update the progress bar: ``done/total`` words in DB ``idx`` of ``count``."""
        frac = done / total if total else 0.0
        self._progress.set(frac)
        self._status_var.set(f"Recalculating {pair} — {done}/{total}  (db {idx}/{count})")

    def _on_finished(self, message: str, *, error_title: str | None = None) -> None:
        """Re-enable the form and either pop an error dialog or return home."""
        is_error = error_title is not None
        self._progress.set(1.0 if not is_error else 0)
        self._status_var.set(message)
        self._btn_save.configure(state="normal")
        self._btn_cancel.configure(state="normal")
        self._btn_back.configure(state="normal")
        self._form.set_enabled(True)
        self._btn_cancel_recalc.pack_forget()
        if is_error:
            messagebox.showerror(error_title, message, parent=self)
        else:
            self._app.show_db_select()

    def _cancel_recalc(self) -> None:
        """Signal the recalc worker to stop after the current DB finishes."""
        if self._recalc_job.is_running:
            self._recalc_job.stop()
            self._btn_cancel_recalc.configure(state="disabled")
            self._status_var.set("Cancelling…")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_back(self) -> None:
        """Return to the home screen, reverting any unsaved appearance preview."""
        if self._recalc_job.is_running:
            return
        if self._form.appearance != self._original_appearance:
            ctk.set_appearance_mode(self._original_appearance)
        self._app.show_db_select()
