"""Training screen view — UI, controls, and worker coordination.

Holds two sequential :class:`BackgroundJob`s (training, then schedule
recalc) plus a :class:`LossPlot`. The view decides which controls are
enabled in each phase; the plot owns its own matplotlib figure; the
workers contain no Tk imports.
"""
from __future__ import annotations

from pathlib import Path
from tkinter import messagebox
import tkinter as tk
from typing import TYPE_CHECKING

import customtkinter as ctk

from src.model.config import TrainConfig

from ..background import BackgroundJob
from ..base_screen import BaseScreen
from ..theme import Colors, Defaults, Fonts, Limits, PollIntervals, Spacing
from ..widgets import build_header
from .plot import LossPlot
from .workers import schedule_worker, training_worker

if TYPE_CHECKING:
    from ..app import App


class TrainScreen(BaseScreen):
    """Train a model, plot live loss curves, then auto-recalc all schedules."""

    def __init__(self, master: App) -> None:
        super().__init__(master)

        self._plot = LossPlot(self._plot_label)

        self._train_job = BackgroundJob(
            self,
            handlers={
                "epoch": self._on_epoch,
                "done": self._on_training_done,
                "cancelled": self._on_training_cancelled,
                "error": self._on_training_error,
            },
            poll_ms=PollIntervals.MS,
        )
        self._schedule_job = BackgroundJob(
            self,
            handlers={
                "schedule_progress": self._on_schedule_progress,
                "schedules_done": self._on_schedules_done,
                "schedules_cancelled": self._on_schedules_cancelled,
                "schedules_error": self._on_schedules_error,
            },
            poll_ms=PollIntervals.MS,
        )

        # Schedule the initial blank plot draw once the widget has a size.
        self._initial_render_after_id = self.after(100, self._plot.render)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the header, plot area, status line, progress bar, and controls."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header = build_header(
            self,
            title=f"Train Model  —  {self._ctx.src_lang} ↔ {self._ctx.tgt_lang}",
            on_back=self._go_back,
        )
        header.grid(
            row=0, column=0, sticky="ew",
            padx=Spacing.SCREEN_PAD_X, pady=Spacing.SCREEN_PAD_Y,
        )

        plot_frame = ctk.CTkFrame(self, fg_color="transparent")
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=Spacing.SCREEN_PAD_X, pady=(4, 4))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        self._plot_label = tk.Label(plot_frame, borderwidth=0, bg=Colors.PLOT_DARK_BG)
        self._plot_label.grid(row=0, column=0, sticky="nsew")

        self._status_var = ctk.StringVar(value="Ready. Set epoch count and press Train.")
        ctk.CTkLabel(
            self, textvariable=self._status_var, font=ctk.CTkFont(**Fonts.SMALL)
        ).grid(row=2, column=0, sticky="ew", padx=Spacing.SCREEN_PAD_X, pady=(2, 2))

        self._progress = ctk.CTkProgressBar(self, mode="indeterminate")
        self._progress.grid(row=3, column=0, sticky="ew", padx=Spacing.SCREEN_PAD_X, pady=(0, 4))
        self._progress.set(0)

        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.grid(row=4, column=0, pady=(4, 16))

        ctk.CTkLabel(ctrl, text="Epochs:").pack(side="left", padx=(0, 4))
        self._epochs_var = ctk.StringVar(value=str(Defaults.EPOCHS))
        self._epochs_entry = ctk.CTkEntry(ctrl, textvariable=self._epochs_var, width=60)
        self._epochs_entry.pack(side="left", padx=(0, 20))

        self._btn_train = ctk.CTkButton(
            ctrl, text="Train", width=100, command=self._start_training,
        )
        self._btn_train.pack(side="left", padx=6)

        self._btn_cancel = ctk.CTkButton(
            ctrl,
            text="Cancel",
            width=100,
            state="disabled",
            fg_color=Colors.DANGER,
            hover_color=Colors.DANGER_HOVER,
            command=self._cancel_active_job,
        )
        self._btn_cancel.pack(side="left", padx=6)

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def _start_training(self) -> None:
        """Validate inputs and start the training worker."""
        if self._train_job.is_running:
            return
        try:
            epochs = int(self._epochs_var.get())
            if epochs < Limits.EPOCHS_MIN:
                raise ValueError
        except ValueError:
            messagebox.showwarning(
                "Invalid input", "Epochs must be a positive integer.", parent=self,
            )
            return

        self._plot.reset()

        self._btn_train.configure(state="disabled")
        self._epochs_entry.configure(state="disabled")
        self._btn_cancel.configure(state="normal")
        self._status_var.set("Starting training…")
        self._progress.start()

        cfg = TrainConfig(epochs=epochs)
        self._train_job.start(
            training_worker,
            self._ctx.db_url,
            self._ctx.src_lang,
            self._ctx.tgt_lang,
            cfg,
            self._train_job.queue,
            self._train_job.stop_event,
        )

    # ------------------------------------------------------------------
    # Training-job handlers
    # ------------------------------------------------------------------

    def _on_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self._plot.append(epoch, train_loss, val_loss)
        total = self._epochs_var.get()
        self._status_var.set(
            f"Epoch {epoch}/{total} — Train: {train_loss:.5f}  |  Val: {val_loss:.5f}"
        )

    def _on_training_done(self, checkpoint_path: str) -> None:
        """Training finished — start the schedule recalc phase."""
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._progress.set(0)
        self._status_var.set("Computing schedules…")
        self._btn_cancel.configure(state="normal")

        self._schedule_job.start(
            schedule_worker,
            Path(checkpoint_path), self._schedule_job.queue, self._schedule_job.stop_event,
        )

    def _on_training_cancelled(self) -> None:
        self._reset_controls("Training cancelled.")

    def _on_training_error(self, msg: str) -> None:
        self._reset_controls(f"Error: {msg}", error_title="Training error")

    # ------------------------------------------------------------------
    # Schedule-job handlers
    # ------------------------------------------------------------------

    def _on_schedule_progress(self, done: int, total: int) -> None:
        frac = done / total if total else 0.0
        self._progress.set(frac)
        self._status_var.set(f"Computing schedules… {done}/{total}")

    def _on_schedules_done(self) -> None:
        self._reset_controls("Done — model trained and schedules updated.", success=True)

    def _on_schedules_cancelled(self) -> None:
        self._reset_controls("Schedule computation cancelled.")

    def _on_schedules_error(self, msg: str) -> None:
        self._reset_controls(f"Schedule error: {msg}", error_title="Schedule error")

    # ------------------------------------------------------------------
    # Shared UI reset / cancel
    # ------------------------------------------------------------------

    def _reset_controls(
        self,
        message: str,
        *,
        error_title: str | None = None,
        success: bool = False,
    ) -> None:
        """Restore the controls to their idle state and show a status message."""
        self._progress.stop()
        self._progress.configure(mode="indeterminate")
        self._progress.set(1.0 if success else 0)
        self._btn_train.configure(state="normal")
        self._epochs_entry.configure(state="normal")
        self._btn_cancel.configure(state="disabled")
        self._status_var.set(message)
        if error_title is not None:
            messagebox.showerror(error_title, message, parent=self)

    def _cancel_active_job(self) -> None:
        """Signal whichever job is running to stop."""
        if self._train_job.is_running:
            self._train_job.stop()
            self._btn_cancel.configure(state="disabled")
            self._status_var.set("Cancelling… waiting for current epoch to finish.")
        elif self._schedule_job.is_running:
            self._schedule_job.stop()
            self._btn_cancel.configure(state="disabled")
            self._status_var.set("Cancelling schedule computation…")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def on_destroy(self) -> None:
        """Stop owned jobs (via super), cancel the initial render, free the plot."""
        super().on_destroy()
        if self._initial_render_after_id is not None:
            try:
                self.after_cancel(self._initial_render_after_id)
            except Exception:
                pass
            self._initial_render_after_id = None
        self._plot.destroy()

    def _go_back(self) -> None:
        """Navigate to the word list, confirming if work is in progress."""
        if self._train_job.is_running:
            confirmed = messagebox.askyesno(
                "Training in progress",
                "Training is still running. Cancel it and go back?",
                parent=self,
            )
            if not confirmed:
                return
            self._train_job.stop()
        elif self._schedule_job.is_running:
            confirmed = messagebox.askyesno(
                "Computing schedules",
                "Schedule computation is running. Cancel it and go back?\n\n"
                "Due times will be updated next time you train.",
                parent=self,
            )
            if not confirmed:
                return
            self._schedule_job.stop()
        self._app.back_to_word_list()
