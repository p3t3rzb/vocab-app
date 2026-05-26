"""Global settings screen.

Shows the user-configurable :class:`AppSettings` fields and, on Save, triggers
a per-database schedule recalculation whenever a setting that affects
scheduling has changed. Appearance changes preview live (so the user can see
the new mode immediately) and revert if Back is pressed without saving.
"""
from __future__ import annotations

import queue
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox
from typing import TYPE_CHECKING

import customtkinter as ctk

from src.settings import AppSettings, load_settings, save_settings

if TYPE_CHECKING:
    from .app import App

STORAGE_DIR = Path(__file__).parent.parent.parent / "storage"
MODELS_DIR = STORAGE_DIR / "models"

_APPEARANCE_OPTIONS = ("Light", "Dark", "System")


def _iter_recalc_targets() -> list[tuple[Path, Path, str, str, int]]:
    """Return ``(db_path, ckpt_path, src, tgt, word_count)`` for every DB that has a trained model.

    Used to enumerate which databases the Save handler needs to recompute
    schedules for. Databases without a matching ``.pt`` checkpoint are
    skipped — the next training run will pick up the new settings.
    """
    if not STORAGE_DIR.exists():
        return []
    targets: list[tuple[Path, Path, str, str, int]] = []
    for db_path in sorted(STORAGE_DIR.glob("*.db")):
        try:
            con = sqlite3.connect(str(db_path))
            row = con.execute(
                "SELECT source_language, target_language FROM language_pair LIMIT 1"
            ).fetchone()
            count_row = con.execute("SELECT COUNT(*) FROM words").fetchone()
            con.close()
        except Exception:
            continue
        if not row:
            continue
        src, tgt = row[0], row[1]
        count = count_row[0] if count_row else 0
        ckpt = MODELS_DIR / f"{src.lower()}_{tgt.lower()}.pt"
        if ckpt.exists():
            targets.append((db_path, ckpt, src, tgt, count))
    return targets


class SettingsScreen(ctk.CTkFrame):
    """Edit :class:`AppSettings` and, on Save, recompute schedules across every DB."""

    _POLL_MS = 100

    def __init__(self, master: "App") -> None:
        super().__init__(master, corner_radius=0)
        self._app = master
        self._initial = load_settings()
        self._original_appearance = self._initial.appearance_mode

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._recalculating = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct every section of the form, plus the status/progress and buttons."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        header.grid_columnconfigure(1, weight=1)

        self._btn_back = ctk.CTkButton(
            header, text="← Back", width=80, command=self._go_back
        )
        self._btn_back.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header,
            text="Settings",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=1, padx=12, sticky="w")

        # Body
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=32, pady=(8, 8))
        body.grid_columnconfigure(1, weight=1)

        row = 0

        # --- Recall threshold ---
        ctk.CTkLabel(
            body, text="Recall threshold:", anchor="w",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(16, 2))
        row += 1

        ctk.CTkLabel(
            body,
            text=(
                "Schedule a word for review when the predicted probability of "
                "recalling it drops below this value."
            ),
            anchor="w",
            wraplength=560,
            text_color="gray",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1

        slider_row = ctk.CTkFrame(body, fg_color="transparent")
        slider_row.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        slider_row.grid_columnconfigure(0, weight=1)

        self._threshold_var = ctk.DoubleVar(value=self._initial.recall_threshold)
        self._threshold_label_var = ctk.StringVar(
            value=f"{self._initial.recall_threshold:.2f}"
        )
        self._threshold_slider = ctk.CTkSlider(
            slider_row,
            from_=0.50,
            to=0.95,
            number_of_steps=45,
            variable=self._threshold_var,
            command=self._on_threshold_change,
        )
        self._threshold_slider.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        ctk.CTkLabel(
            slider_row, textvariable=self._threshold_label_var, width=50,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=1, sticky="e")
        row += 1

        # --- Max interval ---
        ctk.CTkLabel(
            body, text="Max repetition interval (days):", anchor="w",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(20, 2))
        row += 1

        ctk.CTkLabel(
            body,
            text="Hard cap on how far into the future a word can be scheduled.",
            anchor="w",
            wraplength=560,
            text_color="gray",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1

        self._max_days_var = ctk.StringVar(
            value=str(int(round(self._initial.max_delta_seconds / 86400.0)))
        )
        self._max_days_entry = ctk.CTkEntry(
            body, textvariable=self._max_days_var, width=120
        )
        self._max_days_entry.grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1

        # --- Appearance ---
        ctk.CTkLabel(
            body, text="Appearance:", anchor="w",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(20, 4))
        row += 1

        self._appearance_var = ctk.StringVar(value=self._initial.appearance_mode)
        radio_row = ctk.CTkFrame(body, fg_color="transparent")
        radio_row.grid(row=row, column=0, columnspan=2, sticky="w")
        for i, mode in enumerate(_APPEARANCE_OPTIONS):
            ctk.CTkRadioButton(
                radio_row,
                text=mode,
                variable=self._appearance_var,
                value=mode,
                command=self._on_appearance_change,
            ).pack(side="left", padx=(0, 16))
        row += 1

        # --- Status + progress (hidden until recalc starts) ---
        self._status_var = ctk.StringVar(value="")
        self._status_label = ctk.CTkLabel(
            self, textvariable=self._status_var, font=ctk.CTkFont(size=13)
        )
        self._status_label.grid(row=2, column=0, sticky="ew", padx=16, pady=(2, 2))

        self._progress = ctk.CTkProgressBar(self)
        self._progress.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 4))
        self._progress.set(0)
        self._progress.grid_remove()

        # --- Buttons ---
        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.grid(row=4, column=0, pady=(8, 16))

        self._btn_cancel = ctk.CTkButton(
            btns, text="Cancel", width=100, command=self._go_back
        )
        self._btn_cancel.pack(side="left", padx=6)

        self._btn_save = ctk.CTkButton(
            btns, text="Save", width=100, command=self._on_save
        )
        self._btn_save.pack(side="left", padx=6)

        self._btn_cancel_recalc = ctk.CTkButton(
            btns,
            text="Cancel recalc",
            width=140,
            fg_color="#c0392b",
            hover_color="#922b21",
            command=self._cancel_recalc,
        )
        self._btn_cancel_recalc.pack(side="left", padx=6)
        self._btn_cancel_recalc.pack_forget()

    # ------------------------------------------------------------------
    # Field event handlers
    # ------------------------------------------------------------------

    def _on_threshold_change(self, value: float) -> None:
        """Update the numeric label next to the threshold slider as it drags."""
        self._threshold_label_var.set(f"{float(value):.2f}")

    def _on_appearance_change(self) -> None:
        """Live-preview the new appearance mode without persisting it."""
        ctk.set_appearance_mode(self._appearance_var.get())

    # ------------------------------------------------------------------
    # Save / recalc
    # ------------------------------------------------------------------

    def _read_form(self) -> AppSettings | None:
        """Parse and validate the form, returning a new :class:`AppSettings` or ``None``."""
        try:
            days = int(self._max_days_var.get().strip())
        except ValueError:
            messagebox.showwarning(
                "Invalid input",
                "Max repetition interval must be an integer number of days.",
                parent=self,
            )
            return None
        if not (1 <= days <= 3650):
            messagebox.showwarning(
                "Out of range",
                "Max repetition interval must be between 1 and 3650 days.",
                parent=self,
            )
            return None
        return AppSettings(
            recall_threshold=round(float(self._threshold_var.get()), 4),
            max_delta_seconds=float(days) * 86400.0,
            appearance_mode=self._appearance_var.get(),
        )

    def _on_save(self) -> None:
        """Persist the form, then either navigate home or start a schedule recalc."""
        new = self._read_form()
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

        targets = _iter_recalc_targets()
        if not targets:
            messagebox.showinfo(
                "Saved",
                "Settings saved. No trained models exist yet, so no schedules need recalculation.",
                parent=self,
            )
            self._app.show_db_select()
            return

        self._start_recalc(new, targets)

    def _start_recalc(
        self,
        new_settings: AppSettings,
        targets: list[tuple[Path, Path, str, str, int]],
    ) -> None:
        """Disable the form and spawn the worker that recalculates schedules per DB."""
        self._recalculating = True
        self._stop_event.clear()
        self._initial = replace(new_settings)  # so re-saving without changes is a no-op

        self._btn_save.configure(state="disabled")
        self._btn_cancel.configure(state="disabled")
        self._btn_back.configure(state="disabled")
        self._threshold_slider.configure(state="disabled")
        self._max_days_entry.configure(state="disabled")
        self._btn_cancel_recalc.pack(side="left", padx=6)

        self._progress.grid()
        self._progress.set(0)
        self._status_var.set("Starting recalculation…")

        self._thread = threading.Thread(
            target=self._recalc_worker,
            args=(new_settings, targets),
            daemon=True,
        )
        self._thread.start()
        self._poll()

    def _recalc_worker(
        self,
        new_settings: AppSettings,
        targets: list[tuple[Path, Path, str, str, int]],
    ) -> None:
        """Background: iterate over the target DBs and run :func:`compute_all_schedules` on each."""
        try:
            from src.database import init_db
            from src.model.schedule import compute_all_schedules

            cfg = new_settings.to_predict_config()

            total_dbs = len(targets)
            for db_idx, (db_path, ckpt, src, tgt, _count) in enumerate(targets, start=1):
                if self._stop_event.is_set():
                    break
                init_db(f"sqlite:///{db_path}", src, tgt)
                pair_label = f"{src}↔{tgt}"

                def on_progress(done: int, total: int, _pair=pair_label, _i=db_idx, _n=total_dbs) -> None:
                    self._queue.put(("recalc_progress", _pair, done, total, _i, _n))

                compute_all_schedules(
                    model_path=ckpt,
                    on_progress=on_progress,
                    stop_event=self._stop_event,
                    cfg=cfg,
                )

            if self._stop_event.is_set():
                self._queue.put(("recalc_cancelled",))
            else:
                self._queue.put(("recalc_done",))
        except Exception as exc:
            self._queue.put(("recalc_error", str(exc)))

    def _poll(self) -> None:
        """Drain recalc events and reschedule until the worker finishes."""
        try:
            while True:
                item = self._queue.get_nowait()
                tag = item[0]

                if tag == "recalc_progress":
                    _, pair, done, total, i, n = item
                    frac = done / total if total else 0.0
                    self._progress.set(frac)
                    self._status_var.set(
                        f"Recalculating {pair} — {done}/{total}  (db {i}/{n})"
                    )

                elif tag == "recalc_done":
                    self._on_recalc_finished("Done — schedules updated.")
                    return

                elif tag == "recalc_cancelled":
                    self._on_recalc_finished("Recalculation cancelled.")
                    return

                elif tag == "recalc_error":
                    _, msg = item
                    self._on_recalc_finished(f"Error: {msg}", is_error=True)
                    return

        except queue.Empty:
            pass

        if self._recalculating:
            self.after(self._POLL_MS, self._poll)

    def _on_recalc_finished(self, message: str, is_error: bool = False) -> None:
        """Re-enable the form and either show an error dialog or return home."""
        self._recalculating = False
        self._progress.set(1.0 if not is_error else 0)
        self._status_var.set(message)
        self._btn_save.configure(state="normal")
        self._btn_cancel.configure(state="normal")
        self._btn_back.configure(state="normal")
        self._threshold_slider.configure(state="normal")
        self._max_days_entry.configure(state="normal")
        self._btn_cancel_recalc.pack_forget()
        if is_error:
            messagebox.showerror("Recalculation error", message, parent=self)
        else:
            self._app.show_db_select()

    def _cancel_recalc(self) -> None:
        """Signal the recalc worker to stop after the current DB finishes."""
        if self._recalculating:
            self._stop_event.set()
            self._btn_cancel_recalc.configure(state="disabled")
            self._status_var.set("Cancelling…")

    def _go_back(self) -> None:
        """Return to the home screen, reverting any unsaved appearance preview."""
        if self._recalculating:
            return
        # If user toggled appearance for preview but never saved, revert.
        current = self._appearance_var.get()
        if current != self._original_appearance:
            ctk.set_appearance_mode(self._original_appearance)
        self._app.show_db_select()
