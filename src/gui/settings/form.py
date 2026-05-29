"""Form widgets and validation for the settings screen.

:class:`SettingsForm` owns the three input widgets (threshold slider,
max-interval entry, appearance radio buttons), the live threshold label,
and the :meth:`read` validator that returns an :class:`AppSettings` or
``None`` (after surfacing a warning dialog).

The view never touches Tk variables directly — it only calls
:meth:`read`, :meth:`set_enabled`, and the :attr:`appearance` property.
"""
from __future__ import annotations

from collections.abc import Callable
from tkinter import messagebox

import customtkinter as ctk

from src.settings import AppSettings

from ..theme import Fonts, Limits

_APPEARANCE_OPTIONS = ("Light", "Dark", "System")
_SECONDS_PER_DAY = 86400.0


class SettingsForm:
    """Group of widgets that edit an :class:`AppSettings`."""

    def __init__(
        self,
        parent: ctk.CTkBaseClass,
        *,
        initial: AppSettings,
        on_appearance_change: Callable[[str], None],
    ) -> None:
        self._parent = parent
        self._initial = initial
        self._on_appearance_change = on_appearance_change

        self._threshold_var = ctk.DoubleVar(value=initial.recall_threshold)
        self._threshold_label_var = ctk.StringVar(value=f"{initial.recall_threshold:.2f}")
        self._max_days_var = ctk.StringVar(
            value=str(int(round(initial.max_delta_seconds / _SECONDS_PER_DAY)))
        )
        self._appearance_var = ctk.StringVar(value=initial.appearance_mode)

        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Lay out the three sections inside ``parent``."""
        self._parent.grid_columnconfigure(1, weight=1)
        row = 0

        # Recall threshold
        ctk.CTkLabel(
            self._parent, text="Recall threshold:", anchor="w",
            font=ctk.CTkFont(**Fonts.SECTION),
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(16, 2))
        row += 1

        ctk.CTkLabel(
            self._parent,
            text=(
                "Schedule a word for review when the predicted probability of "
                "recalling it drops below this value."
            ),
            anchor="w", wraplength=560, text_color="gray",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1

        slider_row = ctk.CTkFrame(self._parent, fg_color="transparent")
        slider_row.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        slider_row.grid_columnconfigure(0, weight=1)

        threshold_min, threshold_max = Limits.RECALL_THRESHOLD
        self._threshold_slider = ctk.CTkSlider(
            slider_row,
            from_=threshold_min,
            to=threshold_max,
            number_of_steps=Limits.RECALL_THRESHOLD_STEPS,
            variable=self._threshold_var,
            command=self._on_threshold_change,
        )
        self._threshold_slider.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        ctk.CTkLabel(
            slider_row, textvariable=self._threshold_label_var, width=50,
            font=ctk.CTkFont(**Fonts.SECTION),
        ).grid(row=0, column=1, sticky="e")
        row += 1

        # Max interval
        ctk.CTkLabel(
            self._parent, text="Max repetition interval (days):", anchor="w",
            font=ctk.CTkFont(**Fonts.SECTION),
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(20, 2))
        row += 1

        ctk.CTkLabel(
            self._parent,
            text="Hard cap on how far into the future a word can be scheduled.",
            anchor="w", wraplength=560, text_color="gray",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1

        self._max_days_entry = ctk.CTkEntry(
            self._parent, textvariable=self._max_days_var, width=120,
        )
        self._max_days_entry.grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1

        # Appearance
        ctk.CTkLabel(
            self._parent, text="Appearance:", anchor="w",
            font=ctk.CTkFont(**Fonts.SECTION),
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(20, 4))
        row += 1

        radio_row = ctk.CTkFrame(self._parent, fg_color="transparent")
        radio_row.grid(row=row, column=0, columnspan=2, sticky="w")
        for mode in _APPEARANCE_OPTIONS:
            ctk.CTkRadioButton(
                radio_row,
                text=mode,
                variable=self._appearance_var,
                value=mode,
                command=self._handle_appearance,
            ).pack(side="left", padx=(0, 16))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_threshold_change(self, value: float) -> None:
        self._threshold_label_var.set(f"{float(value):.2f}")

    def _handle_appearance(self) -> None:
        self._on_appearance_change(self._appearance_var.get())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def appearance(self) -> str:
        """Currently selected appearance mode (may not be saved yet)."""
        return self._appearance_var.get()

    def read(self) -> AppSettings | None:
        """Validate inputs and return a new :class:`AppSettings`, or ``None`` on error."""
        try:
            days = int(self._max_days_var.get().strip())
        except ValueError:
            messagebox.showwarning(
                "Invalid input",
                "Max repetition interval must be an integer number of days.",
                parent=self._parent,
            )
            return None
        min_days, max_days = Limits.MAX_INTERVAL_DAYS
        if not (min_days <= days <= max_days):
            messagebox.showwarning(
                "Out of range",
                f"Max repetition interval must be between {min_days} and {max_days} days.",
                parent=self._parent,
            )
            return None
        return AppSettings(
            recall_threshold=round(float(self._threshold_var.get()), 4),
            max_delta_seconds=float(days) * _SECONDS_PER_DAY,
            appearance_mode=self._appearance_var.get(),
        )

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable all editable widgets in the form."""
        state = "normal" if enabled else "disabled"
        self._threshold_slider.configure(state=state)
        self._max_days_entry.configure(state=state)
