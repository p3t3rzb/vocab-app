"""Root window for the customtkinter desktop GUI.

The :class:`App` class is the top-level ``CTk`` window. It owns the
currently visible frame, the active :class:`DbContext`, and exposes one
``show_*`` method per screen — each one destroys the old frame and
instantiates the new one. Screen imports are deferred so the import graph
stays free of cycles and startup remains cheap.
"""
from __future__ import annotations

import math
import tkinter as tk
from types import SimpleNamespace
from typing import TYPE_CHECKING

import customtkinter as ctk

from src.settings import load_settings

from .db_context import DbContext
from .theme import WindowSizes
from .widgets import apply_treeview_style, rescale_tree_columns

if TYPE_CHECKING:
    from .base_screen import BaseScreen


class App(ctk.CTk):
    """The main application window and navigation router."""

    def __init__(self) -> None:
        """Build the window and show the database-selection screen."""
        super().__init__()
        self.title("Vocab Repetition")
        self.geometry(WindowSizes.MAIN)
        min_w, min_h = WindowSizes.MAIN_MIN
        self.minsize(
            int(min_w * WindowSizes.MIN_UI_SCALE), int(min_h * WindowSizes.MIN_UI_SCALE)
        )

        ctk.set_appearance_mode(load_settings().appearance_mode)

        self._ui_scale = 1.0
        self._rescale_after_id: str | None = None
        self.bind("<Configure>", self._on_configure, add="+")
        # Tk under-reports font descent at small pixel sizes (1 px at 6 px), so
        # a CTkLabel's inner label clips descenders once the UI is scaled down.
        # One pixel of padding gives them room; at full size it's invisible
        # because the text sits well inside the label's 28 px height.
        self.bind_class("Label", "<Map>", self._pad_ctk_label, add="+")

        self._current_frame: ctk.CTkFrame | None = None
        self._ctx: DbContext | None = None
        # Word-list due-time cache: {word_id: (fwd_due_ts, rev_due_ts)}. Built
        # once per database (it's derived from stored half-lives, not the
        # clock) and reused across navigations. ``None`` means "rebuild on next
        # word-list load"; invalidated when params or the recall threshold change.
        self._due_cache: dict[int, tuple[int | None, int | None]] | None = None

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
        """Remember ``ctx`` as the active database (invalidating the cache on a DB switch)."""
        if self._ctx is None or self._ctx.db_url != ctx.db_url:
            self.invalidate_due_cache()
        self._ctx = ctx

    # ------------------------------------------------------------------
    # Word-list due-time cache
    # ------------------------------------------------------------------

    @property
    def due_cache(self) -> dict[int, tuple[int | None, int | None]] | None:
        """The cached per-word due timestamps, or ``None`` if it needs rebuilding."""
        return self._due_cache

    def set_due_cache(self, cache: dict[int, tuple[int | None, int | None]]) -> None:
        """Store a freshly built due-time cache for the active database."""
        self._due_cache = cache

    def invalidate_due_cache(self) -> None:
        """Drop the due-time cache so the word list rebuilds it on next load.

        Called whenever the stored half-lives or the live recall threshold
        change (practice answers, training completion, settings save, DB switch).
        """
        self._due_cache = None

    # ------------------------------------------------------------------
    # Window-size-driven UI scaling
    # ------------------------------------------------------------------

    def _on_configure(self, event: tk.Event) -> None:
        """Debounce window resizes into a single :meth:`_rescale` call.

        A binding on the root window also fires for every child's
        ``<Configure>``, so only the root's own events are considered.
        """
        if event.widget is not self:
            return
        if self._rescale_after_id is not None:
            self.after_cancel(self._rescale_after_id)
        self._rescale_after_id = self.after(WindowSizes.RESCALE_DEBOUNCE_MS, self._rescale)

    def _rescale(self) -> None:
        """Shrink the UI proportionally once the window is smaller than ``MAIN_MIN``.

        Uses customtkinter's global widget scaling, which resizes fonts,
        widget dimensions, and paddings of every CTk widget together. ttk
        treeviews are outside that mechanism, so they're rescaled separately.
        """
        self._rescale_after_id = None
        min_w, min_h = WindowSizes.MAIN_MIN
        width = self._reverse_window_scaling(self.winfo_width())
        height = self._reverse_window_scaling(self.winfo_height())
        step = WindowSizes.UI_SCALE_STEP
        scale = min(1.0, width / min_w, height / min_h)
        # Round down: rounding up would make the content slightly too big to fit.
        # The small epsilon absorbs float error so an exact multiple isn't floored a step lower.
        scale = max(WindowSizes.MIN_UI_SCALE, math.floor(scale / step + 1e-6) * step)
        if abs(scale - self._ui_scale) < step / 2:
            return
        old_scale, self._ui_scale = self._ui_scale, scale
        ctk.set_widget_scaling(scale)
        apply_treeview_style()
        rescale_tree_columns(self, scale / old_scale)
        self.update_idletasks()
        self._resync_widget_sizes(self)
        on_rescale = getattr(self._current_frame, "on_rescale", None)
        if callable(on_rescale):
            on_rescale()

    @staticmethod
    def _pad_ctk_label(event: tk.Event) -> None:
        """Give the text label inside a CTkLabel 1 px of vertical padding."""
        if isinstance(event.widget.master, ctk.CTkLabel):
            event.widget.configure(pady=1)

    def _resync_widget_sizes(self, widget: tk.Misc) -> None:
        """Redraw CTk widgets whose pixel size didn't change across a rescale.

        customtkinter stores each widget's size in unscaled units and only
        refreshes it on a ``<Configure>`` event. A widget stretched by
        ``fill``/``sticky`` keeps its pixel size when the scale changes, so no
        event fires and it would keep drawing its background at the stale
        size — leaving unpainted patches. Feeding it its real size fixes that.
        """
        for child in widget.winfo_children():
            if isinstance(child, ctk.CTkBaseClass):
                child._update_dimensions_event(
                    SimpleNamespace(width=child.winfo_width(), height=child.winfo_height())
                )
            self._resync_widget_sizes(child)

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
