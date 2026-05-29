"""Practice screen view — UI, key handling, and state-machine dispatch.

The screen owns two :class:`BackgroundJob` instances (an init job for the
queue/model loader, and a per-answer job for predictor calls) and tracks a
:class:`PracticeState` that decides which widgets are visible.
"""
from __future__ import annotations

import time
from tkinter import messagebox
from typing import TYPE_CHECKING

import customtkinter as ctk

from src.database import init_db

from ..background import BackgroundJob
from ..base_screen import BaseScreen
from ..formatting import format_future, format_past
from ..theme import Fonts, Hints, PollIntervals, Spacing
from ..widgets import build_header
from .queue_model import Card
from .state import ArrowKey, PracticeState
from .workers import answer_worker, init_worker

if TYPE_CHECKING:
    from src.model import Predictor

    from ..app import App


class PracticeScreen(BaseScreen):
    """Arrow-key driven spaced-repetition session."""

    def __init__(self, master: App) -> None:
        self._predictor: Predictor | None = None
        self._cards: list[Card] = []
        self._idx = 0
        self._answered_count = 0
        self._state: PracticeState = PracticeState.LOADING
        self._current: Card | None = None
        self._keys_bound = False

        super().__init__(master)

        init_db(self._ctx.db_url, self._ctx.src_lang, self._ctx.tgt_lang)

        self._init_job = BackgroundJob(
            self,
            handlers={
                "ready": self._on_ready,
                "error": self._on_init_error,
            },
            poll_ms=PollIntervals.MS,
        )
        self._answer_job = BackgroundJob(
            self,
            handlers={
                "answered": self._on_answered,
                "answer_error": self._on_answer_error,
            },
            poll_ms=PollIntervals.MS,
        )

        self._bind_keys()
        self._init_job.start(init_worker, self._ctx, self._init_job.queue)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the header, prompt/answer area, and the hint bar."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._stats_var = ctk.StringVar(value="")

        def _add_stats(parent: ctk.CTkFrame) -> ctk.CTkLabel:
            label = ctk.CTkLabel(
                parent,
                textvariable=self._stats_var,
                font=ctk.CTkFont(**Fonts.SMALL),
            )
            label.grid(row=0, column=2, sticky="e")
            return label

        header = build_header(
            self,
            title=f"Practice  —  {self._ctx.src_lang} ↔ {self._ctx.tgt_lang}",
            on_back=self._go_back,
            right_widget_factory=_add_stats,
        )
        header.grid(
            row=0, column=0, sticky="ew",
            padx=Spacing.SCREEN_PAD_X, pady=Spacing.SCREEN_PAD_Y,
        )

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=Spacing.SCREEN_PAD_X, pady=(20, 8))
        content.grid_rowconfigure(0, weight=1)
        content.grid_rowconfigure(7, weight=1)
        content.grid_columnconfigure(0, weight=1)

        self._direction_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            content,
            textvariable=self._direction_var,
            font=ctk.CTkFont(**Fonts.BODY),
            text_color=("gray40", "gray70"),
        ).grid(row=1, column=0, pady=(0, 16))

        self._prompt_var = ctk.StringVar(value="Loading…")
        ctk.CTkLabel(
            content,
            textvariable=self._prompt_var,
            font=ctk.CTkFont(**Fonts.PROMPT),
            wraplength=700,
        ).grid(row=2, column=0, pady=(0, 12))

        self._sep_label = ctk.CTkLabel(
            content,
            text="─ ─ ─ ─ ─",
            text_color=("gray60", "gray50"),
        )
        self._sep_label.grid(row=3, column=0, pady=(8, 8))

        self._answer_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            content,
            textvariable=self._answer_var,
            font=ctk.CTkFont(**Fonts.ANSWER),
            wraplength=700,
        ).grid(row=4, column=0, pady=(0, 24))

        self._last_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            content,
            textvariable=self._last_var,
            font=ctk.CTkFont(**Fonts.SMALL),
            text_color=("gray40", "gray70"),
        ).grid(row=5, column=0, pady=(0, 4))

        self._next_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            content,
            textvariable=self._next_var,
            font=ctk.CTkFont(**Fonts.BODY_BOLD),
        ).grid(row=6, column=0, pady=(0, 4))

        self._hint_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            self,
            textvariable=self._hint_var,
            font=ctk.CTkFont(**Fonts.SMALL),
            text_color=("gray30", "gray70"),
        ).grid(row=2, column=0, sticky="ew", padx=Spacing.SCREEN_PAD_X, pady=(0, 16))

        self._set_state(PracticeState.LOADING)

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _bind_keys(self) -> None:
        """Bind arrow-key handlers on the root window (idempotent)."""
        if self._keys_bound:
            return
        root = self._app
        root.bind("<Down>", lambda _e: self._on_key(ArrowKey.DOWN))
        root.bind("<Up>", lambda _e: self._on_key(ArrowKey.UP))
        root.bind("<Left>", lambda _e: self._on_key(ArrowKey.LEFT))
        root.bind("<Right>", lambda _e: self._on_key(ArrowKey.RIGHT))
        self._keys_bound = True

    def _unbind_keys(self) -> None:
        """Remove the root-level arrow-key handlers (idempotent)."""
        if not self._keys_bound:
            return
        root = self._app
        root.unbind("<Down>")
        root.unbind("<Up>")
        root.unbind("<Left>")
        root.unbind("<Right>")
        self._keys_bound = False

    def on_destroy(self) -> None:
        """Stop owned jobs (via super) and detach root-level key bindings."""
        super().on_destroy()
        self._unbind_keys()

    def _on_key(self, key: ArrowKey) -> None:
        """Dispatch an arrow keypress according to the current state."""
        if self._state is PracticeState.PROMPT and key is ArrowKey.DOWN:
            self._reveal_answer()
        elif self._state is PracticeState.ANSWER:
            if key is ArrowKey.RIGHT:
                self._submit_answer(remembered=True)
            elif key is ArrowKey.LEFT:
                self._submit_answer(remembered=False)
        elif self._state is PracticeState.RESULT:
            self._advance()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _set_state(self, state: PracticeState) -> None:
        """Update the visible widgets and hint bar to reflect the new state."""
        self._state = state

        if state is PracticeState.LOADING:
            self._direction_var.set("")
            self._prompt_var.set("Loading…")
            self._sep_label.grid_remove()
            self._answer_var.set("")
            self._last_var.set("")
            self._next_var.set("")
            self._hint_var.set("")

        elif state is PracticeState.PROMPT:
            self._sep_label.grid_remove()
            self._answer_var.set("")
            self._next_var.set("")
            self._hint_var.set(Hints.PROMPT_DOWN)

        elif state is PracticeState.ANSWER:
            self._sep_label.grid()
            self._next_var.set("")
            self._hint_var.set(Hints.ANSWER_BAR)

        elif state is PracticeState.SAVING:
            self._hint_var.set(Hints.SAVING_BAR)

        elif state is PracticeState.RESULT:
            self._sep_label.grid()
            self._hint_var.set(Hints.RESULT_BAR)

        elif state is PracticeState.DONE:
            self._direction_var.set("")
            self._prompt_var.set("No more words to repeat now.")
            self._sep_label.grid_remove()
            self._answer_var.set("")
            self._last_var.set("")
            self._next_var.set("")
            self._hint_var.set(Hints.DONE_BAR)

        self._update_stats()

    def _update_stats(self) -> None:
        """Refresh the "Answered N • Remaining M" counter in the header."""
        if self._state is PracticeState.LOADING:
            self._stats_var.set("")
            return
        remaining = max(0, len(self._cards) - self._idx)
        if self._state is PracticeState.DONE:
            self._stats_var.set(f"Answered {self._answered_count}")
        else:
            self._stats_var.set(
                f"Answered {self._answered_count}  •  Remaining {remaining}"
            )

    # ------------------------------------------------------------------
    # Card flow
    # ------------------------------------------------------------------

    def _show_current(self) -> None:
        """Render the card at ``_idx``, or transition to DONE when exhausted."""
        if self._idx >= len(self._cards):
            self._set_state(PracticeState.DONE)
            return

        card = self._cards[self._idx]
        self._current = card

        self._direction_var.set(card.direction_label(self._ctx.src_lang, self._ctx.tgt_lang))
        self._prompt_var.set(card.prompt_text())

        if card.last_practiced is None:
            self._last_var.set("never practiced before")
        else:
            elapsed = int(time.time()) - card.last_practiced
            self._last_var.set(f"last revised {format_past(elapsed)}")

        self._set_state(PracticeState.PROMPT)

    def _reveal_answer(self) -> None:
        """PROMPT → ANSWER: reveal the translation."""
        if self._current is not None:
            self._answer_var.set(self._current.answer_text())
        self._set_state(PracticeState.ANSWER)

    def _submit_answer(self, remembered: bool) -> None:
        """ANSWER → SAVING: dispatch the per-answer worker."""
        if self._current is None:
            return
        card = self._current
        self._set_state(PracticeState.SAVING)
        self._answer_job.start(
            answer_worker, card, remembered, self._predictor, self._answer_job.queue,
        )

    def _advance(self) -> None:
        """RESULT → PROMPT (or DONE): move to the next card."""
        self._show_current()

    # ------------------------------------------------------------------
    # BackgroundJob handlers
    # ------------------------------------------------------------------

    def _on_ready(self, predictor: Predictor | None, cards: list[Card]) -> None:
        """Init worker finished — install the queue and show the first card."""
        self._predictor = predictor
        self._cards = cards
        self._idx = 0
        self._show_current()

    def _on_init_error(self, msg: str) -> None:
        self._show_error(f"Failed to load practice session: {msg}")

    def _on_answered(self, card: Card, practiced_at: int, next_ts: int | None) -> None:
        """Answer worker finished — show result and maybe re-queue the card."""
        self._answered_count += 1

        if next_ts is None:
            self._next_var.set("next repetition: —  (no trained model)")
        else:
            delta = next_ts - int(time.time())
            self._next_var.set(f"next repetition {format_future(delta)}")

        if next_ts is not None and next_ts <= int(time.time()):
            # Card is still due now — append a refreshed copy to the queue.
            refreshed = Card(
                word_id=card.word_id,
                direction=card.direction,
                source_text=card.source_text,
                target_text=card.target_text,
                last_practiced=practiced_at,
            )
            self._cards.append(refreshed)

        self._idx += 1
        self._set_state(PracticeState.RESULT)

    def _on_answer_error(self, msg: str) -> None:
        self._show_error(f"Failed to save answer: {msg}")

    # ------------------------------------------------------------------
    # Navigation / errors
    # ------------------------------------------------------------------

    def _show_error(self, msg: str) -> None:
        """Pop up an error dialog and bounce back to the word list."""
        messagebox.showerror("Practice error", msg, parent=self)
        self._app.back_to_word_list()

    def _go_back(self) -> None:
        """Detach key bindings and return to the word list."""
        self._unbind_keys()
        self._app.back_to_word_list()
