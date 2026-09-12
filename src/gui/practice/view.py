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
from ..formatting import day_start, format_future, format_past
from ..theme import Fonts, Hints, PollIntervals, Spacing
from ..widgets import build_header
from .queue_model import ERROR_PRIORITY, Card, PracticeQueue, card_gain
from .state import ArrowKey, PracticeState
from .workers import answer_worker, init_worker

from src.model import HeuristicPredictor

if TYPE_CHECKING:
    from src.model import RecallEstimator

    from ..app import App

#: Appended to the result line while the untrained-pair fallback is scheduling.
HEURISTIC_NOTE = "  (heuristic — no trained model yet)"


class PracticeScreen(BaseScreen):
    """Arrow-key driven spaced-repetition session."""

    def __init__(self, master: App) -> None:
        self._predictor: RecallEstimator | None = None
        self._queue: PracticeQueue = PracticeQueue()
        self._waiting: PracticeQueue = PracticeQueue()
        self._answered_count = 0
        self._today_count = 0
        self._today_start = day_start()
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
        """Refresh the "Answered N • Today K • Remaining M" counter in the header.

        ``Answered`` counts this session only; ``Today`` counts every repetition
        recorded in this database since local midnight, earlier sessions included.
        """
        if self._state is PracticeState.LOADING:
            self._stats_var.set("")
            return
        # The currently-shown card has already been popped off the heap, so add
        # it back to the count while it's still pending an answer.
        pending = 1 if self._state in (
            PracticeState.PROMPT, PracticeState.ANSWER, PracticeState.SAVING
        ) else 0
        remaining = len(self._queue) + pending
        parts = [f"Answered {self._answered_count}", f"Today {self._today_count}"]
        if self._state is not PracticeState.DONE:
            parts.append(f"Remaining {remaining}")
        self._stats_var.set("  •  ".join(parts))

    # ------------------------------------------------------------------
    # Card flow
    # ------------------------------------------------------------------

    def _horizon(self) -> float | None:
        """Window a gain is scored over, or ``None`` when there is no estimator.

        The user's interval cap doubles as the retention horizon. The ordering is
        close to horizon-independent but not exactly so — unlike a retention
        *level*, a gain is a difference of two nearly-equal integrals, so the
        horizon survives in it. On the French deck the served order correlates
        ρ ≈ 0.98 between a 30-day and a 2-year horizon and ρ > 0.999 between a year
        and two, reshuffling only near-tied cards; which cards lead a session does
        not change. So the cap is a reasonable stand-in, but it is a mild tuning
        knob rather than a pure choice of units.
        """
        if self._predictor is None:
            return None
        return self._predictor.config.max_delta_seconds

    def _promote_due(self) -> None:
        """Move every waiting card whose due time has passed into the main heap.

        The card is re-scored on the way across rather than reusing the score it
        was built with: a gain depends on how long the card has been waiting (see
        :func:`src.model.curve.expected_gain`), and the whole point of parking it
        was that time would pass. Only a session with no estimator at all has no
        horizon to score against, and that session never re-queues anything.
        """
        now = int(time.time())
        horizon = self._horizon()
        while True:
            due_ts = self._waiting.peek_priority()
            if due_ts is None or due_ts > now:
                break
            card = self._waiting.pop()
            if card is None:
                continue
            if horizon is not None:
                card.score = card_gain(card, now, horizon)
            self._queue.push(card, -card.score)

    def _show_current(self) -> None:
        """Pop and render the next-most-urgent card, or transition to DONE when empty."""
        self._promote_due()
        card = self._queue.pop()
        if card is None:
            self._current = None
            self._set_state(PracticeState.DONE)
            return

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

    def _on_ready(
        self,
        predictor: RecallEstimator | None,
        queue: PracticeQueue,
        waiting: PracticeQueue,
        today_count: int,
    ) -> None:
        """Init worker finished — install both queues and show the first card."""
        self._predictor = predictor
        self._queue = queue
        self._waiting = waiting
        self._today_count = today_count
        self._show_current()

    def _on_init_error(self, msg: str) -> None:
        self._show_error(f"Failed to load practice session: {msg}")

    def _on_answered(
        self,
        card: Card,
        practiced_at: int,
        next_ts: int | None,
        curves: tuple[float | None, float | None, float | None],
    ) -> None:
        """Answer worker finished — show result and maybe re-queue the card."""
        self._answered_count += 1
        answer_day = day_start(practiced_at)
        if answer_day == self._today_start:
            self._today_count += 1
        else:
            # The session ran past midnight — start the day tally over.
            self._today_start = answer_day
            self._today_count = 1
        # This word's stored half-lives just changed — the word list's cached
        # due times are now stale.
        self._app.invalidate_due_cache()

        if next_ts is None:
            self._next_var.set("next repetition: —  (no estimator)")
        else:
            delta = next_ts - int(time.time())
            note = HEURISTIC_NOTE if isinstance(self._predictor, HeuristicPredictor) else ""
            self._next_var.set(f"next repetition {format_future(delta)}{note}")

        if next_ts is not None:
            current, success, failure = curves
            refreshed = Card(
                word_id=card.word_id,
                direction=card.direction,
                source_text=card.source_text,
                target_text=card.target_text,
                last_practiced=practiced_at,
                score=0.0,
                current=current,
                success=success,
                failure=failure,
            )
            now = int(time.time())
            horizon = self._horizon()
            if success is None or failure is None or horizon is None:
                # Nothing to score the card from — send it to the back rather than
                # the front.
                self._queue.push(refreshed, ERROR_PRIORITY)
            elif next_ts <= now:
                # Still due right after this attempt — re-queue at its
                # score-sorted position so it returns later in the session. It is
                # re-scored again if it waits, so this score only has to order it
                # against the queue as it stands now.
                refreshed.score = card_gain(refreshed, now, horizon)
                self._queue.push(refreshed, -refreshed.score)
            else:
                # Due in the future — park it in the waiting heap so it can be
                # promoted back if it comes due before the session ends.
                self._waiting.push(refreshed, next_ts)

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
