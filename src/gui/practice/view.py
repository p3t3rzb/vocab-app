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

from src.database import WordRepository, get_session, init_db

from ..background import BackgroundJob
from ..base_screen import BaseScreen
from ..formatting import day_start, format_future, format_past
from ..theme import Colors, Fonts, Hints, PollIntervals, Spacing
from ..widgets import build_header
from .queue_model import (
    ERROR_PRIORITY,
    TIER_EXTRA,
    TIER_REVIEW,
    Card,
    PracticeQueue,
    card_gain,
    drain_waiting,
    park_not_due,
    refresh_top,
)
from .state import ArrowKey, PracticeState
from .workers import answer_worker, init_worker, refresh_post_rep_curves

from src.model import HeuristicPredictor
from src.model.curve import Curve

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
        self._use_threshold = True
        self._remembered_count = 0
        self._forgot_count = 0
        # Outcome of the answer being saved, read back when the worker finishes.
        self._pending_remembered = False
        self._result_color: str | tuple[str, str] = Colors.CARD_BORDER
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
        """Build the brand-band header, the progress bar, and the word card."""
        self.configure(fg_color=Colors.PRACTICE_BG)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header = build_header(
            self,
            title=f"{self._ctx.src_lang} ↔ {self._ctx.tgt_lang}",
            on_back=self._go_back,
            right_widget_factory=self._build_stats,
        )
        header.title_label.configure(
            font=ctk.CTkFont(**Fonts.BODY_BOLD), text_color=Colors.ON_BRAND
        )
        assert header.back_button is not None
        header.back_button.configure(
            text="‹ Words",
            text_color=Colors.ON_BRAND_MUTED,
            hover_color=Colors.BRAND_HOVER,
        )
        header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=Spacing.SCREEN_PAD_X,
            pady=Spacing.SCREEN_PAD_Y,
        )

        # Session progress: answered cards out of answered + still queued, split
        # into a remembered (green) and a forgot (red) segment.
        track = ctk.CTkFrame(
            self, height=4, corner_radius=2, fg_color=Colors.PROGRESS_TRACK
        )
        track.grid(row=1, column=0, sticky="ew", padx=Spacing.SCREEN_PAD_X)
        self._progress_good = ctk.CTkFrame(
            track, height=4, corner_radius=2, fg_color=Colors.GREEN
        )
        self._progress_bad = ctk.CTkFrame(
            track, height=4, corner_radius=2, fg_color=Colors.RED_ON_BRAND
        )

        card = ctk.CTkFrame(
            self,
            fg_color=Colors.SURFACE,
            corner_radius=Spacing.CARD_RADIUS,
            border_width=3,
            border_color=Colors.CARD_BORDER,
        )
        card.grid(
            row=2, column=0, sticky="nsew", padx=Spacing.CARD_PAD_X, pady=(28, 36)
        )
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)
        self._card = card

        # Top row: direction chip + Ignore button
        top = ctk.CTkFrame(card, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=28, pady=(22, 0))
        top.grid_columnconfigure(0, weight=1)

        self._direction_var = ctk.StringVar(value="")
        self._direction_chip = ctk.CTkLabel(
            top,
            textvariable=self._direction_var,
            font=ctk.CTkFont(**Fonts.CHIP),
            fg_color=Colors.CHIP_BG,
            text_color=Colors.TEXT,
            corner_radius=12,
            height=24,
        )
        self._direction_chip.grid(row=0, column=0, sticky="w")

        self._ignore_button = ctk.CTkButton(
            top,
            text="Ignore word",
            width=96,
            height=28,
            corner_radius=8,
            fg_color="transparent",
            border_width=1,
            border_color=Colors.BORDER,
            hover_color=Colors.CHIP_BG,
            text_color=Colors.TEXT_MUTED,
            font=ctk.CTkFont(**Fonts.SMALL),
            command=self._ignore_current,
        )
        self._ignore_button.grid(row=0, column=1, sticky="e")

        # Middle: prompt, divider, answer, last-revised note
        middle = ctk.CTkFrame(card, fg_color="transparent")
        middle.grid(row=1, column=0, sticky="nsew", padx=28)
        middle.grid_columnconfigure(0, weight=1)
        middle.grid_rowconfigure(0, weight=1)
        middle.grid_rowconfigure(5, weight=1)
        self._content = middle
        middle.bind("<Configure>", lambda _e: self._fit_wraplength(), add="+")

        self._fonts = {
            "prompt": ctk.CTkFont(**Fonts.PROMPT),
            "revealed": ctk.CTkFont(**Fonts.PROMPT_REVEALED),
            "message": ctk.CTkFont(**Fonts.MESSAGE),
        }
        self._prompt_var = ctk.StringVar(value="Loading…")
        self._prompt_label = ctk.CTkLabel(
            middle,
            textvariable=self._prompt_var,
            font=self._fonts["prompt"],
            wraplength=560,
        )
        self._prompt_label.grid(row=1, column=0)

        self._sep = ctk.CTkFrame(
            middle, width=48, height=3, corner_radius=2, fg_color=Colors.BORDER
        )
        self._sep.grid(row=2, column=0, pady=12)

        self._answer_var = ctk.StringVar(value="")
        self._answer_label = ctk.CTkLabel(
            middle,
            textvariable=self._answer_var,
            font=ctk.CTkFont(**Fonts.ANSWER),
            wraplength=560,
        )
        self._answer_label.grid(row=3, column=0)

        self._last_var = ctk.StringVar(value="")
        self._last_label = ctk.CTkLabel(
            middle,
            textvariable=self._last_var,
            font=ctk.CTkFont(**Fonts.SMALL),
            text_color=Colors.TEXT_MUTED,
        )
        self._last_label.grid(row=4, column=0, pady=(6, 0))

        # Bottom bar: hint / answer buttons / result, stacked in one cell and
        # swapped per state by raising one over the others. They stay mapped:
        # re-gridding a CTk widget redraws it a frame late, which flickered.
        # Fixed height so swapping them never makes the card jump.
        bottom = ctk.CTkFrame(card, fg_color="transparent", height=48)
        bottom.grid(row=2, column=0, sticky="ew", padx=28, pady=(0, 22))
        bottom.grid_propagate(False)
        bottom.grid_rowconfigure(0, weight=1)
        bottom.grid_columnconfigure(0, weight=1)

        self._hint_var = ctk.StringVar(value="")
        self._hint_label = ctk.CTkLabel(
            bottom,
            textvariable=self._hint_var,
            font=ctk.CTkFont(**Fonts.SMALL),
            text_color=Colors.TEXT_MUTED,
        )
        self._hint_label.grid(row=0, column=0, sticky="nsew")

        self._answer_buttons = ctk.CTkFrame(bottom, fg_color="transparent")
        self._answer_buttons.grid(row=0, column=0, sticky="nsew")
        self._answer_buttons.grid_rowconfigure(0, weight=1)
        self._answer_buttons.grid_columnconfigure((0, 3), weight=1)
        button_font = ctk.CTkFont(**Fonts.BODY_BOLD)
        # Routed through _on_key so a click obeys the same state gating as the
        # arrow keys (e.g. a second click while SAVING is ignored).
        ctk.CTkButton(
            self._answer_buttons,
            text=Hints.FORGOT_BUTTON,
            width=200,
            height=44,
            corner_radius=22,
            fg_color=Colors.RED_TINT,
            hover_color=Colors.RED_TINT_HOVER,
            text_color=Colors.RED_TINT_TEXT,
            font=button_font,
            command=lambda: self._on_key(ArrowKey.LEFT),
        ).grid(row=0, column=1, padx=8)
        ctk.CTkButton(
            self._answer_buttons,
            text=Hints.REMEMBERED_BUTTON,
            width=200,
            height=44,
            corner_radius=22,
            fg_color=Colors.GREEN_TINT,
            hover_color=Colors.GREEN_TINT_HOVER,
            text_color=Colors.GREEN_TINT_TEXT,
            font=button_font,
            command=lambda: self._on_key(ArrowKey.RIGHT),
        ).grid(row=0, column=2, padx=8)

        self._result_frame = ctk.CTkFrame(bottom, fg_color="transparent")
        self._result_frame.grid(row=0, column=0, sticky="nsew")
        self._result_frame.grid_rowconfigure(0, weight=1)
        self._result_frame.grid_columnconfigure((0, 4), weight=1)
        self._result_pill = ctk.CTkLabel(
            self._result_frame,
            text="",
            font=ctk.CTkFont(**Fonts.CHIP),
            corner_radius=14,
            height=28,
        )
        self._result_pill.grid(row=0, column=1)
        self._next_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            self._result_frame,
            textvariable=self._next_var,
            font=ctk.CTkFont(**Fonts.BODY),
        ).grid(row=0, column=2, padx=12)
        ctk.CTkLabel(
            self._result_frame,
            text=Hints.RESULT_BAR,
            font=ctk.CTkFont(**Fonts.SMALL),
            text_color=Colors.TEXT_MUTED,
        ).grid(row=0, column=3)

        self._set_state(PracticeState.LOADING)

    def _build_stats(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        """Grid the remembered / forgot / today / left counters into the header."""
        stats = ctk.CTkFrame(parent, fg_color="transparent")
        stats.grid(row=0, column=2, sticky="e")
        font = ctk.CTkFont(**Fonts.SMALL)
        colors = {
            "remembered": Colors.GREEN,
            "forgot": Colors.RED_ON_BRAND,
            "today": Colors.ON_BRAND_MUTED,
            "left": Colors.ON_BRAND,
        }
        self._stat_labels: dict[str, ctk.CTkLabel] = {}
        for col, (key, color) in enumerate(colors.items()):
            label = ctk.CTkLabel(stats, text="", font=font, text_color=color)
            label.grid(row=0, column=col, padx=(14, 0))
            self._stat_labels[key] = label
        return stats

    def on_rescale(self) -> None:
        """Re-apply the state's layout and re-fit the wrap width after a rescale.

        customtkinter rescales a widget by replaying its last ``grid()`` call,
        which re-shows anything hidden with ``grid_remove``. The wrap width is
        stored unscaled and so goes stale too.
        """
        self._apply_layout(self._state)
        self._fit_wraplength()

    def _fit_wraplength(self) -> None:
        """Wrap the prompt and answer at the card's current inner width.

        A fixed wrap width lets long sentences overflow a narrow window.
        ``wraplength`` is given in unscaled units (customtkinter applies the
        widget scaling itself), so the pixel width is divided by the scale.
        """
        scale = ctk.ScalingTracker.get_widget_scaling(self._content)
        wraplength = max(1, int(self._content.winfo_width() / scale))
        self._prompt_label.configure(wraplength=wraplength)
        self._answer_label.configure(wraplength=wraplength)

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
        elif self._state is PracticeState.OFFER_EXTRA:
            if key is ArrowKey.RIGHT:
                self._start_extra()
            elif key is ArrowKey.LEFT:
                self._go_back()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _set_state(self, state: PracticeState) -> None:
        """Update the visible widgets, texts, and bottom bar to reflect the new state."""
        self._state = state
        self._apply_layout(state)

        if state is PracticeState.LOADING:
            self._direction_var.set("")
            self._prompt_var.set("Loading…")
            self._answer_var.set("")
            self._last_var.set("")
            self._next_var.set("")
            self._hint_var.set("")

        elif state is PracticeState.PROMPT:
            self._answer_var.set("")
            self._next_var.set("")
            self._hint_var.set(Hints.PROMPT_DOWN)

        elif state is PracticeState.ANSWER:
            self._next_var.set("")

        elif state is PracticeState.OFFER_EXTRA:
            self._direction_var.set("")
            self._prompt_var.set(
                "Nothing is due and there are no new words left.\n"
                "Keep practicing words you already know?"
            )
            self._answer_var.set("")
            self._last_var.set("")
            self._next_var.set("")
            self._hint_var.set(Hints.OFFER_EXTRA_BAR)

        elif state is PracticeState.DONE:
            self._direction_var.set("")
            self._prompt_var.set("No more words to repeat now.")
            self._answer_var.set("")
            self._last_var.set("")
            self._next_var.set("")
            self._hint_var.set(Hints.DONE_BAR)

        self._update_stats()

    def _apply_layout(self, state: PracticeState) -> None:
        """Show, hide, and restyle the card's widgets for ``state``."""
        on_card = state in (
            PracticeState.PROMPT,
            PracticeState.ANSWER,
            PracticeState.SAVING,
            PracticeState.RESULT,
        )
        revealed = state in (
            PracticeState.ANSWER,
            PracticeState.SAVING,
            PracticeState.RESULT,
        )

        # Kept up through SAVING, which lasts only a moment between ANSWER and
        # RESULT — hiding it there made it flicker on every answer.
        # `_ignore_current` ignores the click instead.
        for widget in (self._ignore_button, self._direction_chip):
            if on_card:
                widget.grid()
            else:
                widget.grid_remove()

        if revealed:
            self._prompt_label.configure(
                font=self._fonts["revealed"], text_color=Colors.TEXT_MUTED
            )
            self._sep.grid()
            self._answer_label.grid()
            self._last_label.grid_remove()
        else:
            font = self._fonts["prompt" if on_card else "message"]
            self._prompt_label.configure(font=font, text_color=Colors.TEXT)
            self._sep.grid_remove()
            self._answer_label.grid_remove()
            self._last_label.grid()

        # The answer buttons stay up through SAVING for the same no-flicker reason.
        bottom = {
            PracticeState.ANSWER: self._answer_buttons,
            PracticeState.SAVING: self._answer_buttons,
            PracticeState.RESULT: self._result_frame,
        }.get(state, self._hint_label)
        bottom.tkraise()

        self._card.configure(
            border_color=self._result_color
            if state is PracticeState.RESULT
            else Colors.CARD_BORDER
        )

    def _update_stats(self) -> None:
        """Refresh the header counters and the session progress bar.

        ``remembered`` / ``forgot`` count this session only; ``today`` counts
        every repetition recorded in this database since local midnight, earlier
        sessions included. The progress bar spans answered + still-queued cards.
        """
        if self._state is PracticeState.LOADING:
            for label in self._stat_labels.values():
                label.configure(text="")
            self._progress_good.place_forget()
            self._progress_bad.place_forget()
            return
        # The currently-shown card has already been popped off the heap, so add
        # it back to the count while it's still pending an answer.
        pending = (
            1
            if self._state
            in (PracticeState.PROMPT, PracticeState.ANSWER, PracticeState.SAVING)
            else 0
        )
        finished = self._state in (PracticeState.DONE, PracticeState.OFFER_EXTRA)
        remaining = 0 if finished else len(self._queue) + pending
        self._stat_labels["remembered"].configure(
            text=f"{self._remembered_count} remembered"
        )
        self._stat_labels["forgot"].configure(text=f"{self._forgot_count} forgot")
        self._stat_labels["today"].configure(text=f"{self._today_count} today")
        self._stat_labels["left"].configure(
            text="" if finished else f"{remaining} left"
        )

        total = self._remembered_count + self._forgot_count + remaining
        good = self._remembered_count / total if total else 0.0
        bad = self._forgot_count / total if total else 0.0
        for bar, x, width in (
            (self._progress_good, 0.0, good),
            (self._progress_bad, good, bad),
        ):
            if width > 0:
                bar.place(relx=x, rely=0, relwidth=width, relheight=1)
            else:
                bar.place_forget()

    # ------------------------------------------------------------------
    # Card flow
    # ------------------------------------------------------------------

    def _has_estimator(self) -> bool:
        """Whether this session can score a card at all.

        A gain is computed from the card's own stored time constants
        (:func:`~src.gui.practice.queue_model.card_gain`), so it needs no config —
        but a pair with no estimator at all has no time constants either, and such a
        session never re-scores or re-queues anything.
        """
        return self._predictor is not None

    def _promote_due(self) -> None:
        """Move every waiting card whose due time has passed into the main heap.

        The card is re-scored on the way across rather than reusing the score it
        was built with: a gain depends on how long the card has been waiting (see
        :func:`src.model.curve.expected_gain`), and the whole point of parking it
        was that time would pass. Only a session with no estimator at all has
        nothing to score with, and that session never re-queues anything.
        """
        now = int(time.time())
        can_score = self._has_estimator()
        while True:
            due_ts = self._waiting.peek_priority()
            if due_ts is None or due_ts > now:
                break
            card = self._waiting.pop()
            if card is None:
                continue
            if can_score:
                card.score = card_gain(card, now)
            self._queue.push(card, -card.score, TIER_REVIEW)

    def _show_current(self) -> None:
        """Pop and render the next-most-urgent card, or transition to DONE when empty."""
        self._promote_due()
        if self._predictor is not None:
            predictor = self._predictor
            refresh_top(
                self._queue,
                int(time.time()),
                lambda cards, now: refresh_post_rep_curves(cards, predictor, now),
            )
        card = self._queue.pop()
        if card is None:
            self._current = None
            # Nothing due and no new words left. Offer the not-due cards rather
            # than ending the session — unless there are none, or the session has
            # no estimator to score them with.
            if len(self._waiting) and self._has_estimator():
                self._set_state(PracticeState.OFFER_EXTRA)
            else:
                self._set_state(PracticeState.DONE)
            return

        self._current = card

        label = card.direction_label(self._ctx.src_lang, self._ctx.tgt_lang)
        self._direction_var.set(f"  {label.upper()}  ")
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
        self._pending_remembered = remembered
        self._set_state(PracticeState.SAVING)
        self._answer_job.start(
            answer_worker,
            card,
            remembered,
            self._predictor,
            self._answer_job.queue,
        )

    def _ignore_current(self) -> None:
        """Flag the shown card's word as ignored and move on.

        Both directions are dropped from the session — the word's other card may
        still be queued, or the answered card re-queued — and :func:`build_queue`
        leaves the word out of every later session until it is un-ignored from
        the edit dialog.
        """
        if self._current is None or self._state is PracticeState.SAVING:
            return
        word_id = self._current.word_id
        try:
            with get_session() as session:
                word = WordRepository(session).get_by_id(word_id)
                if word is not None:
                    word.ignored = True
        except Exception as exc:
            messagebox.showerror("Ignore failed", str(exc), parent=self)
            return
        self._queue.remove_word(word_id)
        self._waiting.remove_word(word_id)
        self._show_current()

    def _advance(self) -> None:
        """RESULT → PROMPT (or OFFER_EXTRA / DONE): move to the next card."""
        self._show_current()

    def _start_extra(self) -> None:
        """OFFER_EXTRA → PROMPT: fold the not-due cards in and carry on.

        Every learned card is now in the one queue, ordered by expected gain
        exactly as the due ones were — the threshold decided only whether a card
        was served, never in what order. Answering keeps working as it does in
        the first phase, so each of them comes up once and is then parked until
        it is genuinely due again.

        That is deliberately not "re-queue it immediately, the gain ordering will
        sort it out". It would not: a card answered a moment ago has a large gain,
        because the curve a *further* review would leave is much longer than the
        one just fitted, so it would come straight back — measured on the French
        deck, often as the very next card. The queue holds the whole vocabulary
        here, so serving each card once is days of practice anyway; when it does
        empty, the offer is simply made again.
        """
        if not self._has_estimator():  # not reachable: the offer needs one
            self._set_state(PracticeState.DONE)
            return
        drain_waiting(self._queue, self._waiting, int(time.time()))
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
        use_threshold: bool,
    ) -> None:
        """Init worker finished — install both queues and show the first card."""
        self._predictor = predictor
        self._queue = queue
        self._waiting = waiting
        self._use_threshold = use_threshold
        self._today_count = today_count
        self._show_current()

    def _on_init_error(self, msg: str) -> None:
        self._show_error(f"Failed to load practice session: {msg}")

    def _on_answered(
        self,
        card: Card,
        practiced_at: int,
        next_ts: int | None,
        curves: tuple[Curve | None, Curve | None, Curve | None],
    ) -> None:
        """Answer worker finished — show result and maybe re-queue the card."""
        if self._pending_remembered:
            self._remembered_count += 1
            self._result_color = Colors.GREEN
            self._result_pill.configure(
                text="  Remembered  ", fg_color=Colors.GREEN, text_color=Colors.ON_GREEN
            )
        else:
            self._forgot_count += 1
            self._result_color = Colors.RED
            self._result_pill.configure(
                text="  Didn't remember  ", fg_color=Colors.RED, text_color="#FFFFFF"
            )
        answer_day = day_start(practiced_at)
        if answer_day == self._today_start:
            self._today_count += 1
        else:
            # The session ran past midnight — start the day tally over.
            self._today_start = answer_day
            self._today_count = 1
        # This word's stored time constants just changed — the word list's cached
        # due times are now stale.
        self._app.invalidate_due_cache()

        if not self._use_threshold:
            # Without a threshold there is no due time — words compete on gain.
            self._next_var.set("")
        elif next_ts is None:
            self._next_var.set("next repetition: —  (no estimator)")
        else:
            delta = next_ts - int(time.time())
            note = (
                HEURISTIC_NOTE
                if isinstance(self._predictor, HeuristicPredictor)
                else ""
            )
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
                priority=card.priority,
                # Each curve carries the ceiling the model emitted beside it, so
                # the refreshed card needs nothing further to be scoreable.
                current=current,
                success=success,
                failure=failure,
            )
            now = int(time.time())
            if success is None or failure is None or not self._has_estimator():
                # Nothing to score the card from — send it to the back rather than
                # the front.
                self._queue.push(refreshed, ERROR_PRIORITY, TIER_EXTRA)
            elif next_ts <= now:
                # Still due right after this attempt — re-queue at its
                # score-sorted position so it returns later in the session. It is
                # re-scored again if it waits, so this score only has to order it
                # against the queue as it stands now.
                refreshed.score = card_gain(refreshed, now)
                self._queue.push(refreshed, -refreshed.score, TIER_REVIEW)
            else:
                # Due in the future — park it in the waiting heap so it can be
                # promoted back if it comes due before the session ends; without
                # the threshold it competes on its gain straight away instead.
                refreshed.score = card_gain(refreshed, now)
                park_not_due(
                    self._queue, self._waiting, refreshed, next_ts, self._use_threshold
                )

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
