"""Full-screen spaced-repetition practice loop.

Arrow-key driven: **↓** reveal answer, **→** remembered, **←** forgot, any
arrow advances after a result is shown. The queue is built once on entry
(review-due pairs first, then never-practiced pairs) and re-extended at
runtime when the predictor marks a card still due immediately after a fail.

Heavy work (model loading, queue construction, per-answer prediction) runs
on background threads which push events through a queue; the main thread
drains them every ``_POLL_MS``.
"""
from __future__ import annotations

import queue
import random
import threading
import time
from pathlib import Path
from tkinter import messagebox
from typing import TYPE_CHECKING

import customtkinter as ctk

from src.database import (
    Direction,
    Repetition,
    RepetitionRepository,
    WordRepository,
    get_session,
    init_db,
)

if TYPE_CHECKING:
    from .app import App


_POLL_MS = 50

_STATE_LOADING = "loading"
_STATE_PROMPT = "prompt"
_STATE_ANSWER = "answer"
_STATE_SAVING = "saving"
_STATE_RESULT = "result"
_STATE_DONE = "done"


def _fmt_past(seconds: int) -> str:
    """Format an elapsed duration as ``"just now"`` / ``"5m ago"`` / ``"2d 3h ago"``."""
    if seconds < 60:
        return "just now"
    m = seconds // 60
    if m < 60:
        return f"{m}m ago"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h ago" if m == 0 else f"{h}h {m}m ago"
    d, h = divmod(h, 24)
    return f"{d}d ago" if h == 0 else f"{d}d {h}h ago"


def _fmt_future(seconds: int) -> str:
    """Format a forward-looking duration as ``"due now"`` / ``"in 12m"`` / ``"in 5d 2h"``."""
    if seconds <= 0:
        return "due now"
    m = seconds // 60
    if m < 60:
        return f"in {m}m"
    h, m = divmod(m, 60)
    if h < 24:
        return f"in {h}h" if m == 0 else f"in {h}h {m}m"
    d, h = divmod(h, 24)
    return f"in {d}d" if h == 0 else f"in {d}d {h}h"


class PracticeScreen(ctk.CTkFrame):
    """Arrow-key driven spaced-repetition session.

    The screen walks through a queue of ``(word, direction)`` cards. Each
    card transitions through the prompt → answer → saving → result states;
    after the result, any arrow key advances to the next card.

    If the predictor marks a card still due immediately after a failed
    attempt, the card is appended to the queue so it returns later in the
    same session.
    """

    def __init__(
        self,
        master: App,
        db_path: Path,
        src_lang: str,
        tgt_lang: str,
    ) -> None:
        super().__init__(master, corner_radius=0)
        self._app = master
        self._db_path = db_path
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang

        init_db(f"sqlite:///{db_path}", src_lang, tgt_lang)

        self._queue: queue.Queue = queue.Queue()
        self._predictor = None  # set after init worker completes
        self._items: list[dict] = []
        self._idx = 0
        self._answered_count = 0
        self._state: str = _STATE_LOADING
        self._current: dict | None = None
        self._keys_bound = False
        self._destroyed = False

        self._build_ui()
        self._bind_keys()
        self._start_init_worker()
        self.after(_POLL_MS, self._poll)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the header, prompt/answer area, and the hint bar."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            header, text="← Back", width=80, command=self._go_back
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header,
            text=f"Practice  —  {self._src_lang} ↔ {self._tgt_lang}",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=1, padx=12, sticky="w")

        self._stats_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            header, textvariable=self._stats_var, font=ctk.CTkFont(size=13)
        ).grid(row=0, column=2, sticky="e")

        # Main content area
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=16, pady=(20, 8))
        content.grid_rowconfigure(0, weight=1)
        content.grid_rowconfigure(7, weight=1)
        content.grid_columnconfigure(0, weight=1)

        self._direction_var = ctk.StringVar(value="")
        self._direction_label = ctk.CTkLabel(
            content,
            textvariable=self._direction_var,
            font=ctk.CTkFont(size=14),
            text_color=("gray40", "gray70"),
        )
        self._direction_label.grid(row=1, column=0, pady=(0, 16))

        self._prompt_var = ctk.StringVar(value="Loading…")
        self._prompt_label = ctk.CTkLabel(
            content,
            textvariable=self._prompt_var,
            font=ctk.CTkFont(size=36, weight="bold"),
            wraplength=700,
        )
        self._prompt_label.grid(row=2, column=0, pady=(0, 12))

        self._sep_label = ctk.CTkLabel(
            content,
            text="─ ─ ─ ─ ─",
            text_color=("gray60", "gray50"),
        )
        self._sep_label.grid(row=3, column=0, pady=(8, 8))

        self._answer_var = ctk.StringVar(value="")
        self._answer_label = ctk.CTkLabel(
            content,
            textvariable=self._answer_var,
            font=ctk.CTkFont(size=32),
            wraplength=700,
        )
        self._answer_label.grid(row=4, column=0, pady=(0, 24))

        self._last_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            content,
            textvariable=self._last_var,
            font=ctk.CTkFont(size=13),
            text_color=("gray40", "gray70"),
        ).grid(row=5, column=0, pady=(0, 4))

        self._next_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            content,
            textvariable=self._next_var,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=6, column=0, pady=(0, 4))

        # Hint bar
        self._hint_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            self,
            textvariable=self._hint_var,
            font=ctk.CTkFont(size=13),
            text_color=("gray30", "gray70"),
        ).grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))

        self._set_state(_STATE_LOADING)

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _bind_keys(self) -> None:
        """Bind arrow-key handlers on the root window (idempotent)."""
        if self._keys_bound:
            return
        root = self._app
        root.bind("<Down>", lambda _e: self._on_key("Down"))
        root.bind("<Up>", lambda _e: self._on_key("Up"))
        root.bind("<Left>", lambda _e: self._on_key("Left"))
        root.bind("<Right>", lambda _e: self._on_key("Right"))
        self.bind("<Destroy>", self._on_destroy)
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

    def _on_destroy(self, event: object) -> None:
        """Mark the screen destroyed and detach key bindings when Tk tears it down."""
        if getattr(event, "widget", None) is self:
            self._destroyed = True
            self._unbind_keys()

    def _on_key(self, key: str) -> None:
        """Route an arrow keypress to the appropriate handler for the current state."""
        if self._state == _STATE_PROMPT and key == "Down":
            self._reveal_answer()
        elif self._state == _STATE_ANSWER:
            if key == "Right":
                self._submit_answer(remembered=True)
            elif key == "Left":
                self._submit_answer(remembered=False)
        elif self._state == _STATE_RESULT:
            if key in ("Left", "Right", "Up", "Down"):
                self._advance()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        """Update the visible widgets and hint bar to reflect the new state."""
        self._state = state

        if state == _STATE_LOADING:
            self._direction_var.set("")
            self._prompt_var.set("Loading…")
            self._sep_label.grid_remove()
            self._answer_var.set("")
            self._last_var.set("")
            self._next_var.set("")
            self._hint_var.set("")

        elif state == _STATE_PROMPT:
            self._sep_label.grid_remove()
            self._answer_var.set("")
            self._next_var.set("")
            self._hint_var.set("↓  show translation")

        elif state == _STATE_ANSWER:
            self._sep_label.grid()
            self._next_var.set("")
            self._hint_var.set("←  didn't remember     →  remembered")

        elif state == _STATE_SAVING:
            self._hint_var.set("Saving…")

        elif state == _STATE_RESULT:
            self._sep_label.grid()
            self._hint_var.set("press any arrow for the next word")

        elif state == _STATE_DONE:
            self._direction_var.set("")
            self._prompt_var.set("No more words to repeat now.")
            self._sep_label.grid_remove()
            self._answer_var.set("")
            self._last_var.set("")
            self._next_var.set("")
            self._hint_var.set("press ← Back to return to the word list")

        self._update_stats()

    def _update_stats(self) -> None:
        """Refresh the "Answered N • Remaining M" counter in the header."""
        if self._state == _STATE_LOADING:
            self._stats_var.set("")
            return
        remaining = max(0, len(self._items) - self._idx)
        if self._state == _STATE_DONE:
            self._stats_var.set(f"Answered {self._answered_count}")
        else:
            self._stats_var.set(
                f"Answered {self._answered_count}  •  Remaining {remaining}"
            )

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def _show_current(self) -> None:
        """Render the card at ``_idx``, or transition to the done state when exhausted."""
        if self._idx >= len(self._items):
            self._set_state(_STATE_DONE)
            return

        item = self._items[self._idx]
        self._current = item

        if item["direction"] is Direction.FORWARD:
            self._direction_var.set(f"{self._src_lang} → {self._tgt_lang}")
            self._prompt_var.set(item["source_text"])
        else:
            self._direction_var.set(f"{self._tgt_lang} → {self._src_lang}")
            self._prompt_var.set(item["target_text"])

        last = item.get("last_practiced")
        if last is None:
            self._last_var.set("never practiced before")
        else:
            self._last_var.set(
                f"last revised {_fmt_past(int(time.time()) - int(last))}"
            )

        self._set_state(_STATE_PROMPT)

    def _reveal_answer(self) -> None:
        """Reveal the translation for the current card (PROMPT → ANSWER)."""
        if self._current is not None:
            if self._current["direction"] is Direction.FORWARD:
                self._answer_var.set(self._current["target_text"])
            else:
                self._answer_var.set(self._current["source_text"])
        self._set_state(_STATE_ANSWER)

    def _submit_answer(self, remembered: bool) -> None:
        """Record the result on a background thread (ANSWER → SAVING)."""
        if self._current is None:
            return
        item = self._current
        self._set_state(_STATE_SAVING)
        thread = threading.Thread(
            target=self._answer_worker,
            args=(item, remembered),
            daemon=True,
        )
        thread.start()

    def _on_answered(
        self, item: dict, practiced_at: int, next_ts: int | None
    ) -> None:
        """Process the worker's reply: update stats, show the result, and maybe re-queue."""
        self._answered_count += 1

        # Update local item state so a re-queue shows the most recent practice
        item = dict(item)
        item["last_practiced"] = practiced_at

        # Show result
        if next_ts is None:
            self._next_var.set("next repetition: —  (no trained model)")
        else:
            delta = next_ts - int(time.time())
            self._next_var.set(f"next repetition {_fmt_future(delta)}")

        # Re-queue if predictor said it's still due now
        if next_ts is not None and next_ts <= int(time.time()):
            self._items.append(item)

        self._idx += 1
        self._set_state(_STATE_RESULT)

    def _advance(self) -> None:
        """Move on to the next card (RESULT → PROMPT or DONE)."""
        self._show_current()

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def _start_init_worker(self) -> None:
        """Spawn the background thread that loads the predictor and builds the queue."""
        threading.Thread(target=self._init_worker, daemon=True).start()

    def _init_worker(self) -> None:
        """Background: load the trained model (if any) and build the review/new queues.

        Builds two lists in the worker, shuffles each independently, and
        sends ``review + new`` back so review cards are always shown before
        brand-new words.
        """
        try:
            model_path = (
                Path(__file__).parent.parent.parent
                / "storage"
                / "models"
                / f"{self._src_lang.lower()}_{self._tgt_lang.lower()}.pt"
            )
            predictor = None
            if model_path.exists():
                from src.model.train import load_model
                from src.model.predictor import Predictor
                from src.settings import load_settings
                model = load_model(str(model_path))
                predictor = Predictor(model, load_settings().to_predict_config())

            now = int(time.time())
            review_items: list[dict] = []
            new_items: list[dict] = []
            with get_session() as session:
                words = WordRepository(session).get_all()
                reps_repo = RepetitionRepository(session)
                for w in words:
                    for direction, due_ts in (
                        (Direction.FORWARD, w.next_rep_fwd_at),
                        (Direction.REVERSE, w.next_rep_rev_at),
                    ):
                        latest = reps_repo.get_latest_for_word(w.id, direction)
                        item = {
                            "word_id": w.id,
                            "direction": direction,
                            "source_text": w.source_text,
                            "target_text": w.target_text,
                            "last_practiced": latest.practiced_at if latest else None,
                        }
                        if latest is None:
                            new_items.append(item)
                        elif due_ts is None or due_ts <= now:
                            review_items.append(item)

            random.shuffle(review_items)
            random.shuffle(new_items)
            full_queue = review_items + new_items
            self._queue.put(("ready", predictor, full_queue))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    def _answer_worker(self, item: dict, remembered: bool) -> None:
        """Background: record the repetition and recompute this direction's next-due time.

        Falls back to ``next_ts = None`` (and skips the DB update of
        ``next_rep_*_at``) when no model has been trained yet.
        """
        try:
            practiced_at = int(time.time())
            direction: Direction = item["direction"]
            word_id: int = item["word_id"]
            next_ts: int | None = None

            with get_session() as session:
                reps_repo = RepetitionRepository(session)
                reps_repo.add(
                    Repetition(
                        word_id=word_id,
                        direction=int(direction),
                        practiced_at=practiced_at,
                        remembered=remembered,
                    )
                )

                if self._predictor is not None:
                    session.flush()
                    all_reps = reps_repo.get_for_word(word_id, direction)
                    try:
                        delta = self._predictor.next_repetition_delta(all_reps)
                        next_ts = practiced_at + int(delta)
                    except Exception:
                        next_ts = 0
                    word = WordRepository(session).get_by_id(word_id)
                    if word is not None:
                        if direction is Direction.FORWARD:
                            word.next_rep_fwd_at = next_ts
                        else:
                            word.next_rep_rev_at = next_ts

            self._queue.put(("answered", item, practiced_at, next_ts))
        except Exception as exc:
            self._queue.put(("answer_error", str(exc)))

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        """Drain worker events from the queue, dispatching to the right handler."""
        if self._destroyed:
            return
        try:
            while True:
                item = self._queue.get_nowait()
                tag = item[0]

                if tag == "ready":
                    _, predictor, full_queue = item
                    self._predictor = predictor
                    self._items = full_queue
                    self._idx = 0
                    self._show_current()

                elif tag == "answered":
                    _, q_item, practiced_at, next_ts = item
                    self._on_answered(q_item, practiced_at, next_ts)

                elif tag == "error":
                    _, msg = item
                    self._show_error(f"Failed to load practice session: {msg}")
                    return

                elif tag == "answer_error":
                    _, msg = item
                    self._show_error(f"Failed to save answer: {msg}")
                    return

        except queue.Empty:
            pass

        if not self._destroyed:
            self.after(_POLL_MS, self._poll)

    def _show_error(self, msg: str) -> None:
        """Pop up an error dialog and bounce back to the word list."""
        messagebox.showerror("Practice error", msg, parent=self)
        self._app.back_to_word_list()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_back(self) -> None:
        """Detach key bindings and return to the word list."""
        self._unbind_keys()
        self._app.back_to_word_list()
