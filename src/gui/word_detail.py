"""Word-detail screen — full repetition history for a single word.

Reached from the word list by double-clicking a row. Shows every practice
event (in either direction) ordered by timestamp.
"""
from __future__ import annotations

from datetime import datetime
from tkinter import ttk
from typing import TYPE_CHECKING

import customtkinter as ctk

from src.database import Direction, RepetitionRepository, WordRepository, get_session
from src.database.models import Repetition
from .db_select import _apply_treeview_style

if TYPE_CHECKING:
    from .app import App


def _fmt_ts(ts: int) -> str:
    """Format a Unix timestamp as ``"YYYY-MM-DD  HH:MM:SS"``."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d  %H:%M:%S")


def _fmt_direction(rep: Repetition, src_lang: str, tgt_lang: str) -> str:
    """Format a repetition's direction as ``"source → target"`` or vice versa."""
    if rep.direction == int(Direction.FORWARD):
        return f"{src_lang} → {tgt_lang}"
    return f"{tgt_lang} → {src_lang}"


class WordDetailScreen(ctk.CTkFrame):
    """Lists every repetition event for one word, in either direction."""

    def __init__(
        self,
        master: App,
        word_id: int,
        src_lang: str,
        tgt_lang: str,
    ) -> None:
        super().__init__(master, corner_radius=0)
        self._app = master
        self._word_id = word_id
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang

        self._build_ui()
        self._load()

    def _build_ui(self) -> None:
        """Construct the header, summary label, and history treeview."""
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            header,
            text="← Back",
            width=80,
            command=self._app.back_to_word_list,
        ).grid(row=0, column=0, sticky="w")

        self._word_label = ctk.CTkLabel(
            header,
            text="",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self._word_label.grid(row=0, column=1, padx=12, sticky="w")

        self._btn_edit = ctk.CTkButton(
            header, text="Edit", width=80, command=self._edit_word
        )
        self._btn_edit.grid(row=0, column=2, sticky="e")

        # Repetition count label
        self._rep_label = ctk.CTkLabel(
            self,
            text="Repetition history",
            font=ctk.CTkFont(size=14),
            anchor="w",
        )
        self._rep_label.grid(row=1, column=0, sticky="w", padx=20, pady=(8, 4))

        # Treeview
        tree_frame = ctk.CTkFrame(self, fg_color="transparent")
        tree_frame.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        style = ttk.Style()
        _apply_treeview_style(style)

        self._tree = ttk.Treeview(
            tree_frame,
            columns=("direction", "date", "remembered"),
            show="headings",
            selectmode="none",
            style="App.Treeview",
        )
        self._tree.heading("direction", text="Direction")
        self._tree.heading("date", text="Date & Time")
        self._tree.heading("remembered", text="Remembered")
        self._tree.column("direction", width=240, minwidth=160)
        self._tree.column("date", width=200, minwidth=160)
        self._tree.column("remembered", width=110, minwidth=80, anchor="center")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

    def _load(self) -> None:
        """Fetch the word and its repetition history, then populate the treeview."""
        with get_session() as session:
            word = WordRepository(session).get_by_id(self._word_id)
            if word is None:
                self._word_label.configure(text="(word not found)")
                return

            self._word_label.configure(
                text=f"{word.source_text}  ↔  {word.target_text}"
            )

            reps_fwd = RepetitionRepository(session).get_for_word(
                self._word_id, Direction.FORWARD
            )
            reps_rev = RepetitionRepository(session).get_for_word(
                self._word_id, Direction.REVERSE
            )

        all_reps = sorted(reps_fwd + reps_rev, key=lambda r: r.practiced_at)

        self._tree.delete(*self._tree.get_children())
        for rep in all_reps:
            remembered = "✓" if rep.remembered else "✗"
            self._tree.insert(
                "",
                "end",
                values=(
                    _fmt_direction(rep, self._src_lang, self._tgt_lang),
                    _fmt_ts(rep.practiced_at),
                    remembered,
                ),
            )

        count = len(all_reps)
        self._rep_label.configure(
            text=f"Repetition history  ({count:,} event{'s' if count != 1 else ''})"
        )

    def _edit_word(self) -> None:
        """Open the edit dialog for this word; reload the history on close."""
        with get_session() as session:
            word = WordRepository(session).get_by_id(self._word_id)
            if word is None:
                return
            # Detach from session so we can pass a plain object to the dialog
            session.expunge(word)

        from .word_edit import WordEditDialog
        dialog = WordEditDialog(
            self,
            src_lang=self._src_lang,
            tgt_lang=self._tgt_lang,
            word=word,
        )
        self.wait_window(dialog)
        self._load()
