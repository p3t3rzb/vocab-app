"""Word-detail screen — full repetition history for a single word.

Reached from the word list by double-clicking a row. Shows every practice
event (in either direction) ordered by timestamp.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import customtkinter as ctk

from src.database import Direction, RepetitionRepository, WordRepository, get_session
from src.database.models import Repetition
from .base_screen import BaseScreen
from .formatting import format_timestamp
from .theme import Fonts
from .widgets import ColumnSpec, TreeSorter, apply_treeview_style, build_header, build_tree

if TYPE_CHECKING:
    from .app import App


def _fmt_direction(rep: Repetition, src_lang: str, tgt_lang: str) -> str:
    """Format a repetition's direction as ``"source → target"`` or vice versa."""
    if rep.direction == int(Direction.FORWARD):
        return f"{src_lang} → {tgt_lang}"
    return f"{tgt_lang} → {src_lang}"


class WordDetailScreen(BaseScreen):
    """Lists every repetition event for one word, in either direction."""

    def __init__(self, master: App, word_id: int) -> None:
        self._word_id = word_id
        self._all_reps: list[Repetition] = []
        super().__init__(master)
        self._load()

    def build(self) -> None:
        """Construct the header, summary label, and history treeview."""
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header — Back button + word label + Edit button
        def _add_edit(parent: ctk.CTkFrame) -> ctk.CTkButton:
            self._btn_edit = ctk.CTkButton(
                parent, text="Edit", width=80, command=self._edit_word
            )
            self._btn_edit.grid(row=0, column=2, sticky="e")
            return self._btn_edit

        # The word label is the header's title slot — rewritten in _load().
        self._header = build_header(
            self,
            title="",
            on_back=self._app.back_to_word_list,
            right_widget_factory=_add_edit,
        )
        self._header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))

        # Repetition count label
        self._rep_label = ctk.CTkLabel(
            self,
            text="Repetition history",
            font=ctk.CTkFont(**Fonts.BODY),
            anchor="w",
        )
        self._rep_label.grid(row=1, column=0, sticky="w", padx=20, pady=(8, 4))

        # Treeview
        tree_frame = ctk.CTkFrame(self, fg_color="transparent")
        tree_frame.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        apply_treeview_style()
        columns = (
            ColumnSpec("direction", "Direction", 240, 160, sort_key=lambda r: r.direction),
            ColumnSpec("date", "Date & Time", 200, 160, sort_key=lambda r: r.practiced_at),
            ColumnSpec(
                "remembered", "Remembered", 110, 80, anchor="center",
                sort_key=lambda r: r.remembered,
            ),
        )
        self._tree, vsb = build_tree(tree_frame, columns=columns, selectmode="none")
        self._sorter = TreeSorter(self._tree, columns, on_change=self._render)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

    def _load(self) -> None:
        """Fetch the word and its repetition history, then populate the treeview."""
        with get_session() as session:
            word = WordRepository(session).get_by_id(self._word_id)
            if word is None:
                self._header.set_title("(word not found)")
                return

            self._header.set_title(f"{word.source_text}  ↔  {word.target_text}")

            reps_fwd = RepetitionRepository(session).get_for_word(
                self._word_id, Direction.FORWARD
            )
            reps_rev = RepetitionRepository(session).get_for_word(
                self._word_id, Direction.REVERSE
            )

        self._all_reps = sorted(reps_fwd + reps_rev, key=lambda r: r.practiced_at)
        self._render()

    def _render(self) -> None:
        """(Re)populate the treeview from the cached reps in the current sort order."""
        reps = self._sorter.order(self._all_reps)

        self._tree.delete(*self._tree.get_children())
        for rep in reps:
            remembered = "✓" if rep.remembered else "✗"
            self._tree.insert(
                "",
                "end",
                values=(
                    _fmt_direction(rep, self._ctx.src_lang, self._ctx.tgt_lang),
                    format_timestamp(rep.practiced_at),
                    remembered,
                ),
            )

        count = len(self._all_reps)
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

        from .dialogs import WordEditDialog
        dialog = WordEditDialog(
            self,
            src_lang=self._ctx.src_lang,
            tgt_lang=self._ctx.tgt_lang,
            word=word,
        )
        self.wait_window(dialog)
        self._load()
