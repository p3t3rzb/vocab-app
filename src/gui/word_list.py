"""Word-list screen — browse, search, add, edit, delete words for one database.

Also hosts the entry buttons that navigate to the Practice and Train Model
screens. The treeview shows per-direction due times rendered by
:func:`format_due`.
"""
from __future__ import annotations

from tkinter import messagebox
from typing import TYPE_CHECKING

import customtkinter as ctk

from src.database import WordRepository, init_db, get_session
from src.database.models import Word
from .base_screen import BaseScreen
from .formatting import format_due
from .theme import Colors, Fonts
from .widgets import ColumnSpec, apply_treeview_style, build_header, build_tree

if TYPE_CHECKING:
    from .app import App


class WordListScreen(BaseScreen):
    """Scrollable, searchable table of words with edit / delete / practice / train actions."""

    def __init__(self, master: App) -> None:
        self._all_words: list[Word] = []
        self._filtered: list[Word] = []
        super().__init__(master)
        init_db(self._ctx.db_url, self._ctx.src_lang, self._ctx.tgt_lang)
        self._load_words()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the header, search/action toolbar, and word treeview."""
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header row
        def _add_count(parent: ctk.CTkFrame) -> ctk.CTkLabel:
            self._count_label = ctk.CTkLabel(parent, text="", font=ctk.CTkFont(**Fonts.SMALL))
            self._count_label.grid(row=0, column=2, sticky="e")
            return self._count_label

        header = build_header(
            self,
            title=self._ctx.title,
            on_back=self._app.show_db_select,
            right_widget_factory=_add_count,
        )
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))

        # Search + action buttons
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 6))
        toolbar.grid_columnconfigure(0, weight=1)

        self._search_var = ctk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._apply_filter())
        ctk.CTkEntry(
            toolbar,
            textvariable=self._search_var,
            placeholder_text="Search words…",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(toolbar, text="Add", width=70, command=self._add_word).grid(
            row=0, column=1, padx=2
        )
        self._btn_edit = ctk.CTkButton(
            toolbar, text="Edit", width=70, state="disabled", command=self._edit_word
        )
        self._btn_edit.grid(row=0, column=2, padx=2)
        self._btn_delete = ctk.CTkButton(
            toolbar,
            text="Delete",
            width=70,
            state="disabled",
            fg_color=Colors.DANGER,
            hover_color=Colors.DANGER_HOVER,
            command=self._delete_word,
        )
        self._btn_delete.grid(row=0, column=3, padx=2)

        ctk.CTkButton(
            toolbar, text="Practice", width=90, command=self._open_practice
        ).grid(row=0, column=4, padx=(14, 2))

        ctk.CTkButton(
            toolbar, text="Train Model", width=110, command=self._open_train_dialog
        ).grid(row=0, column=5, padx=2)

        # Treeview
        tree_frame = ctk.CTkFrame(self, fg_color="transparent")
        tree_frame.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        apply_treeview_style()
        src3 = self._ctx.src_lang[:3]
        tgt3 = self._ctx.tgt_lang[:3]
        self._tree, vsb = build_tree(
            tree_frame,
            columns=(
                ColumnSpec("source", self._ctx.src_lang, 240, 100),
                ColumnSpec("target", self._ctx.tgt_lang, 240, 100),
                ColumnSpec("fwd_due", f"{src3}→{tgt3}", 110, 70),
                ColumnSpec("rev_due", f"{tgt3}→{src3}", 110, 70),
            ),
        )
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self._tree.bind("<<TreeviewSelect>>", self._on_select)
        self._tree.bind("<Double-1>", lambda _e: self._open_detail())

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _load_words(self) -> None:
        """Load every word from the database and re-render the treeview."""
        with get_session() as session:
            self._all_words = WordRepository(session).get_all()
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Filter the cached word list by the current search query and refresh the view."""
        query = self._search_var.get().strip().lower()
        if query:
            self._filtered = [
                w for w in self._all_words
                if query in w.source_text.lower() or query in w.target_text.lower()
            ]
        else:
            self._filtered = list(self._all_words)

        self._tree.delete(*self._tree.get_children())
        for word in self._filtered:
            self._tree.insert(
                "", "end",
                values=(
                    word.source_text,
                    word.target_text,
                    format_due(word.next_rep_fwd_at),
                    format_due(word.next_rep_rev_at),
                ),
            )

        total = len(self._all_words)
        shown = len(self._filtered)
        if query:
            self._count_label.configure(text=f"{shown} / {total:,} words")
        else:
            self._count_label.configure(text=f"{total:,} words")

        self._btn_edit.configure(state="disabled")
        self._btn_delete.configure(state="disabled")

    def _selected_word(self) -> Word | None:
        """Return the :class:`Word` for the currently highlighted row, or ``None``."""
        sel = self._tree.selection()
        if not sel:
            return None
        idx = self._tree.index(sel[0])
        if idx >= len(self._filtered):
            return None
        return self._filtered[idx]

    def _on_select(self, _event: object) -> None:
        """Enable/disable the Edit and Delete buttons based on selection state."""
        has = bool(self._tree.selection())
        state = "normal" if has else "disabled"
        self._btn_edit.configure(state=state)
        self._btn_delete.configure(state=state)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _open_detail(self) -> None:
        """Open the per-word detail screen for the selected row."""
        word = self._selected_word()
        if word is None:
            return
        self._app.show_word_detail(word.id)

    def _add_word(self) -> None:
        """Open the modal "Add word" dialog and reload on close."""
        from .dialogs import WordEditDialog
        dialog = WordEditDialog(
            self,
            src_lang=self._ctx.src_lang,
            tgt_lang=self._ctx.tgt_lang,
            word=None,
        )
        self.wait_window(dialog)
        self._load_words()

    def _edit_word(self) -> None:
        """Open the modal "Edit word" dialog for the selected row."""
        word = self._selected_word()
        if word is None:
            return
        from .dialogs import WordEditDialog
        dialog = WordEditDialog(
            self,
            src_lang=self._ctx.src_lang,
            tgt_lang=self._ctx.tgt_lang,
            word=word,
        )
        self.wait_window(dialog)
        self._load_words()

    def _delete_word(self) -> None:
        """Prompt for confirmation, then delete the selected word and its history."""
        word = self._selected_word()
        if word is None:
            return
        confirmed = messagebox.askyesno(
            "Confirm delete",
            f'Delete "{word.source_text}" / "{word.target_text}"?\n\n'
            "All repetition history for this word will also be removed.",
            icon="warning",
        )
        if not confirmed:
            return
        with get_session() as session:
            repo = WordRepository(session)
            w = repo.get_by_id(word.id)
            if w is not None:
                repo.delete(w)
        self._load_words()

    def _open_train_dialog(self) -> None:
        """Navigate to the Train Model screen."""
        self._app.show_train_screen(self._ctx)

    def _open_practice(self) -> None:
        """Navigate to the Practice screen."""
        self._app.show_practice_screen(self._ctx)
