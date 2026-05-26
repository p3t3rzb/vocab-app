"""Database-selection screen — the app's home view.

Lists every ``*.db`` file in ``storage/`` along with its language pair, word
count, and most recent training timestamp. Also hosts the "New database"
dialog and the navigation entry point to the global settings screen.
"""
from __future__ import annotations

from pathlib import Path
from tkinter import messagebox
from typing import TYPE_CHECKING

import customtkinter as ctk

from src.database import count_words, init_db, read_language_pair

from .base_screen import BaseScreen
from .db_context import DbContext
from .db_select_util import DbEntry, last_trained_label, safe_db_basename, slugify
from .dialogs import BaseDialog
from .theme import Fonts, Paths, WindowSizes
from .widgets import ColumnSpec, apply_treeview_style, build_tree

if TYPE_CHECKING:
    from .app import App


class DatabaseSelectScreen(BaseScreen):
    """Lists every database in ``storage/`` and lets the user open or create one."""

    def __init__(self, master: App) -> None:
        self._db_entries: list[DbEntry] = []
        super().__init__(master)
        self._load_databases()

    def build(self) -> None:
        """Construct the title, treeview, scroll bar, and action buttons."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self,
            text="Vocabulary Databases",
            font=ctk.CTkFont(**Fonts.TITLE),
        )
        title.grid(row=0, column=0, pady=(24, 12), padx=24, sticky="w")

        tree_frame = ctk.CTkFrame(self, fg_color="transparent")
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 12))
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        apply_treeview_style()
        self._tree, vsb = build_tree(
            tree_frame,
            columns=(
                ColumnSpec("file", "File", 180, 120),
                ColumnSpec("source", "Source language", 150, 100),
                ColumnSpec("target", "Target language", 150, 100),
                ColumnSpec("words", "Words", 70, 60, anchor="e"),
                ColumnSpec("trained", "Last Trained", 160, 120),
            ),
        )
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self._tree.bind("<Double-1>", lambda _e: self._open_selected())

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=24, pady=(0, 20), sticky="e")

        ctk.CTkButton(btn_frame, text="Settings", width=100, command=self._app.show_settings).pack(side="left", padx=4)
        ctk.CTkButton(btn_frame, text="New", width=100, command=self._new_database).pack(side="left", padx=4)
        ctk.CTkButton(btn_frame, text="Open", width=100, command=self._open_selected).pack(side="left", padx=4)

    def _load_databases(self) -> None:
        """Re-scan ``storage/`` and populate the treeview from scratch."""
        self._tree.delete(*self._tree.get_children())
        self._db_entries.clear()

        if not Paths.STORAGE_DIR.exists():
            return

        for db_path in sorted(Paths.STORAGE_DIR.glob("*.db")):
            pair = read_language_pair(db_path)
            if pair is None:
                continue
            src, tgt = pair
            count = count_words(db_path)
            trained = last_trained_label(src, tgt)
            self._db_entries.append(DbEntry(db_path=db_path, src_lang=src, tgt_lang=tgt))
            self._tree.insert(
                "",
                "end",
                values=(db_path.name, src, tgt, f"{count:,}", trained),
            )

    def _new_database(self) -> None:
        """Open the "New database" dialog and, on success, navigate to it."""
        dialog = NewDatabaseDialog(self)
        self.wait_window(dialog)
        if dialog.created_path is not None:
            self._load_databases()
            self._app.show_word_list(
                DbContext(dialog.created_path, dialog.src_lang, dialog.tgt_lang)
            )

    def _open_selected(self) -> None:
        """Navigate to the word list of the currently selected database."""
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        entry = self._db_entries[idx]
        self._app.show_word_list(DbContext(entry.db_path, entry.src_lang, entry.tgt_lang))


class NewDatabaseDialog(BaseDialog):
    """Modal dialog: collect language names and create an empty database.

    On Create, the dialog calls :func:`init_db` to materialise the new SQLite
    file and stores the resulting path in :attr:`created_path` so the parent
    screen can pick it up.
    """

    def __init__(self, master: ctk.CTkBaseClass) -> None:
        self.created_path: Path | None = None
        self.src_lang: str = ""
        self.tgt_lang: str = ""
        super().__init__(master, title="New Database", size=WindowSizes.NEW_DB_DIALOG)

    def _build(self) -> None:
        """Lay out the language entries, filename preview, and action buttons."""
        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self, text="Source language:", anchor="e", width=130).grid(
            row=0, column=0, padx=(20, 8), pady=(28, 6), sticky="e"
        )
        self._src_entry = ctk.CTkEntry(self, width=220)
        self._src_entry.grid(row=0, column=1, padx=(0, 20), pady=(28, 6), sticky="ew")

        ctk.CTkLabel(self, text="Target language:", anchor="e", width=130).grid(
            row=1, column=0, padx=(20, 8), pady=6, sticky="e"
        )
        self._tgt_entry = ctk.CTkEntry(self, width=220)
        self._tgt_entry.grid(row=1, column=1, padx=(0, 20), pady=6, sticky="ew")

        ctk.CTkLabel(self, text="File:", anchor="e", width=130).grid(
            row=2, column=0, padx=(20, 8), pady=6, sticky="e"
        )
        self._filename_label = ctk.CTkLabel(self, text="", anchor="w", text_color="gray")
        self._filename_label.grid(row=2, column=1, padx=(0, 20), pady=6, sticky="ew")

        self._src_entry.bind("<KeyRelease>", lambda _e: self._update_filename())
        self._tgt_entry.bind("<KeyRelease>", lambda _e: self._update_filename())

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(20, 16))

        ctk.CTkButton(btn_frame, text="Cancel", width=100, command=self.destroy).pack(side="left", padx=8)
        ctk.CTkButton(btn_frame, text="Create", width=100, command=self._create).pack(side="left", padx=8)

        self._src_entry.focus()
        self.bind_default_keys(on_save=self._create)

    def _update_filename(self) -> None:
        """Refresh the read-only filename label whenever an entry changes."""
        src = self._src_entry.get().strip()
        tgt = self._tgt_entry.get().strip()
        base = safe_db_basename(src, tgt)
        if base:
            self._filename_label.configure(text=f"{base}.db")
        elif src:
            src_c = slugify(src)
            self._filename_label.configure(text=f"{src_c}_....db" if src_c else "")
        else:
            self._filename_label.configure(text="")

    def _create(self) -> None:
        """Validate inputs, create the database file, and close the dialog."""
        src_lang = self._src_entry.get().strip()
        tgt_lang = self._tgt_entry.get().strip()

        if not src_lang or not tgt_lang:
            messagebox.showwarning("Missing fields", "Both language fields must be filled in.", parent=self)
            return

        base = safe_db_basename(src_lang, tgt_lang)
        if not base:
            messagebox.showwarning(
                "Invalid name",
                "Language names must contain at least one letter or digit.",
                parent=self,
            )
            return
        filename = f"{base}.db"
        db_path = Paths.STORAGE_DIR / filename

        if db_path.exists():
            messagebox.showwarning(
                "Already exists",
                f"A database named '{filename}' already exists.",
                parent=self,
            )
            return

        try:
            Paths.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            init_db(f"sqlite:///{db_path}", src_lang, tgt_lang)
        except Exception as exc:
            messagebox.showerror("Creation failed", str(exc), parent=self)
            return

        self.created_path = db_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.destroy()
