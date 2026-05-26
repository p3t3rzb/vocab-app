"""Database-selection screen — the app's home view.

Lists every ``*.db`` file in ``storage/`` along with its language pair, word
count, and most recent training timestamp. Also hosts the "New database"
dialog and the navigation entry point to the global settings screen.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import customtkinter as ctk
from tkinter import messagebox, ttk, font as tkfont

from src.database import init_db

if TYPE_CHECKING:
    from .app import App

STORAGE_DIR = Path(__file__).parent.parent.parent / "storage"


def _read_language_pair(db_path: Path) -> tuple[str, str] | None:
    """Return ``(source_language, target_language)`` for ``db_path``, or ``None`` on failure.

    Uses raw sqlite3 (not SQLAlchemy) so we can peek at every database in
    ``storage/`` without re-initialising the global engine each time.
    """
    try:
        con = sqlite3.connect(str(db_path))
        row = con.execute(
            "SELECT source_language, target_language FROM language_pair LIMIT 1"
        ).fetchone()
        con.close()
        return (row[0], row[1]) if row else None
    except Exception:
        return None


class DatabaseSelectScreen(ctk.CTkFrame):
    """Lists every database in ``storage/`` and lets the user open or create one."""

    def __init__(self, master: App) -> None:
        super().__init__(master, corner_radius=0)
        self._app = master
        self._db_entries: list[tuple[Path, str, str]] = []

        self._build_ui()
        self._load_databases()

    def _build_ui(self) -> None:
        """Construct the title, treeview, scroll bar, and action buttons."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self,
            text="Vocabulary Databases",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.grid(row=0, column=0, pady=(24, 12), padx=24, sticky="w")

        tree_frame = ctk.CTkFrame(self, fg_color="transparent")
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 12))
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        style = ttk.Style()
        _apply_treeview_style(style)

        self._tree = ttk.Treeview(
            tree_frame,
            columns=("file", "source", "target", "words", "trained"),
            show="headings",
            selectmode="browse",
            style="App.Treeview",
        )
        self._tree.heading("file", text="File")
        self._tree.heading("source", text="Source language")
        self._tree.heading("target", text="Target language")
        self._tree.heading("words", text="Words")
        self._tree.heading("trained", text="Last Trained")
        self._tree.column("file", width=180, minwidth=120)
        self._tree.column("source", width=150, minwidth=100)
        self._tree.column("target", width=150, minwidth=100)
        self._tree.column("words", width=70, minwidth=60, anchor="e")
        self._tree.column("trained", width=160, minwidth=120)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)

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

        if not STORAGE_DIR.exists():
            return

        for db_path in sorted(STORAGE_DIR.glob("*.db")):
            pair = _read_language_pair(db_path)
            if pair is None:
                continue
            src, tgt = pair
            count = _word_count(db_path)
            trained = _last_trained(src, tgt)
            self._db_entries.append((db_path, src, tgt))
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
            self._app.show_word_list(dialog.created_path, dialog.src_lang, dialog.tgt_lang)

    def _open_selected(self) -> None:
        """Navigate to the word list of the currently selected database."""
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        db_path, src, tgt = self._db_entries[idx]
        self._app.show_word_list(db_path, src, tgt)


class NewDatabaseDialog(ctk.CTkToplevel):
    """Modal dialog: collect language names and create an empty database.

    On Create, the dialog calls :func:`init_db` to materialise the new SQLite
    file and stores the resulting path in :attr:`created_path` so the parent
    screen can pick it up.
    """

    def __init__(self, master: ctk.CTkBaseClass) -> None:
        super().__init__(master)
        self.created_path: Path | None = None
        self.src_lang: str = ""
        self.tgt_lang: str = ""

        self.title("New Database")
        self.geometry("420x260")
        self.resizable(False, False)

        self._build_ui()

        self.transient(master)  # type: ignore[arg-type]
        self.grab_set()
        self.lift()
        self.focus_force()
        self.after(50, self._safe_grab)

    def _safe_grab(self) -> None:
        """Re-attempt the modal grab once the window is fully realised.

        On some window managers ``grab_set`` raises if called too early; we
        ignore the failure since the up-front ``grab_set`` usually succeeds.
        """
        try:
            self.grab_set()
        except Exception:
            pass

    def _build_ui(self) -> None:
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
        self.bind("<Return>", lambda _e: self._create())
        self.bind("<Escape>", lambda _e: self.destroy())

    def _update_filename(self) -> None:
        """Refresh the read-only filename label whenever an entry changes."""
        src = self._src_entry.get().strip()
        tgt = self._tgt_entry.get().strip()
        if src and tgt:
            filename = f"{src.lower()}_{tgt.lower()}".replace(" ", "_") + ".db"
            self._filename_label.configure(text=filename)
        elif src:
            self._filename_label.configure(text=f"{src.lower()}_...db".replace(" ", "_"))
        else:
            self._filename_label.configure(text="")

    def _create(self) -> None:
        """Validate inputs, create the database file, and close the dialog."""
        src_lang = self._src_entry.get().strip()
        tgt_lang = self._tgt_entry.get().strip()

        if not src_lang or not tgt_lang:
            messagebox.showwarning("Missing fields", "Both language fields must be filled in.", parent=self)
            return

        filename = f"{src_lang.lower()}_{tgt_lang.lower()}".replace(" ", "_") + ".db"
        db_path = STORAGE_DIR / filename

        if db_path.exists():
            messagebox.showwarning(
                "Already exists",
                f"A database named '{filename}' already exists.",
                parent=self,
            )
            return

        try:
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            init_db(f"sqlite:///{db_path}", src_lang, tgt_lang)
        except Exception as exc:
            messagebox.showerror("Creation failed", str(exc), parent=self)
            return

        self.created_path = db_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.destroy()


def _last_trained(src: str, tgt: str) -> str:
    """Return the checkpoint mtime as a display string, or ``"—"`` if not trained."""
    from datetime import datetime
    ckpt = STORAGE_DIR / "models" / f"{src.lower()}_{tgt.lower()}.pt"
    if not ckpt.exists():
        return "—"
    return datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%b %d, %Y  %H:%M")


def _word_count(db_path: Path) -> int:
    """Return ``COUNT(*)`` from the ``words`` table, or ``0`` on any error."""
    try:
        con = sqlite3.connect(str(db_path))
        row = con.execute("SELECT COUNT(*) FROM words").fetchone()
        con.close()
        return row[0] if row else 0
    except Exception:
        return 0


def _apply_treeview_style(style: ttk.Style) -> None:
    """Configure the shared ``App.Treeview`` ttk style for the current appearance mode.

    Called from every screen that hosts a treeview so colour, padding, and
    selection styles stay consistent with the surrounding customtkinter
    widgets in both light and dark mode.
    """
    appearance = ctk.get_appearance_mode()
    if appearance == "Dark":
        bg, fg, sel_bg, heading_bg, border = "#2b2b2b", "#dce4ee", "#1f6aa5", "#1a1a2e", "#3a3a3a"
        row_odd, row_even = "#2b2b2b", "#323232"
    else:
        bg, fg, sel_bg, heading_bg, border = "#f0f0f0", "#1a1a1a", "#1f6aa5", "#dde1e7", "#cccccc"
        row_odd, row_even = "#ffffff", "#f5f5f5"

    style.theme_use("default")
    style.configure(
        "App.Treeview",
        background=bg,
        foreground=fg,
        rowheight=28,
        fieldbackground=bg,
        borderwidth=0,
        font=("", 13),
    )
    style.configure(
        "App.Treeview.Heading",
        background=heading_bg,
        foreground=fg,
        font=("", 13, "bold"),
        borderwidth=1,
        relief="flat",
    )
    style.map(
        "App.Treeview",
        background=[("selected", sel_bg)],
        foreground=[("selected", "#ffffff")],
    )
