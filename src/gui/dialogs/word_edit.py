"""Modal "Add word" / "Edit word" dialog.

Pass ``word=None`` to create a new word, or an existing :class:`Word`
to edit it. Newly created words have ``next_rep_*_at`` left ``NULL``;
the practice queue picks them up via the never-practiced ("new") bucket.
"""
from __future__ import annotations

from tkinter import messagebox

import customtkinter as ctk

from src.database import WordRepository, get_session
from src.database.models import Word

from ..theme import WindowSizes
from .base import BaseDialog


class WordEditDialog(BaseDialog):
    """Two-field modal that adds a new word or edits an existing one."""

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        src_lang: str,
        tgt_lang: str,
        word: Word | None,
    ) -> None:
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._word = word
        self._saved = False

        super().__init__(
            master,
            title="Edit Word" if word else "Add Word",
            size=WindowSizes.WORD_EDIT_DIALOG,
        )

    def _build(self) -> None:
        """Lay out the two text fields, action buttons, and key bindings."""
        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self,
            text=f"{self._src_lang}:",
            anchor="e",
            width=110,
        ).grid(row=0, column=0, padx=(20, 8), pady=(28, 6), sticky="e")

        self._src_entry = ctk.CTkEntry(self, width=260)
        self._src_entry.grid(row=0, column=1, padx=(0, 20), pady=(28, 6), sticky="ew")

        ctk.CTkLabel(
            self,
            text=f"{self._tgt_lang}:",
            anchor="e",
            width=110,
        ).grid(row=1, column=0, padx=(20, 8), pady=6, sticky="e")

        self._tgt_entry = ctk.CTkEntry(self, width=260)
        self._tgt_entry.grid(row=1, column=1, padx=(0, 20), pady=6, sticky="ew")

        if self._word is not None:
            self._src_entry.insert(0, self._word.source_text)
            self._tgt_entry.insert(0, self._word.target_text)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(20, 16))

        ctk.CTkButton(btn_frame, text="Cancel", width=100, command=self.destroy).pack(
            side="left", padx=8
        )
        ctk.CTkButton(btn_frame, text="Save", width=100, command=self._save).pack(
            side="left", padx=8
        )

        self._src_entry.focus()
        self.bind_default_keys(on_save=self._save)

    def _save(self) -> None:
        """Validate inputs, persist the new/updated word, and close the dialog."""
        src_text = self._src_entry.get().strip()
        tgt_text = self._tgt_entry.get().strip()

        if not src_text or not tgt_text:
            messagebox.showwarning(
                "Missing fields",
                "Both fields must be filled in.",
                parent=self,
            )
            return

        try:
            if self._word is None:
                with get_session() as session:
                    repo = WordRepository(session)
                    if repo.find_by_source_text(src_text) is not None:
                        messagebox.showwarning(
                            "Duplicate word",
                            f'A word with {self._src_lang.lower()} "{src_text}" already exists.',
                            parent=self,
                        )
                        return
                    new_id = repo.get_next_id()
                    repo.add(Word(
                        id=new_id,
                        source_text=src_text,
                        target_text=tgt_text,
                    ))
            else:
                with get_session() as session:
                    word = WordRepository(session).get_by_id(self._word.id)
                    if word is not None:
                        word.source_text = src_text
                        word.target_text = tgt_text
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)
            return

        self._saved = True
        self.destroy()
