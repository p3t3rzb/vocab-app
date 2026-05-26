"""Shared widget builders and the global ``ttk.Treeview`` style.

Every screen consumes :func:`build_header` for its top bar and
:func:`build_tree` for any tabular content; :func:`apply_treeview_style`
keeps colours consistent with the surrounding customtkinter widgets in
both light and dark mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from tkinter import ttk
from typing import Callable, Sequence

import customtkinter as ctk

from .theme import Colors, Fonts, Spacing


HeaderRight = Callable[[ctk.CTkFrame], ctk.CTkBaseClass]


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Declarative spec for one column in :func:`build_tree`."""

    key: str
    heading: str
    width: int
    minwidth: int = 60
    anchor: str | None = None


class ScreenHeader(ctk.CTkFrame):
    """Top-of-screen header frame: optional Back button + centred title slot.

    The title label is exposed as :attr:`title_label` so callers can rewrite
    its text later (e.g. once async-loaded data resolves).
    """

    def __init__(
        self,
        parent: ctk.CTkBaseClass,
        *,
        title: str,
        on_back: Callable[[], None] | None,
    ) -> None:
        super().__init__(parent, fg_color="transparent")
        self.grid_columnconfigure(1, weight=1)

        self.back_button: ctk.CTkButton | None = None
        if on_back is not None:
            self.back_button = ctk.CTkButton(
                self, text="← Back", width=80, command=on_back
            )
            self.back_button.grid(row=0, column=0, sticky="w")

        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(**Fonts.HEADER),
        )
        self.title_label.grid(
            row=0,
            column=1,
            padx=Spacing.HEADER_TITLE_PAD,
            sticky="w",
        )

    def set_title(self, title: str) -> None:
        """Rewrite the header title at runtime."""
        self.title_label.configure(text=title)


def build_header(
    parent: ctk.CTkBaseClass,
    *,
    title: str,
    on_back: Callable[[], None] | None,
    right_widget_factory: HeaderRight | None = None,
) -> ScreenHeader:
    """Build the standard Back-button + title (+ optional right element) header.

    ``on_back=None`` suppresses the Back button (e.g. on the home screen).
    The optional ``right_widget_factory`` receives the header frame and is
    responsible for gridding the widget it creates into column 2 itself.
    """
    header = ScreenHeader(parent, title=title, on_back=on_back)
    if right_widget_factory is not None:
        right_widget_factory(header)
    return header


def build_tree(
    parent: ctk.CTkBaseClass,
    *,
    columns: Sequence[ColumnSpec],
    selectmode: str = "browse",
) -> tuple[ttk.Treeview, ttk.Scrollbar]:
    """Build a styled treeview + vertical scrollbar pair.

    The caller still owns gridding the tree and scrollbar into ``parent``
    (so it can mix them with other widgets), but the per-column heading and
    width/anchor calls are done here.
    """
    tree = ttk.Treeview(
        parent,
        columns=tuple(c.key for c in columns),
        show="headings",
        selectmode=selectmode,
        style="App.Treeview",
    )
    for col in columns:
        tree.heading(col.key, text=col.heading)
        if col.anchor is not None:
            tree.column(col.key, width=col.width, minwidth=col.minwidth, anchor=col.anchor)
        else:
            tree.column(col.key, width=col.width, minwidth=col.minwidth)

    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    return tree, scrollbar


def apply_treeview_style() -> None:
    """Configure the shared ``App.Treeview`` ttk style for the current appearance mode.

    Called from every screen that hosts a treeview so colour, padding, and
    selection styles stay consistent with the surrounding customtkinter
    widgets in both light and dark mode.
    """
    style = ttk.Style()
    is_dark = ctk.get_appearance_mode() == "Dark"
    if is_dark:
        bg = Colors.TREE_DARK_BG
        fg = Colors.TREE_DARK_FG
        sel_bg = Colors.TREE_DARK_SEL_BG
        heading_bg = Colors.TREE_DARK_HEADING_BG
    else:
        bg = Colors.TREE_LIGHT_BG
        fg = Colors.TREE_LIGHT_FG
        sel_bg = Colors.TREE_LIGHT_SEL_BG
        heading_bg = Colors.TREE_LIGHT_HEADING_BG

    style.theme_use("default")
    style.configure(
        "App.Treeview",
        background=bg,
        foreground=fg,
        rowheight=Colors.TREE_ROW_HEIGHT,
        fieldbackground=bg,
        borderwidth=0,
        font=("", Fonts.SMALL["size"]),
    )
    style.configure(
        "App.Treeview.Heading",
        background=heading_bg,
        foreground=fg,
        font=("", Fonts.SMALL["size"], "bold"),
        borderwidth=1,
        relief="flat",
    )
    style.map(
        "App.Treeview",
        background=[("selected", sel_bg)],
        foreground=[("selected", "#ffffff")],
    )
