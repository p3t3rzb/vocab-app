"""Common base class for every full-window screen.

:class:`BaseScreen` is concrete: subclasses implement :meth:`build`
(required) and optionally override :meth:`on_show` / :meth:`on_destroy`.
There is no abstract contract beyond ``build`` — forcing every subclass to
declare empty ``on_show`` / ``on_destroy`` methods would be cargo cult.

Subclasses must call ``super().__init__(app)`` and may then access
``self._app`` and (if a database has been opened) ``self._ctx``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import customtkinter as ctk

if TYPE_CHECKING:
    from .app import App
    from .db_context import DbContext


class BaseScreen(ctk.CTkFrame):
    """Lifecycle-aware base for screens managed by :class:`App._swap`.

    Lifecycle:
      1. ``__init__`` runs ``build()`` then binds ``<Destroy>`` so
         ``on_destroy()`` fires exactly once when Tk tears this frame down.
      2. ``App._swap`` calls ``on_show()`` after packing the frame.
      3. Tk fires ``<Destroy>`` (filtered to ``event.widget is self``),
         which invokes ``on_destroy()``.
    """

    def __init__(self, app: App) -> None:
        super().__init__(app, corner_radius=0)
        self._app: App = app
        self._destroyed = False

        self.build()
        self.bind("<Destroy>", self._dispatch_destroy)

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    @property
    def _ctx(self) -> DbContext:
        """Active database context — convenience proxy to ``self._app.ctx``."""
        return self._app.ctx

    # ------------------------------------------------------------------
    # Hooks (override in subclasses)
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Construct widgets. Subclasses must override."""
        raise NotImplementedError(
            f"{type(self).__name__} must override BaseScreen.build()"
        )

    def on_show(self) -> None:
        """Called by :meth:`App._swap` after the frame is packed. No-op by default."""

    def on_destroy(self) -> None:
        """Called once when this frame is torn down. No-op by default."""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dispatch_destroy(self, event: object) -> None:
        """Filter the ``<Destroy>`` event to fire ``on_destroy`` exactly once."""
        if self._destroyed:
            return
        if getattr(event, "widget", None) is not self:
            return
        self._destroyed = True
        self.on_destroy()
