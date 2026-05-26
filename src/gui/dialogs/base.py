"""Base class for modal dialogs.

Encapsulates the boilerplate that every dialog in this app needs:

- title/geometry setup,
- the ``transient`` → ``grab_set`` → ``lift`` → ``focus_force`` sequence,
- the deferred re-grab that some window managers require,
- standard ``<Return>``/``<Escape>`` key bindings.

Subclasses implement only :meth:`_build`.
"""
from __future__ import annotations

from typing import Callable

import customtkinter as ctk


class BaseDialog(ctk.CTkToplevel):
    """Modal :class:`CTkToplevel` with the usual grab/focus boilerplate."""

    _SAFE_GRAB_DELAY_MS = 50

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        *,
        title: str,
        size: str,
        resizable: bool = False,
    ) -> None:
        super().__init__(master)
        self.title(title)
        self.geometry(size)
        self.resizable(resizable, resizable)

        self._build()

        self.transient(master)  # type: ignore[arg-type]
        self.grab_set()
        self.lift()
        self.focus_force()
        self.after(self._SAFE_GRAB_DELAY_MS, self._safe_grab)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Construct dialog widgets. Subclasses must override."""
        raise NotImplementedError(
            f"{type(self).__name__} must override BaseDialog._build()"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def bind_default_keys(
        self,
        *,
        on_save: Callable[[], None],
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        """Wire ``Enter`` to ``on_save`` and ``Escape`` to ``on_cancel`` (or destroy)."""
        cancel = on_cancel if on_cancel is not None else self.destroy
        self.bind("<Return>", lambda _e: on_save())
        self.bind("<Escape>", lambda _e: cancel())

    def _safe_grab(self) -> None:
        """Re-grab focus once the window is fully realised (ignoring failures)."""
        try:
            self.grab_set()
        except Exception:
            pass
