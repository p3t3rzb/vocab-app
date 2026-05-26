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
        self._safe_grab_after_id: str | None = self.after(
            self._SAFE_GRAB_DELAY_MS, self._safe_grab,
        )
        self.bind("<Destroy>", self._on_destroy_event, add="+")

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
        self._safe_grab_after_id = None
        try:
            self.grab_set()
        except Exception:
            pass

    def _on_destroy_event(self, event: object) -> None:
        """Cancel the pending ``_safe_grab`` so it doesn't fire after destroy."""
        if getattr(event, "widget", None) is not self:
            return
        if self._safe_grab_after_id is not None:
            try:
                self.after_cancel(self._safe_grab_after_id)
            except Exception:
                pass
            self._safe_grab_after_id = None
