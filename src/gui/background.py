"""Background work runner that marshals worker-thread events onto the Tk loop.

A :class:`BackgroundJob` is a thin wrapper around a daemon thread and a
``queue.Queue``. The worker pushes ``(tag, *payload)`` tuples; the job's
``after``-polled drainer dispatches them to handlers registered by tag.

Composition (not inheritance) is the right shape: one screen can own
multiple ``BackgroundJob`` instances (e.g. the train screen runs the
trainer and then the scheduler), and the per-job state — queue, stop
event, polling — is encapsulated rather than mixed into the screen.

Typical usage::

    self._job = BackgroundJob(
        owner=self,
        handlers={
            "progress": self._on_progress,
            "done": self._on_done,
            "error": self._on_error,
        },
    )
    self._job.start(self._worker_fn, arg1, arg2)

The worker calls ``self._job.queue.put(("progress", 5, 100))`` etc., and
``self._job.stop_event`` is the cooperative cancellation flag.
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Mapping

import customtkinter as ctk

from .theme import PollIntervals


Handler = Callable[..., None]


class BackgroundJob:
    """Daemon-thread worker + Tk-main-thread event dispatcher."""

    def __init__(
        self,
        owner: ctk.CTkBaseClass,
        *,
        handlers: Mapping[str, Handler],
        poll_ms: int = PollIntervals.MS,
    ) -> None:
        self._owner = owner
        self._handlers = dict(handlers)
        self._poll_ms = poll_ms
        self.queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True`` while the worker thread is alive or the queue still has events."""
        return self._running

    def start(self, target: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        """Spawn ``target`` on a daemon thread and begin polling its queue."""
        if self._running:
            raise RuntimeError("BackgroundJob is already running")

        self.stop_event.clear()
        # Drain any leftover events from a prior run.
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

        self._running = True
        self._thread = threading.Thread(
            target=target, args=args, kwargs=kwargs, daemon=True,
        )
        self._thread.start()
        self._owner.after(self._poll_ms, self._poll)

    def stop(self) -> None:
        """Signal cooperative cancellation. The worker decides when to exit."""
        self.stop_event.set()

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        """Drain the queue, dispatch each event to its handler, reschedule."""
        if not self._running:
            return
        try:
            while True:
                event = self.queue.get_nowait()
                tag, *payload = event
                handler = self._handlers.get(tag)
                if handler is not None:
                    handler(*payload)
        except queue.Empty:
            pass

        # Stop polling once the thread is gone AND the queue is empty.
        thread_alive = self._thread is not None and self._thread.is_alive()
        if not thread_alive and self.queue.empty():
            self._running = False
            return

        self._owner.after(self._poll_ms, self._poll)
