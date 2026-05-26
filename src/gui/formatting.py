"""Human-readable formatting for timestamps and durations.

Each function returns a short string suitable for inline labels:

- :func:`format_past`        — ``"just now"`` / ``"5m ago"`` / ``"2d 3h ago"``
- :func:`format_future`      — ``"due now"`` / ``"in 12m"``  / ``"in 5d 2h"``
- :func:`format_due`         — ``"Due now"`` / ``"in 2d 4h"`` / ``"–"``  (None ⇒ "–")
- :func:`format_timestamp`   — ``"YYYY-MM-DD  HH:MM:SS"``

``format_future`` and ``format_due`` overlap but have intentionally
different contracts (none-handling, "due now" vs "–", minute precision)
and are kept separate.
"""
from __future__ import annotations

import time
from datetime import datetime


def format_past(seconds: int) -> str:
    """Render an elapsed duration as a casual "X ago" string."""
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h ago" if minutes == 0 else f"{hours}h {minutes}m ago"
    days, hours = divmod(hours, 24)
    return f"{days}d ago" if hours == 0 else f"{days}d {hours}h ago"


def format_future(seconds: int) -> str:
    """Render a forward-looking duration as ``"in 12m"`` / ``"in 5d 2h"``."""
    if seconds <= 0:
        return "due now"
    minutes = seconds // 60
    if minutes < 60:
        return f"in {minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"in {hours}h" if minutes == 0 else f"in {hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"in {days}d" if hours == 0 else f"in {days}d {hours}h"


def format_due(ts: int | None) -> str:
    """Render a due timestamp as ``"Due now"`` / ``"in 2d 4h"`` / ``"–"``.

    ``None`` becomes ``"–"`` (no model trained yet), ``≤ now`` becomes
    ``"Due now"``, and otherwise a coarse "in Xd Yh" string. Minute-level
    precision is only shown when the remaining time is under an hour.
    """
    if ts is None:
        return "–"
    remaining = ts - int(time.time())
    if remaining <= 0:
        return "Due now"
    days, rem = divmod(remaining, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days >= 1:
        return f"in {days}d {hours}h" if hours else f"in {days}d"
    if hours >= 1:
        return f"in {hours}h {minutes}m" if minutes else f"in {hours}h"
    return f"in {minutes}m" if minutes else "Due now"


def format_timestamp(ts: int) -> str:
    """Format a Unix timestamp as ``"YYYY-MM-DD  HH:MM:SS"``."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d  %H:%M:%S")
