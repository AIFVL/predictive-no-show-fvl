"""Date helpers for appointment filtering (calendar year 2026)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Sequence, TypeVar

CALENDAR_YEAR = 2026
DEFAULT_DAYS_WINDOW = 8
ALLOWED_DAYS_WINDOWS = (0, 8, 15, 30)


def normalize_days_window(days: int | None) -> int | None:
    """Return validated window size. 0 disables the filter."""
    if days is None:
        return DEFAULT_DAYS_WINDOW
    value = int(days)
    if value == 0:
        return None
    if value not in ALLOWED_DAYS_WINDOWS:
        raise ValueError(f"days must be one of {ALLOWED_DAYS_WINDOWS}")
    return value


def appointment_to_date(appointment) -> date:
    return date(CALENDAR_YEAR, int(appointment.month), int(appointment.day))


def is_within_forward_window(
    appointment,
    days: int | None,
    *,
    reference: date | None = None,
) -> bool:
    """True if appointment falls between reference date and reference + days (inclusive)."""
    if days is None:
        return True

    ref = reference or date.today()
    appt_date = appointment_to_date(appointment)
    end = ref + timedelta(days=int(days))
    return ref <= appt_date <= end


T = TypeVar("T")


def filter_by_forward_window(
    appointments: Sequence[T],
    days: int | None,
    *,
    reference: date | None = None,
) -> list[T]:
    return [
        appt
        for appt in appointments
        if is_within_forward_window(appt, days, reference=reference)
    ]
