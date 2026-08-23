"""Money handling.

Every amount in this system is an integer count of the currency's minor unit
(paise for INR), exactly as Razorpay's APIs represent them. Floats are never
used for money anywhere in this codebase.
"""

from __future__ import annotations

from typing import Final

MINOR_UNITS_PER_MAJOR: Final[int] = 100


class Paise(int):
    """An amount in paise. A distinct type so it cannot be confused with rupees."""

    __slots__ = ()

    @classmethod
    def from_rupees(cls, rupees: int) -> Paise:
        return cls(rupees * MINOR_UNITS_PER_MAJOR)

    @property
    def rupees(self) -> float:
        """Display only. Never use the result for arithmetic."""
        return self / MINOR_UNITS_PER_MAJOR

    def __str__(self) -> str:
        return f"Rs.{self / MINOR_UNITS_PER_MAJOR:,.2f}"
