"""The experience timeline: role records, merged months and total experience.

Months (ints) are the source of truth everywhere in ``rag/``. Overlapping or adjacent roles
are merged so concurrent jobs count once, gaps don't count, and internships never count
towards experience. Years exist only for display, rounded half up to one decimal (the same
rule as the ground truth). Thresholds compare months: ``meets_min_years(59, 5)`` is False.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from rag.dates import DateRange, DateRangeError, parse_date_range
from rag.sections import Role

# "Intern" / "Internship" as whole words: "Internal Tools Engineer" and "International
# Sales" don't match.
INTERNSHIP_TITLE = re.compile(r"\b(?:intern|internship)\b", re.IGNORECASE)


def is_internship_title(title: str) -> bool:
    """True if the role title says it's an internship."""
    return INTERNSHIP_TITLE.search(title) is not None


def months_to_years(months: int) -> float:
    """Display years: months / 12, rounded half up to one decimal (75 months -> 6.3)."""
    years = Decimal(months) / Decimal(12)
    return float(years.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def meets_min_years(months: int, years: int | float) -> bool:
    """True if ``months`` of experience is at least ``years`` years."""
    return months >= years * 12


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge inclusive ``(start, end)`` month-index intervals that overlap or touch."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def merged_months(intervals: Iterable[tuple[int, int]]) -> int:
    """Months covered by the intervals, counting overlaps once."""
    return sum(end - start + 1 for start, end in merge_intervals(intervals))


@dataclass(frozen=True)
class RoleRecord:
    """One role with parsed dates. ``dates`` is None when its date text couldn't be parsed."""

    title: str
    company: str
    date_text: str
    dates: DateRange | None
    is_internship: bool

    @property
    def months(self) -> int:
        """Inclusive months in the role (0 if the dates couldn't be parsed)."""
        return self.dates.months if self.dates else 0

    @property
    def counts_towards_experience(self) -> bool:
        """Parsed and not an internship."""
        return self.dates is not None and not self.is_internship


@dataclass(frozen=True)
class Timeline:
    """All roles plus the merged experience total."""

    roles: tuple[RoleRecord, ...]
    total_months: int

    @property
    def total_years(self) -> float:
        """Display only; compare months with ``meets_min_years``."""
        return months_to_years(self.total_months)


def role_record(role: Role, as_of: date) -> RoleRecord:
    """Parse one split role's dates against ``as_of``."""
    try:
        dates = parse_date_range(role.date_text, as_of)
    except DateRangeError:
        dates = None
    return RoleRecord(
        role.title, role.company, role.date_text, dates, is_internship_title(role.title)
    )


def build_timeline(roles: Sequence[Role], as_of: date) -> Timeline:
    """Role records and merged total months (internships and unparseable dates excluded)."""
    records = tuple(role_record(role, as_of) for role in roles)
    intervals = [
        (r.dates.start.index, r.dates.end.index)
        for r in records
        if r.counts_towards_experience and r.dates
    ]
    return Timeline(records, merged_months(intervals))
