"""Date ranges in resume text: detection and parsing to months.

Covers the rendering contract's four formats (``Jan 2020``, ``2020-01``, ``01/2020``,
``2020``), full month names, the separators en dash, em dash, hyphen and ``to``, and the
ongoing words ``Present`` / ``present`` / ``Current`` (any case, plus ``Now``).

Parsing conventions (the same as the ground truth): a year-only start is January and a
year-only end is December; an ongoing end is the ``as_of`` month; an end after ``as_of`` is
clamped to it. Months are inclusive, so Jan 2020 - Jan 2020 is one month. ``as_of`` is always
passed in; nothing here reads today's date.
"""

import re
from dataclasses import dataclass
from datetime import date

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_YEAR = r"(?:19|20)\d{2}"
# Alternation order matters: the ISO and numeric forms must win over a bare year.
_POINT = (
    rf"(?:{_MONTH}\s+{_YEAR}"  # month_name: Jan 2020
    rf"|{_YEAR}-(?:0[1-9]|1[0-2])"  # iso: 2020-01
    rf"|(?:0?[1-9]|1[0-2])/{_YEAR}"  # numeric: 01/2020
    rf"|{_YEAR})"  # year_only: 2020
)
_ONGOING = r"(?i:present|current|now)"
_SEPARATOR = r"(?:\s*[\u2013\u2014-]\s*|\s+to\s+)"

DATE_RANGE = re.compile(
    rf"(?<![\w/-])(?P<start>{_POINT}){_SEPARATOR}(?P<end>{_POINT}|{_ONGOING})(?![\w/-])"
)

_MONTH_NAME = re.compile(r"([A-Za-z]+)\.?\s+(\d{4})")
_ISO = re.compile(r"(\d{4})-(\d{2})")
_NUMERIC = re.compile(r"(\d{1,2})/(\d{4})")
_YEAR_ONLY = re.compile(r"\d{4}")
MONTH_NUMBERS = {
    name: number
    for number, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1
    )
}
ONGOING_WORDS = frozenset({"present", "current", "now"})


class DateRangeError(ValueError):
    """Text isn't a date range, or its start is after its end."""


@dataclass(frozen=True, order=True)
class YearMonth:
    """A calendar month."""

    year: int
    month: int

    @property
    def index(self) -> int:
        """Months since year 0, so differences are month counts."""
        return self.year * 12 + self.month - 1

    @classmethod
    def of(cls, day: date) -> YearMonth:
        """The month containing ``day``."""
        return cls(day.year, day.month)

    def __str__(self) -> str:
        """ISO form, ``2020-01``."""
        return f"{self.year:04d}-{self.month:02d}"


@dataclass(frozen=True)
class DateRange:
    """A parsed date range; ``months`` counts both ends."""

    start: YearMonth
    end: YearMonth
    ongoing: bool = False  # the end was Present / Current / Now
    clamped: bool = False  # the stated end was after as_of and was cut back to it

    @property
    def months(self) -> int:
        """Inclusive month count: (end - start) + 1."""
        return self.end.index - self.start.index + 1


def find_date_range(text: str) -> re.Match[str] | None:
    """The first date range in ``text``, if any."""
    return DATE_RANGE.search(text)


def is_date_range(text: str) -> bool:
    """True if ``text`` (stripped) is exactly one date range and nothing else."""
    return DATE_RANGE.fullmatch(text.strip()) is not None


def parse_point(text: str, *, is_end: bool) -> YearMonth:
    """One end of a range. A bare year is January as a start and December as an end."""
    text = text.strip()
    if match := _MONTH_NAME.fullmatch(text):
        return YearMonth(int(match[2]), MONTH_NUMBERS[match[1][:3].casefold()])
    if match := _ISO.fullmatch(text):
        return YearMonth(int(match[1]), int(match[2]))
    if match := _NUMERIC.fullmatch(text):
        return YearMonth(int(match[2]), int(match[1]))
    if _YEAR_ONLY.fullmatch(text):
        return YearMonth(int(text), 12 if is_end else 1)
    raise DateRangeError(f"not a date: {text!r}")


def parse_date_range(text: str, as_of: date) -> DateRange:
    """Parse text that is exactly one date range (see the module docstring for conventions).

    Raises ``DateRangeError`` if ``text`` isn't a date range or its start is after its end.
    """
    match = DATE_RANGE.fullmatch(text.strip())
    if not match:
        raise DateRangeError(f"not a date range: {text!r}")
    limit = YearMonth.of(as_of)
    start = parse_point(match["start"], is_end=False)
    ongoing = match["end"].casefold() in ONGOING_WORDS
    end = limit if ongoing else parse_point(match["end"], is_end=True)
    clamped = end > limit
    if clamped:
        end = limit
    if start > end:
        raise DateRangeError(f"start {start} is after end {end} in {text!r}")
    return DateRange(start, end, ongoing, clamped)
