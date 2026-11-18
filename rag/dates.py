"""Date-range detection in resume text (detection only; parsing to months is step 6).

Covers the rendering contract's four formats (``Jan 2020``, ``2020-01``, ``01/2020``,
``2020``), full month names, the separators en dash, em dash, hyphen and ``to``, and the
ongoing words ``Present`` / ``present`` / ``Current`` (any case, plus ``Now``).
"""

import re

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


def find_date_range(text: str) -> re.Match[str] | None:
    """The first date range in ``text``, if any."""
    return DATE_RANGE.search(text)


def is_date_range(text: str) -> bool:
    """True if ``text`` (stripped) is exactly one date range and nothing else."""
    return DATE_RANGE.fullmatch(text.strip()) is not None
