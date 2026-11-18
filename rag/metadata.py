"""Candidate metadata extracted from a parsed resume (skills arrive in step 6b).

Rule-based and deterministic. All durations come from ``rag.timeline`` with an injected
``as_of`` date. Contact details (email, phone) are deliberately not extracted: they would end
up in vector-store metadata, and nothing downstream needs them.
"""

import re
from dataclasses import dataclass
from datetime import date
from enum import IntEnum
from pathlib import PurePosixPath

from rag.dates import ONGOING_WORDS, find_date_range
from rag.sections import ParsedResume, Section, SectionKind, split_roles
from rag.timeline import RoleRecord, build_timeline


class DegreeLevel(IntEnum):
    """Education levels, ordered so a higher value is a higher degree."""

    SCHOOL = 1
    DIPLOMA = 2
    BACHELORS = 3
    MASTERS = 4
    PHD = 5


# Abbreviations, matched against the whole degree with dots, spaces and any "(Hons)" removed
# ("B.Sc." -> "bsc"), so an institution such as "Marina Bay ..." never reads as "M.A.".
_ABBREVIATIONS: dict[str, DegreeLevel] = {
    **dict.fromkeys(("phd", "dphil", "edd"), DegreeLevel.PHD),
    **dict.fromkeys(
        ("msc", "ms", "ma", "meng", "mtech", "mba", "mca", "mphil", "mres", "mcom"),
        DegreeLevel.MASTERS,
    ),
    **dict.fromkeys(
        ("bsc", "bs", "ba", "beng", "be", "btech", "bca", "bcom", "bba"), DegreeLevel.BACHELORS
    ),
}
# Spelled-out degrees, matched at the start of the degree text.
_LONG_FORMS: tuple[tuple[re.Pattern[str], DegreeLevel], ...] = (
    (re.compile(r"doctor(?:ate| of)\b", re.IGNORECASE), DegreeLevel.PHD),
    (re.compile(r"master(?:'?s)?\b", re.IGNORECASE), DegreeLevel.MASTERS),
    (re.compile(r"bachelor(?:'?s)?\b", re.IGNORECASE), DegreeLevel.BACHELORS),
    (re.compile(r"(?:diploma|associate degree)\b", re.IGNORECASE), DegreeLevel.DIPLOMA),
    (re.compile(r"(?:high|secondary) school\b|a[- ]levels?\b", re.IGNORECASE), DegreeLevel.SCHOOL),
)
_CONTACT = re.compile(r"@|\|")
_PHONE_LIKE = re.compile(r"\d[\d ()+-]{5,}\d")
_YEAR = re.compile(r"(?:19|20)\d{2}")


def degree_level(degree: str) -> DegreeLevel | None:
    """The level of a degree string ("M.Sc." -> MASTERS), or None if unrecognised."""
    compact = re.sub(r"\(.*?\)|[.\s]", "", degree).casefold()
    if compact in _ABBREVIATIONS:
        return _ABBREVIATIONS[compact]
    for pattern, level in _LONG_FORMS:
        if pattern.match(degree.strip()):
            return level
    return None


@dataclass(frozen=True)
class Education:
    """One education entry."""

    degree: str
    level: DegreeLevel | None
    field: str
    institution: str
    end_year: int | None


@dataclass(frozen=True)
class CandidateMetadata:
    """What the pipeline knows about a candidate (no contact details)."""

    candidate_id: str
    resume_path: str
    name: str
    roles: tuple[RoleRecord, ...]
    total_months: int
    total_years: float  # display only; compare total_months
    education: tuple[Education, ...]
    highest_education: Education | None


def extract_name(header: tuple[str, ...] | list[str], resume_path: str = "") -> str:
    """The first header line that isn't contact info; else a name from the file stem."""
    for line in header:
        if not _CONTACT.search(line) and not _PHONE_LIKE.search(line):
            return line.strip()
    stem = PurePosixPath(resume_path).stem
    return " ".join(part.capitalize() for part in re.split(r"[_\-\s]+", stem) if part)


def _degree_and_field(line: str) -> tuple[str, str]:
    degree, sep, field = line.partition(" in ")
    return (degree.strip(), field.strip()) if sep else (line.strip(), "")


def parse_education(section: Section) -> list[Education]:
    """Education entries: a degree line, then an institution line carrying a date range.

    The degree line is ``Degree in Field``; the institution is the text before the date range
    on the next line with one, and the end year is the range's last year.
    """
    entries = []
    lines = list(section.lines)
    i = 0
    while i < len(lines):
        degree, field = _degree_and_field(lines[i])
        level = degree_level(degree)
        if level is None:
            i += 1
            continue
        institution, end_year, j = "", None, i + 1
        while j < len(lines) and degree_level(_degree_and_field(lines[j])[0]) is None:
            if match := find_date_range(lines[j]):
                institution = lines[j][: match.start()].strip().rstrip(",").strip()
                end = match["end"]
                years = _YEAR.findall(end)
                end_year = int(years[-1]) if years and end.casefold() not in ONGOING_WORDS else None
                j += 1
                break
            j += 1
        entries.append(Education(degree, level, field, institution, end_year))
        i = j
    return entries


def highest_education(entries: list[Education]) -> Education | None:
    """The highest level; a tie goes to the latest end year, then to the first listed."""
    ranked = [(e.level, e.end_year or 0, -n, e) for n, e in enumerate(entries) if e.level]
    return max(ranked, key=lambda t: t[:3])[3] if ranked else None


def extract_metadata(resume_path: str, parsed: ParsedResume, as_of: date) -> CandidateMetadata:
    """Metadata for one resume; ``resume_path`` is relative to the workspace root."""
    roles = [r for s in parsed.of_kind(SectionKind.EXPERIENCE) for r in split_roles(s)]
    timeline = build_timeline(roles, as_of)
    education = [e for s in parsed.of_kind(SectionKind.EDUCATION) for e in parse_education(s)]
    return CandidateMetadata(
        candidate_id=PurePosixPath(resume_path).stem,
        resume_path=resume_path,
        name=extract_name(parsed.header, resume_path),
        roles=timeline.roles,
        total_months=timeline.total_months,
        total_years=timeline.total_years,
        education=tuple(education),
        highest_education=highest_education(education),
    )
