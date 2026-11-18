"""Candidate metadata extracted from a parsed resume.

Rule-based and deterministic. All durations come from ``rag.timeline`` with an injected
``as_of`` date. Contact details (email, phone) are deliberately not extracted: they would end
up in vector-store metadata, and nothing downstream needs them.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum
from pathlib import PurePosixPath

from rag.dates import ONGOING_WORDS, DateRange, YearMonth, find_date_range
from rag.sections import ParsedResume, Role, Section, SectionKind, split_roles
from rag.skill_matcher import mentioned_skills
from rag.skills import expand_implied
from rag.timeline import RoleRecord, build_timeline, merged_months, months_to_years


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
    """What the pipeline knows about a candidate. Contact details are deliberately absent."""

    candidate_id: str
    resume_path: str
    name: str
    roles: tuple[RoleRecord, ...]
    total_months: int
    total_years: float  # display only; compare total_months
    education: tuple[Education, ...]
    highest_education: Education | None
    skills: tuple[str, ...] = ()  # canonical names, implications included, sorted
    skill_months: dict[str, int] = field(default_factory=dict)  # evidenced months per skill
    skill_years: dict[str, float] = field(default_factory=dict)  # display only
    # Where each skill was mentioned: "skills", "projects" or "experience:<role index>".
    skill_evidence: dict[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """A JSON-ready dict; ``from_dict`` restores an equal object."""
        return {
            "candidate_id": self.candidate_id,
            "resume_path": self.resume_path,
            "name": self.name,
            "total_months": self.total_months,
            "total_years": self.total_years,
            "roles": [_role_to_dict(r) for r in self.roles],
            "education": [_education_to_dict(e) for e in self.education],
            "highest_education": (
                _education_to_dict(self.highest_education) if self.highest_education else None
            ),
            "skills": list(self.skills),
            "skill_months": dict(self.skill_months),
            "skill_years": dict(self.skill_years),
            "skill_evidence": {k: list(v) for k, v in self.skill_evidence.items()},
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CandidateMetadata:
        """Inverse of ``to_dict``."""
        highest = data["highest_education"]
        return cls(
            candidate_id=data["candidate_id"],
            resume_path=data["resume_path"],
            name=data["name"],
            roles=tuple(_role_from_dict(r) for r in data["roles"]),
            total_months=data["total_months"],
            total_years=data["total_years"],
            education=tuple(_education_from_dict(e) for e in data["education"]),
            highest_education=_education_from_dict(highest) if highest else None,
            skills=tuple(data["skills"]),
            skill_months=dict(data["skill_months"]),
            skill_years=dict(data["skill_years"]),
            skill_evidence={k: tuple(v) for k, v in data["skill_evidence"].items()},
            warnings=tuple(data["warnings"]),
        )


def _role_to_dict(role: RoleRecord) -> dict:
    dates = role.dates
    return {
        "title": role.title,
        "company": role.company,
        "date_text": role.date_text,
        "start": str(dates.start) if dates else None,
        "end": str(dates.end) if dates else None,
        "ongoing": dates.ongoing if dates else False,
        "clamped": dates.clamped if dates else False,
        "months": role.months,
        "is_internship": role.is_internship,
    }


def _year_month(text: str) -> YearMonth:
    year, month = text.split("-")
    return YearMonth(int(year), int(month))


def _role_from_dict(data: dict) -> RoleRecord:
    dates = None
    if data["start"] is not None:
        dates = DateRange(
            _year_month(data["start"]), _year_month(data["end"]), data["ongoing"], data["clamped"]
        )
    return RoleRecord(
        data["title"], data["company"], data["date_text"], dates, data["is_internship"]
    )


def _education_to_dict(entry: Education) -> dict:
    return {
        "degree": entry.degree,
        "level": entry.level.name if entry.level else None,
        "field": entry.field,
        "institution": entry.institution,
        "end_year": entry.end_year,
    }


def _education_from_dict(data: dict) -> Education:
    level = DegreeLevel[data["level"]] if data["level"] else None
    return Education(data["degree"], level, data["field"], data["institution"], data["end_year"])


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


# --- Skills -------------------------------------------------------------------------------


def _text(lines: tuple[str, ...] | list[str]) -> str:
    return "\n".join(lines)


def _evidence_key(location: str) -> tuple[int, int]:
    kind, _, index = location.partition(":")
    return ({"skills": 0, "experience": 1}.get(kind, 2), int(index or 0))


def _skills(
    parsed: ParsedResume, roles: list[Role], records: tuple[RoleRecord, ...]
) -> tuple[tuple[str, ...], dict[str, int], dict[str, tuple[str, ...]]]:
    """Skills, evidenced months and evidence locations.

    Sources: skills-section lines, role bodies (not the role header lines) and project
    sections. Every other section, and the header block, is ignored.
    """
    evidence: dict[str, set[str]] = defaultdict(set)
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def note(skills: frozenset[str], location: str) -> None:
        for skill in skills:
            evidence[skill].add(location)

    for section in parsed.of_kind(SectionKind.SKILLS):
        note(expand_implied(mentioned_skills(_text(section.lines))), "skills")
    for section in parsed.of_kind(SectionKind.PROJECTS):
        note(expand_implied(mentioned_skills(_text(section.lines))), "projects")
    for index, (role, record) in enumerate(zip(roles, records, strict=True)):
        skills = expand_implied(mentioned_skills(_text(role.lines)))
        note(skills, f"experience:{index}")
        if record.counts_towards_experience and record.dates:
            for skill in skills:
                intervals[skill].append((record.dates.start.index, record.dates.end.index))
    names = tuple(sorted(evidence))
    months = {skill: merged_months(intervals[skill]) for skill in names}
    where = {skill: tuple(sorted(evidence[skill], key=_evidence_key)) for skill in names}
    return names, months, where


def _warnings(records: tuple[RoleRecord, ...], as_of: date) -> tuple[str, ...]:
    warnings = []
    for record in records:
        if record.dates is None:
            warnings.append(f"{record.title}: could not parse dates {record.date_text!r}")
        elif record.dates.clamped:
            warnings.append(f"{record.title}: end date after {as_of.isoformat()}; clamped")
    return tuple(warnings)


def extract_metadata(resume_path: str, parsed: ParsedResume, as_of: date) -> CandidateMetadata:
    """Metadata for one resume; ``resume_path`` is relative to the workspace root."""
    roles = [r for s in parsed.of_kind(SectionKind.EXPERIENCE) for r in split_roles(s)]
    timeline = build_timeline(roles, as_of)
    education = [e for s in parsed.of_kind(SectionKind.EDUCATION) for e in parse_education(s)]
    skills, skill_months, evidence = _skills(parsed, roles, timeline.roles)
    return CandidateMetadata(
        candidate_id=PurePosixPath(resume_path).stem,
        resume_path=resume_path,
        name=extract_name(parsed.header, resume_path),
        roles=timeline.roles,
        total_months=timeline.total_months,
        total_years=timeline.total_years,
        education=tuple(education),
        highest_education=highest_education(education),
        skills=skills,
        skill_months=skill_months,
        skill_years={skill: months_to_years(m) for skill, m in skill_months.items()},
        skill_evidence=evidence,
        warnings=_warnings(timeline.roles, as_of),
    )
