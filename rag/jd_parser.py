"""Rule-based job description parser: JD text -> ``JobRequirements``.

Implements the JD contract in ``data/job_descriptions/README.md``:

- **Sections**: a heading is a whole line equal to a known heading (case-insensitive, one
  optional trailing colon). Lines before the first heading (title, company line) form an
  ``other`` section with no heading. ``HEADINGS`` is this module's own copy of the README's
  heading table, plus a few common synonyms.
- **Sources**: only the requirements and nice-to-have sections are sources of requirements.
  Skills and years anywhere else (the company blurb, responsibilities, benefits) are context.
- **Units**: each bullet is one unit; prose is split into sentences. A unit's skills come from
  ``rag.skill_matcher`` over the whole unit, without implication expansion (candidate skills
  are already expanded, so ``Django`` required matches a candidate with Django).
- **Years**: a years phrase in a unit that names skills is the minimum years of that unit's
  requirement(s); in a unit with no skill it is the minimum total experience.
- **Grouping**: skills joined by ``or`` (or ``/``) in a unit form one ``any_of`` group;
  otherwise each skill is its own must-have, and the unit's years apply to each.
- **Nice-to-haves** contribute skills only.
- **No requirements section**: no must-haves and no total-years minimum (hard filters off);
  every skill mentioned outside about / benefits sections becomes a nice-to-have, with a
  warning. Must-haves are never invented.

``query_text`` (what step 11 embeds) is the title plus the overview, responsibilities,
requirements and nice-to-have sections: no company blurb, no benefits.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from pathlib import PurePosixPath

import fs_tools
from rag.skill_matcher import Mention, find_mentions

NO_REQUIREMENTS_WARNING = "no requirements section found; hard filters disabled"
JD_EXTENSIONS = frozenset({".txt", ".pdf", ".docx"})


class JDSectionKind(StrEnum):
    """What a JD section holds."""

    ABOUT = "about"
    OVERVIEW = "overview"
    RESPONSIBILITIES = "responsibilities"
    REQUIREMENTS = "requirements"
    NICE_TO_HAVE = "nice_to_have"
    BENEFITS = "benefits"
    OTHER = "other"


_K = JDSectionKind
# The README's heading table first in each tuple, then synonyms seen in real postings.
HEADINGS: dict[JDSectionKind, tuple[str, ...]] = {
    _K.ABOUT: ("About us", "Who we are", "About the company", "Company"),
    _K.OVERVIEW: ("About the role", "The role", "The team", "Overview", "Role overview"),
    _K.RESPONSIBILITIES: (
        "What you'll do",
        "Responsibilities",
        "Your day-to-day",
        "Key responsibilities",
        "What you will do",
    ),
    _K.REQUIREMENTS: (
        "Requirements",
        "What you'll need",
        "Must have",
        "You have",
        "What we're looking for",
        "Must-haves",
        "Required skills",
        "Qualifications",
        "Minimum qualifications",
    ),
    _K.NICE_TO_HAVE: (
        "Nice to have",
        "Bonus points",
        "Preferred",
        "Preferred qualifications",
        "Nice-to-haves",
        "Desirable",
        "Good to have",
    ),
    _K.BENEFITS: (
        "Location and benefits",
        "Where you'll work",
        "Benefits",
        "Perks",
        "What we offer",
    ),
}

_KIND_BY_HEADING = {
    " ".join(h.split()).casefold(): kind for kind, headings in HEADINGS.items() for h in headings
}
QUERY_SECTIONS = (_K.OVERVIEW, _K.RESPONSIBILITIES, _K.REQUIREMENTS, _K.NICE_TO_HAVE)
SOURCE_SECTIONS = (_K.REQUIREMENTS, _K.NICE_TO_HAVE)

_BULLET = re.compile(r"^\s*(?:[-*\u2022\u2013\u00b7]|\d{1,2}[.)])\s+")
# Sentence ends: . ! ? followed by space and a capital, digit or opening quote/bracket. Common
# abbreviations ("e.g.", "i.e.") don't end a sentence.
_SENTENCE_END = re.compile(r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_NUMBER_WORD_TEXT = "zero one two three four five six seven eight nine ten eleven twelve"
_NUMBER_WORDS = {w: n for n, w in enumerate(_NUMBER_WORD_TEXT.split())}
_NUMBER = r"\d{1,2}|" + "|".join(_NUMBER_WORDS)
_YEARS = re.compile(
    rf"(?<![\w.])(?P<n>{_NUMBER})"
    rf"(?:\s*(?:-|\u2013|to)\s*(?:{_NUMBER}))?"  # a range "3-5 years": the lower bound
    r"(?:\s*\+|\s+or\s+more)?\s*(?:years?|yrs?)(?!\w)",
    re.IGNORECASE,
)
_OR_SEPARATOR = re.compile(r"\s*(?:,\s*)?(?:or|/)\s*", re.IGNORECASE)
_LIST_SEPARATOR = re.compile(r"\s*,\s*")


class JobDescriptionError(ValueError):
    """A JD source can't be loaded."""


@dataclass(frozen=True)
class JDSection:
    """One section; ``heading`` is None for the lines before the first heading."""

    kind: JDSectionKind
    heading: str | None
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        """The section's lines joined with newlines."""
        return "\n".join(self.lines)


@dataclass(frozen=True)
class Requirement:
    """A must-have: any one of ``any_of`` (canonical names), optionally for ``min_years``."""

    any_of: tuple[str, ...]
    min_years: int | None
    source_text: str

    def to_dict(self) -> dict:
        """JSON form."""
        return {
            "any_of": list(self.any_of),
            "min_years": self.min_years,
            "source_text": self.source_text,
        }


@dataclass(frozen=True)
class JobRequirements:
    """Everything the matcher needs from one JD."""

    title: str
    sections: tuple[JDSection, ...]
    must_have: tuple[Requirement, ...]
    min_total_years: int | None
    nice_to_have: tuple[str, ...]
    unparsed_requirements: tuple[str, ...]
    warnings: tuple[str, ...]
    query_text: str
    full_text: str
    critical_skills: tuple[str, ...] = field(default=())

    @property
    def has_requirements_section(self) -> bool:
        """Whether hard filters apply (a requirements section was found)."""
        return any(s.kind is _K.REQUIREMENTS for s in self.sections)

    def nice_to_have_in_jd_order(self) -> tuple[str, ...]:
        """``nice_to_have`` in the order the JD first mentions them (nice-to-have section first)."""
        return self._nice_to_have_order

    @cached_property
    def _nice_to_have_order(self) -> tuple[str, ...]:
        wanted = set(self.nice_to_have)
        order: list[str] = []
        ordered = sorted(self.sections, key=lambda s: s.kind is not _K.NICE_TO_HAVE)  # stable
        for section in ordered:
            for mention in find_mentions(section.text):
                if mention.canonical in wanted and mention.canonical not in order:
                    order.append(mention.canonical)
        return tuple(order + [s for s in self.nice_to_have if s not in order])

    def to_dict(self, include_text: bool = False) -> dict:
        """JSON form; the section lines and full text only with ``include_text``."""
        data = {
            "title": self.title,
            "must_have": [r.to_dict() for r in self.must_have],
            "min_total_years": self.min_total_years,
            "nice_to_have": list(self.nice_to_have),
            "critical_skills": list(self.critical_skills),
            "unparsed_requirements": list(self.unparsed_requirements),
            "warnings": list(self.warnings),
            "sections": [
                {"kind": s.kind.value, "heading": s.heading}
                | ({"lines": list(s.lines)} if include_text else {})
                for s in self.sections
            ],
            "query_text": self.query_text,
        }
        if include_text:
            data["full_text"] = self.full_text
        return data


# --- Sections and units -------------------------------------------------------------------


def heading_kind(line: str) -> JDSectionKind | None:
    """The section kind if ``line`` is a heading, else None."""
    text = " ".join(line.strip().removesuffix(":").split())
    return _KIND_BY_HEADING.get(text.casefold())


def _trim(lines: list[str]) -> tuple[str, ...]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return tuple(lines[start:end])


def split_sections(text: str) -> list[JDSection]:
    """The JD's sections in order, starting with the lines before the first heading."""
    sections: list[JDSection] = []
    kind, heading, body = _K.OTHER, None, []
    for line in text.splitlines():
        line = line.rstrip()
        found = heading_kind(line)
        if found:
            if heading is not None or _trim(body):
                sections.append(JDSection(kind, heading, _trim(body)))
            kind, heading, body = found, line.strip(), []
        else:
            body.append(line)
    if heading is not None or _trim(body):
        sections.append(JDSection(kind, heading, _trim(body)))
    return sections


def split_sentences(text: str) -> list[str]:
    """Sentences of a prose paragraph."""
    return [s.strip() for s in _SENTENCE_END.split(" ".join(text.split())) if s.strip()]


def units(lines: Iterable[str]) -> list[str]:
    """Requirement units: one per bullet (with any wrapped lines), prose split into sentences."""
    out: list[str] = []
    bullet: list[str] | None = None
    prose: list[str] = []

    def flush_prose() -> None:
        if prose:
            out.extend(split_sentences(" ".join(prose)))
            prose.clear()

    def flush_bullet() -> None:
        nonlocal bullet
        if bullet:
            out.append(" ".join(" ".join(bullet).split()))
        bullet = None

    for line in lines:
        if not line.strip():
            flush_bullet()
            flush_prose()
        elif _BULLET.match(line):
            flush_bullet()
            flush_prose()
            bullet = [_BULLET.sub("", line, count=1)]
        elif bullet is not None:
            bullet.append(line.strip())  # a bullet wrapped onto the next line
        else:
            prose.append(line.strip())
    flush_bullet()
    flush_prose()
    return out


# --- Units -> requirements ----------------------------------------------------------------


def years_in(text: str) -> list[int]:
    """The years figures of every years phrase in ``text``, in order."""
    values = []
    for match in _YEARS.finditer(text):
        number = match.group("n").casefold()
        values.append(int(number) if number.isdigit() else _NUMBER_WORDS[number])
    return values


def _distinct(names: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def skill_groups(text: str, mentions: list[Mention] | None = None) -> list[list[str]]:
    """The unit's skills grouped: a run joined by ``or`` / ``/`` is one group, else singletons.

    ``A, B or C`` is one group; ``A, B and C`` and ``A and B`` are three and two groups.
    """
    mentions = find_mentions(text) if mentions is None else mentions
    runs: list[list[Mention]] = []
    has_or: list[bool] = []
    for mention in mentions:
        if runs:
            between = text[runs[-1][-1].span[1] : mention.span[0]]
            if _OR_SEPARATOR.fullmatch(between):
                runs[-1].append(mention)
                has_or[-1] = True
                continue
            if _LIST_SEPARATOR.fullmatch(between):
                runs[-1].append(mention)
                continue
        runs.append([mention])
        has_or.append(False)
    groups: list[list[str]] = []
    for run, is_or in zip(runs, has_or, strict=True):
        names = _distinct(m.canonical for m in run)
        groups.extend([names] if is_or else [[n] for n in names])
    return groups


def _merge(requirements: list[Requirement]) -> list[Requirement]:
    """One requirement per skill set, keeping the first source and the largest min_years."""
    merged: dict[frozenset[str], Requirement] = {}
    for req in requirements:
        key = frozenset(req.any_of)
        if key in merged:
            old = merged[key]
            years = [y for y in (old.min_years, req.min_years) if y is not None]
            merged[key] = Requirement(old.any_of, max(years) if years else None, old.source_text)
        else:
            merged[key] = req
    return list(merged.values())


def parse_job_description(text: str) -> JobRequirements:
    """Parse JD text into ``JobRequirements``."""
    sections = split_sections(text)
    title = next((line.strip() for line in text.splitlines() if line.strip()), "")
    warnings: list[str] = []
    by_kind = {kind: [s for s in sections if s.kind is kind] for kind in JDSectionKind}
    if not any(s.heading for s in sections):
        warnings.append("no section headings found")
    for kind in SOURCE_SECTIONS:
        if len(by_kind[kind]) > 1:
            warnings.append(f"{len(by_kind[kind])} {kind.value} sections; merged")

    must_have: list[Requirement] = []
    total_years: list[int] = []
    unparsed: list[str] = []
    nice: list[str] = []

    if by_kind[_K.REQUIREMENTS]:
        for section in by_kind[_K.REQUIREMENTS]:
            for unit in units(section.lines):
                groups = skill_groups(unit)
                years = years_in(unit)
                if len(set(years)) > 1:
                    warnings.append(f"several year figures in {unit!r}; using {max(years)}")
                min_years = max(years) if years else None
                if groups:
                    must_have.extend(Requirement(tuple(g), min_years, unit) for g in groups)
                elif min_years is not None:
                    total_years.append(min_years)
                else:
                    unparsed.append(unit)
        if not must_have and not total_years:
            warnings.append("requirements section names no known skill or years")
        nice_sources = by_kind[_K.NICE_TO_HAVE]
    else:
        warnings.append(NO_REQUIREMENTS_WARNING)
        nice_sources = [s for s in sections if s.kind not in (_K.ABOUT, _K.BENEFITS)]

    for section in nice_sources:
        for unit in units(section.lines):
            skills = [m.canonical for m in find_mentions(unit)]
            if not skills and section.kind is _K.NICE_TO_HAVE:
                warnings.append(f"nice-to-have names no known skill: {unit!r}")
            nice.extend(skills)

    must_have = _merge(must_have)
    critical = _distinct(skill for req in must_have for skill in req.any_of)
    nice_to_have = sorted(set(nice) - set(critical), key=str.casefold)

    query_parts = [title] + [s.text for s in sections if s.kind in QUERY_SECTIONS and s.lines]
    query_text = "\n\n".join(query_parts) if len(query_parts) > 1 else text.strip()
    return JobRequirements(
        title=title,
        sections=tuple(sections),
        must_have=tuple(must_have),
        min_total_years=max(total_years) if total_years else None,
        nice_to_have=tuple(nice_to_have),
        unparsed_requirements=tuple(unparsed),
        warnings=tuple(warnings),
        query_text=query_text,
        full_text=text,
        critical_skills=tuple(critical),
    )


# --- Loading ------------------------------------------------------------------------------


def _looks_like_path(source: str) -> bool:
    if "\n" in source or len(source) > 1024:
        return False
    stripped = source.strip()
    if PurePosixPath(stripped.replace("\\", "/")).suffix.lower() in JD_EXTENSIONS:
        return True
    return " " not in stripped and ("/" in stripped or "\\" in stripped)


def load_job_description(source: str) -> tuple[str, str]:
    """``(text, origin)`` for a JD given as a workspace path or as literal text.

    A single line ending in ``.txt``, ``.pdf`` or ``.docx`` (or a space-free line with a path
    separator) is read through ``fs_tools.read_file``, relative to ``WORKSPACE_ROOT``, and the
    origin is the path. Anything else is the JD itself, with origin ``"<text>"``. Raises
    ``JobDescriptionError`` if the file can't be read or has no text.
    """
    if not source or not source.strip():
        raise JobDescriptionError("the job description is empty")
    if not _looks_like_path(source):
        return source, "<text>"
    path = source.strip()
    result = fs_tools.read_file(path)
    if not result["success"]:
        error = result["error"]
        raise JobDescriptionError(f"{error['code']}: {error['message']}")
    text = result["data"]["content"]
    if not text.strip():
        raise JobDescriptionError(f"{path}: no text extracted")
    return text, path
