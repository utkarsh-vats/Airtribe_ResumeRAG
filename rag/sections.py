"""Section detection and per-role splitting for resume text.

``parse_resume`` cleans extracted text (``rag.text``), finds section headings and returns the
header block plus an ordered list of sections. ``split_roles`` splits an experience section
into roles.

Heading detection: a whole line equal to a known heading (case-insensitive, one optional
trailing colon). As a fallback, a short ALL-CAPS line (at most four words, letters only) that
isn't a known heading starts an ``other`` section, so resumes with unfamiliar headings still
split sensibly.

Role headers come in several forms (see ``ROLE_HEADER_FORMS``). Each form is a small function
that looks at the lines from a position and either returns the role's title, company and dates
plus how many lines the header used, or None. Adding a form means adding one function.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from rag.dates import find_date_range, is_date_range
from rag.text import clean_lines


class SectionKind(StrEnum):
    """What a section of a resume holds."""

    HEADER = "header"  # name / contact block before the first heading
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    SKILLS = "skills"
    EDUCATION = "education"
    COURSEWORK = "coursework"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    HOBBIES = "hobbies"
    OTHER = "other"


# Every heading string the dataset's layouts render (data/ground_truth/README.md, "Rendering
# variants"), plus common real-world synonyms. Matching is case-insensitive.
HEADINGS: dict[SectionKind, tuple[str, ...]] = {
    SectionKind.SUMMARY: (
        "Summary",
        "Professional Summary",
        "Profile",
        "About Me",
        "Professional Profile",
        "Objective",
        "Career Objective",
    ),
    SectionKind.EXPERIENCE: (
        "Experience",
        "Professional Experience",
        "Work History",
        "Employment",
        "Work Experience",
        "Employment History",
        "Career History",
    ),
    SectionKind.SKILLS: (
        "Skills",
        "Technical Skills",
        "Core Competencies",
        "Skills & Tools",
        "Key Skills",
        "Tech Stack",
    ),
    SectionKind.EDUCATION: ("Education", "Academic Background", "Qualifications"),
    SectionKind.COURSEWORK: (
        "Coursework",
        "Relevant Coursework",
        "Key Coursework",
        "Study Modules",
    ),
    SectionKind.PROJECTS: ("Projects", "Selected Projects", "Key Projects", "Personal Projects"),
    SectionKind.CERTIFICATIONS: (
        "Certifications",
        "Licenses & Certifications",
        "Courses & Certifications",
        "Certificates",
    ),
    SectionKind.HOBBIES: ("Interests", "Hobbies", "Outside Work", "Hobbies & Interests"),
}

_KIND_BY_HEADING = {
    " ".join(h.split()).casefold(): kind for kind, hs in HEADINGS.items() for h in hs
}
_CAPS_HEADING = re.compile(r"[A-Z][A-Z&/ -]*")
_CONTACT = re.compile(r"@|\+?\d[\d ()-]{6,}\d")
HEADER_SCAN_LINES = 5


@dataclass(frozen=True)
class Section:
    """A section; ``start_line``/``end_line`` index ``ParsedResume.lines`` (end exclusive).

    ``start_line`` is the heading line, or the first content line when there is no heading.
    """

    kind: SectionKind
    heading: str | None
    lines: tuple[str, ...]
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ParsedResume:
    """A resume split into its header block and ordered sections."""

    lines: tuple[str, ...]  # all cleaned lines
    header: tuple[str, ...]
    sections: tuple[Section, ...] = field(default_factory=tuple)

    def of_kind(self, kind: SectionKind) -> list[Section]:
        """The sections of ``kind``, in document order."""
        return [s for s in self.sections if s.kind is kind]


@dataclass(frozen=True)
class Role:
    """One role in an experience section."""

    title: str
    company: str
    date_text: str
    lines: tuple[str, ...]  # the role's body (bullets), without its header lines


# --- Headings -----------------------------------------------------------------------------


def heading_kind(line: str) -> SectionKind | None:
    """The section kind if ``line`` is a heading, else None."""
    text = " ".join(line.strip().removesuffix(":").split())
    known = _KIND_BY_HEADING.get(text.casefold())
    if known:
        return known
    if _CAPS_HEADING.fullmatch(text) and len(text.split()) <= 4 and len(text) > 1:
        return SectionKind.OTHER
    return None


def _header_length(lines: Sequence[str]) -> int:
    """For a document without headings: the name line plus any contact lines just after it."""
    if not lines:
        return 0
    contact = [i for i, line in enumerate(lines[:HEADER_SCAN_LINES]) if _CONTACT.search(line)]
    return contact[-1] + 1 if contact else 1


def parse_resume(text: str) -> ParsedResume:
    """Split extracted resume text into the header block and ordered sections."""
    lines = clean_lines(text)
    starts = [(i, kind) for i, line in enumerate(lines) if (kind := heading_kind(line))]
    if not starts:
        cut = _header_length(lines)
        body = tuple(lines[cut:])
        sections = (Section(SectionKind.OTHER, None, body, cut, len(lines)),) if body else ()
        return ParsedResume(tuple(lines), tuple(lines[:cut]), sections)
    sections = []
    for n, (start, kind) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        heading = lines[start]
        sections.append(Section(kind, heading, tuple(lines[start + 1 : end]), start, end))
    return ParsedResume(tuple(lines), tuple(lines[: starts[0][0]]), tuple(sections))


# --- Roles --------------------------------------------------------------------------------

type RoleHeader = tuple[str, str, str, int]  # title, company, date_text, lines consumed


def _parts(line: str) -> list[str]:
    return [part.strip() for part in line.split(" | ")]


def _inline(lines: Sequence[str], i: int) -> RoleHeader | None:
    """``Title | Company | dates`` on one line (inline layouts; modern DOCX after tabs)."""
    parts = _parts(lines[i])
    if len(parts) >= 3 and is_date_range(parts[-1]):
        return parts[0], " | ".join(parts[1:-1]), parts[-1], 1
    return None


def _stacked(lines: Sequence[str], i: int) -> RoleHeader | None:
    """``Title`` then ``Company | dates``."""
    if i + 1 >= len(lines) or " | " in lines[i] or find_date_range(lines[i]):
        return None
    parts = _parts(lines[i + 1])
    if len(parts) == 2 and is_date_range(parts[1]):
        return lines[i], parts[0], parts[1], 2
    return None


def _split(lines: Sequence[str], i: int) -> RoleHeader | None:
    """``Title | Company`` then a line with only the dates (modern layout in PDF)."""
    parts = _parts(lines[i])
    if i + 1 < len(lines) and len(parts) == 2 and is_date_range(lines[i + 1]):
        return parts[0], parts[1], lines[i + 1].strip(), 2
    return None


ROLE_HEADER_FORMS: tuple[Callable[[Sequence[str], int], RoleHeader | None], ...] = (
    _inline,
    _stacked,
    _split,
)


def role_header_at(lines: Sequence[str], i: int) -> RoleHeader | None:
    """The first role-header form that matches at ``lines[i]``, or None."""
    for form in ROLE_HEADER_FORMS:
        if found := form(lines, i):
            return found
    return None


def split_roles(section: Section) -> list[Role]:
    """The roles in an experience section, in order.

    Lines after a role header belong to that role until the next header. Lines before the
    first header (an intro sentence, say) belong to no role; they stay in ``section.lines``.
    """
    lines = section.lines
    roles: list[Role] = []
    current: tuple[str, str, str] | None = None
    body: list[str] = []
    i = 0
    while i < len(lines):
        found = role_header_at(lines, i)
        if found:
            if current:
                roles.append(Role(*current, tuple(body)))
            title, company, date_text, used = found
            current, body = (title, company, date_text), []
            i += used
            continue
        if current:
            body.append(lines[i])
        i += 1
    if current:
        roles.append(Role(*current, tuple(body)))
    return roles
