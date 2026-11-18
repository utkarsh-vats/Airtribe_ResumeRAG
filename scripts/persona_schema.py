"""Persona schema for the synthetic resume dataset (dataset tooling, not runtime code).

One JSON file per persona lives in ``data/ground_truth/personas/<id>.json``. Personas are the
single source of truth: the resume generator (step 3) renders them and
``scripts/build_manifest.py`` derives the ground-truth manifest from them.

``load_personas`` checks structure only (keys, types, enum values, date syntax). Semantic rules
(canonical skills, chronology, the year_only rule, bullets mentioning skills) are enforced by
``tests/test_personas.py``. Conventions are documented in ``data/ground_truth/README.md``.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_DIR = ROOT / "data" / "ground_truth"
PERSONA_DIR = GROUND_TRUTH_DIR / "personas"

# Reference date for every ground-truth computation ("present" = this month).
AS_OF = date(2026, 11, 1)
PRESENT = "present"


class EmploymentType(StrEnum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"


class ResumeFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class DateFormat(StrEnum):
    MONTH_NAME = "month_name"  # Jan 2020 - Present
    ISO = "iso"  # 2020-01 - present
    NUMERIC = "numeric"  # 01/2020 - Present
    YEAR_ONLY = "year_only"  # 2020 - Present


class DegreeLevel(IntEnum):
    SCHOOL = 1
    DIPLOMA = 2
    BACHELORS = 3
    MASTERS = 4
    PHD = 5


DEGREE_LEVELS: dict[str, DegreeLevel] = {
    "High School": DegreeLevel.SCHOOL,
    "Diploma": DegreeLevel.DIPLOMA,
    "B.A.": DegreeLevel.BACHELORS,
    "B.S.": DegreeLevel.BACHELORS,
    "B.Sc.": DegreeLevel.BACHELORS,
    "B.Eng.": DegreeLevel.BACHELORS,
    "B.Tech": DegreeLevel.BACHELORS,
    "BCA": DegreeLevel.BACHELORS,
    "M.A.": DegreeLevel.MASTERS,
    "M.S.": DegreeLevel.MASTERS,
    "M.Sc.": DegreeLevel.MASTERS,
    "M.Eng.": DegreeLevel.MASTERS,
    "M.Tech": DegreeLevel.MASTERS,
    "MBA": DegreeLevel.MASTERS,
    "MCA": DegreeLevel.MASTERS,
    "PhD": DegreeLevel.PHD,
}

# Layouts the resume generator (step 3) implements. "plain" is TXT-only, "table-skills" is
# DOCX-only; TXT resumes use "plain" or "minimal".
LAYOUTS: dict[str, str] = {
    "classic": "Single column; name, contact line, caps headings, bulleted roles.",
    "modern": "Name banner + headline, ruled headings, skills grouped by category inline.",
    "compact": "Dense; role, company and dates on one line, skills as one comma list.",
    "sidebar": "Two-column design linearised: contact, skills, education first, then roles.",
    "table-skills": "Classic, but the Skills section is a DOCX table (category | skills).",
    "academic": "Education first, coursework and projects prominent; long-form.",
    "plain": "Plain text, headings underlined with '=' or '-', '-' bullets.",
    "minimal": "Name, contact, experience and education only; no summary or extras.",
}
FORMAT_LAYOUTS: dict[ResumeFormat, frozenset[str]] = {
    ResumeFormat.PDF: frozenset(LAYOUTS) - {"plain", "table-skills"},
    ResumeFormat.DOCX: frozenset(LAYOUTS) - {"plain"},
    ResumeFormat.TXT: frozenset({"plain", "minimal"}),
}

# Section heading synonyms per variant; the section detector must recognise all of them.
HEADER_VARIANTS: dict[str, dict[str, str]] = {
    "standard": {
        "summary": "Summary",
        "experience": "Experience",
        "skills": "Skills",
        "education": "Education",
        "projects": "Projects",
        "certifications": "Certifications",
        "hobbies": "Interests",
    },
    "professional": {
        "summary": "Professional Summary",
        "experience": "Professional Experience",
        "skills": "Technical Skills",
        "education": "Education",
        "projects": "Selected Projects",
        "certifications": "Certifications",
        "hobbies": "Hobbies",
    },
    "history": {
        "summary": "Profile",
        "experience": "Work History",
        "skills": "Core Competencies",
        "education": "Academic Background",
        "projects": "Key Projects",
        "certifications": "Licenses & Certifications",
        "hobbies": "Interests",
    },
    "employment": {
        "summary": "About Me",
        "experience": "Employment",
        "skills": "Skills & Tools",
        "education": "Qualifications",
        "projects": "Personal Projects",
        "certifications": "Courses & Certifications",
        "hobbies": "Outside Work",
    },
}

EDGE_CASES: dict[str, str] = {
    "two_page": "Long resume that must render onto two pages.",
    "minimal": "Very short resume (minimal layout).",
    "skills_table": "Skills rendered as a DOCX table (layout table-skills).",
    "no_skills_section": "No Skills section; skills appear only in experience.",
    "aliases_only": "Skills written only as aliases (k8s, Postgres, JS, ML).",
    "lowercase_skill": "A skill written in lowercase (render.skill_labels).",
    "implied_skill_evidence": "A skill evidenced in roles only through implications.",
    "python_in_hobbies": "Python mentioned only in hobbies (not a skill).",
    "js_for_java_jd": "JavaScript-heavy candidate aimed at the Java JD.",
    "java_for_frontend_jd": "Java candidate aimed at the frontend JD.",
    "overlapping_roles": "Concurrent roles; years must be merged, not summed.",
    "career_gap": "Multi-year gap between roles.",
    "career_changer": "Non-tech years followed by tech years.",
    "coursework_only_skill": "A skill appearing only in coursework (not a skill).",
    "listed_never_used": "Skill listed but used in no role (0 evidenced years).",
    "duplicate_name": "Shares a name with another persona (id suffix _2).",
    "internships_only": "Fresh graduate whose only roles are internships.",
    "non_tech": "Non-technical persona, irrelevant to every JD.",
    "ambiguous_terms": "Common words that are also skill names (airflow, Spark as a company).",
}

_YEAR_MONTH = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


class PersonaError(ValueError):
    """A persona file does not match the schema."""


@dataclass(frozen=True)
class Role:
    title: str
    company: str
    employment_type: EmploymentType
    start: str  # "YYYY-MM"
    end: str  # "YYYY-MM" or "present"
    skills_used: tuple[str, ...]
    bullets: tuple[str, ...]


@dataclass(frozen=True)
class Education:
    degree: str
    field: str
    institution: str
    start_year: int
    end_year: int
    grade: str | None
    coursework: tuple[str, ...] = ()

    @property
    def level(self) -> DegreeLevel:
        return DEGREE_LEVELS[self.degree]


@dataclass(frozen=True)
class Project:
    name: str
    skills_used: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class Render:
    format: ResumeFormat
    layout: str
    header_variant: str
    date_format: DateFormat
    # Display text for a skill in the Skills section (canonical name -> label), e.g. a
    # lowercase "python". Each label must canonicalize back to its key.
    skill_labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    email: str
    phone: str
    location: str
    headline: str
    summary: str
    skills_section: tuple[str, ...]
    experience: tuple[Role, ...]
    education: tuple[Education, ...]
    render: Render
    projects: tuple[Project, ...] = ()
    certifications: tuple[str, ...] = ()
    hobbies: tuple[str, ...] = ()
    edge_cases: tuple[str, ...] = ()


# --- Parsing ------------------------------------------------------------------------------


def _object(
    value: Any, where: str, required: set[str], optional: frozenset[str] | set[str] = frozenset()
) -> dict:
    if not isinstance(value, dict):
        raise PersonaError(f"{where}: expected an object")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise PersonaError(f"{where}: missing keys {sorted(missing)}")
    if unknown:
        raise PersonaError(f"{where}: unknown keys {sorted(unknown)}")
    return value


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PersonaError(f"{where}: expected a non-empty string")
    return value


def _int(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PersonaError(f"{where}: expected an integer")
    return value


def _strs(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise PersonaError(f"{where}: expected a list")
    return tuple(_str(item, f"{where}[{i}]") for i, item in enumerate(value))


def _enum[E: StrEnum](enum: type[E], value: Any, where: str) -> E:
    try:
        return enum(value)
    except ValueError:
        allowed = ", ".join(member.value for member in enum)
        raise PersonaError(f"{where}: {value!r} is not one of {allowed}") from None


def _month(value: Any, where: str, allow_present: bool) -> str:
    text = _str(value, where)
    if (allow_present and text == PRESENT) or _YEAR_MONTH.match(text):
        return text
    expected = "'YYYY-MM' or 'present'" if allow_present else "'YYYY-MM'"
    raise PersonaError(f"{where}: expected {expected}, got {text!r}")


def _choice(value: Any, choices: Any, where: str) -> str:
    text = _str(value, where)
    if text not in choices:
        raise PersonaError(f"{where}: {text!r} is not one of {', '.join(sorted(choices))}")
    return text


def _role(value: Any, where: str) -> Role:
    keys = {"title", "company", "employment_type", "start", "end", "skills_used", "bullets"}
    obj = _object(value, where, keys)
    return Role(
        title=_str(obj["title"], f"{where}.title"),
        company=_str(obj["company"], f"{where}.company"),
        employment_type=_enum(EmploymentType, obj["employment_type"], f"{where}.employment_type"),
        start=_month(obj["start"], f"{where}.start", allow_present=False),
        end=_month(obj["end"], f"{where}.end", allow_present=True),
        skills_used=_strs(obj["skills_used"], f"{where}.skills_used"),
        bullets=_strs(obj["bullets"], f"{where}.bullets"),
    )


def _education(value: Any, where: str) -> Education:
    keys = {"degree", "field", "institution", "start_year", "end_year", "grade"}
    obj = _object(value, where, keys, {"coursework"})
    grade = obj["grade"]
    return Education(
        degree=_choice(obj["degree"], DEGREE_LEVELS, f"{where}.degree"),
        field=_str(obj["field"], f"{where}.field"),
        institution=_str(obj["institution"], f"{where}.institution"),
        start_year=_int(obj["start_year"], f"{where}.start_year"),
        end_year=_int(obj["end_year"], f"{where}.end_year"),
        grade=None if grade is None else _str(grade, f"{where}.grade"),
        coursework=_strs(obj.get("coursework", []), f"{where}.coursework"),
    )


def _project(value: Any, where: str) -> Project:
    obj = _object(value, where, {"name", "skills_used", "description"})
    return Project(
        name=_str(obj["name"], f"{where}.name"),
        skills_used=_strs(obj["skills_used"], f"{where}.skills_used"),
        description=_str(obj["description"], f"{where}.description"),
    )


def _render(value: Any, where: str) -> Render:
    keys = {"format", "layout", "header_variant", "date_format"}
    obj = _object(value, where, keys, {"skill_labels"})
    labels = obj.get("skill_labels", {})
    if not isinstance(labels, dict):
        raise PersonaError(f"{where}.skill_labels: expected an object")
    return Render(
        format=_enum(ResumeFormat, obj["format"], f"{where}.format"),
        layout=_choice(obj["layout"], LAYOUTS, f"{where}.layout"),
        header_variant=_choice(obj["header_variant"], HEADER_VARIANTS, f"{where}.header_variant"),
        date_format=_enum(DateFormat, obj["date_format"], f"{where}.date_format"),
        skill_labels={
            _str(k, f"{where}.skill_labels"): _str(v, f"{where}.skill_labels.{k}")
            for k, v in labels.items()
        },
    )


def _list(value: Any, where: str) -> list:
    if not isinstance(value, list):
        raise PersonaError(f"{where}: expected a list")
    return value


def persona_from_dict(data: Any, where: str = "persona") -> Persona:
    """Build a ``Persona`` from parsed JSON. Raises ``PersonaError`` on a schema violation."""
    required = {
        "id",
        "name",
        "email",
        "phone",
        "location",
        "headline",
        "summary",
        "skills_section",
        "experience",
        "education",
        "render",
    }
    optional = {"projects", "certifications", "hobbies", "edge_cases"}
    obj = _object(data, where, required, optional)
    edge_cases = _strs(obj.get("edge_cases", []), f"{where}.edge_cases")
    for tag in edge_cases:
        _choice(tag, EDGE_CASES, f"{where}.edge_cases")
    return Persona(
        id=_str(obj["id"], f"{where}.id"),
        name=_str(obj["name"], f"{where}.name"),
        email=_str(obj["email"], f"{where}.email"),
        phone=_str(obj["phone"], f"{where}.phone"),
        location=_str(obj["location"], f"{where}.location"),
        headline=_str(obj["headline"], f"{where}.headline"),
        summary=_str(obj["summary"], f"{where}.summary"),
        skills_section=_strs(obj["skills_section"], f"{where}.skills_section"),
        experience=tuple(
            _role(item, f"{where}.experience[{i}]")
            for i, item in enumerate(_list(obj["experience"], f"{where}.experience"))
        ),
        education=tuple(
            _education(item, f"{where}.education[{i}]")
            for i, item in enumerate(_list(obj["education"], f"{where}.education"))
        ),
        render=_render(obj["render"], f"{where}.render"),
        projects=tuple(
            _project(item, f"{where}.projects[{i}]")
            for i, item in enumerate(_list(obj.get("projects", []), f"{where}.projects"))
        ),
        certifications=_strs(obj.get("certifications", []), f"{where}.certifications"),
        hobbies=_strs(obj.get("hobbies", []), f"{where}.hobbies"),
        edge_cases=edge_cases,
    )


def load_persona(path: Path) -> Persona:
    """Load and structurally validate one persona JSON file."""
    with path.open(encoding="utf-8") as handle:
        return persona_from_dict(json.load(handle), where=path.name)


def load_personas(directory: Path = PERSONA_DIR) -> list[Persona]:
    """Load every ``*.json`` persona in ``directory``, sorted by id."""
    return sorted(
        (load_persona(path) for path in sorted(directory.glob("*.json"))), key=lambda p: p.id
    )
