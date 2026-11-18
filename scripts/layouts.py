"""Resume layouts: map a persona to a list of document blocks (dataset tooling).

One function per layout in ``data/ground_truth/persona_plan.md``. Every layout is a fixed
``LayoutStyle`` (heading case, colon, TXT underline, date separator, ongoing-role word, role
header style, skills display, section order) applied by one shared builder; ``sidebar`` adds a
shaded panel. The rendering contract these produce is documented under "Rendering variants" in
``data/ground_truth/README.md``. Bullets, summaries and descriptions are copied verbatim.
"""

from collections.abc import Callable
from dataclasses import dataclass

from rag.skills import Category, get_skill
from scripts.persona_schema import HEADER_VARIANTS, PRESENT, DateFormat, Persona, Role
from scripts.resume_doc import (
    Block,
    BulletList,
    Heading,
    Panel,
    Paragraph,
    RoleHeader,
    RoleStyle,
    Spacer,
    Table,
)

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Display names for skill groups. None of them may itself be a skill mention.
SKILL_GROUPS: dict[Category, str] = {
    Category.LANGUAGE: "Languages",
    Category.BACKEND: "Backend",
    Category.FRONTEND: "Frontend",
    Category.MOBILE: "Mobile",
    Category.DATA: "Data",
    Category.ML: "AI & Modelling",
    Category.DATABASE: "Databases",
    Category.CLOUD: "Cloud",
    Category.DEVOPS: "DevOps & Tooling",
    Category.TESTING: "Testing",
    Category.PRACTICE: "Practices",
    Category.BUSINESS: "Business Tools",
}

SECTIONS = (
    "summary",
    "skills",
    "experience",
    "projects",
    "education",
    "coursework",
    "certifications",
    "hobbies",
)


@dataclass(frozen=True)
class LayoutStyle:
    caps: bool  # ALL CAPS headings; otherwise the variant's Title Case
    colon: bool  # trailing ":" on headings
    underline: str | None  # TXT underline character
    styled_headings: bool  # DOCX Heading styles; otherwise bold paragraphs
    separator: str  # between the two ends of a date range
    ongoing: str  # end of a current role
    role_style: RoleStyle
    bullet: str
    skills: str  # "list" (one comma list), "grouped" (one line per category) or "table"
    sections: tuple[str, ...]  # main-column order
    panel: tuple[str, ...] = ()  # sections in the sidebar panel, before the main column
    headline: bool = True


STYLES: dict[str, LayoutStyle] = {
    "classic": LayoutStyle(
        caps=True,
        colon=False,
        underline=None,
        styled_headings=True,
        separator=" \N{EN DASH} ",
        ongoing="Present",
        role_style=RoleStyle.STACKED,
        bullet="-",
        skills="list",
        sections=(
            "summary",
            "experience",
            "skills",
            "education",
            "coursework",
            "projects",
            "certifications",
            "hobbies",
        ),
    ),
    "modern": LayoutStyle(
        caps=False,
        colon=False,
        underline=None,
        styled_headings=True,
        separator=" - ",
        ongoing="Present",
        role_style=RoleStyle.STACKED,
        bullet="-",
        skills="grouped",
        sections=(
            "summary",
            "skills",
            "experience",
            "projects",
            "education",
            "coursework",
            "certifications",
            "hobbies",
        ),
    ),
    "compact": LayoutStyle(
        caps=True,
        colon=True,
        underline=None,
        styled_headings=False,
        separator=" \N{EN DASH} ",
        ongoing="Current",
        role_style=RoleStyle.INLINE,
        bullet="-",
        skills="list",
        sections=(
            "skills",
            "summary",
            "experience",
            "education",
            "coursework",
            "projects",
            "certifications",
            "hobbies",
        ),
    ),
    "sidebar": LayoutStyle(
        caps=False,
        colon=False,
        underline=None,
        styled_headings=True,
        separator=" to ",
        ongoing="Present",
        role_style=RoleStyle.STACKED,
        bullet="-",
        skills="list",
        panel=("skills", "education", "coursework"),
        sections=("summary", "experience", "projects", "certifications", "hobbies"),
    ),
    "table-skills": LayoutStyle(
        caps=True,
        colon=False,
        underline=None,
        styled_headings=True,
        separator=" - ",
        ongoing="Current",
        role_style=RoleStyle.STACKED,
        bullet="-",
        skills="table",
        sections=(
            "summary",
            "skills",
            "experience",
            "education",
            "coursework",
            "projects",
            "certifications",
            "hobbies",
        ),
    ),
    "academic": LayoutStyle(
        caps=False,
        colon=True,
        underline=None,
        styled_headings=True,
        separator=" to ",
        ongoing="present",
        role_style=RoleStyle.STACKED,
        bullet="-",
        skills="grouped",
        sections=(
            "education",
            "coursework",
            "summary",
            "projects",
            "experience",
            "skills",
            "certifications",
            "hobbies",
        ),
    ),
    "plain": LayoutStyle(
        caps=True,
        colon=False,
        underline="=",
        styled_headings=False,
        separator=" - ",
        ongoing="present",
        role_style=RoleStyle.STACKED,
        bullet="-",
        skills="list",
        sections=(
            "summary",
            "skills",
            "experience",
            "projects",
            "education",
            "coursework",
            "certifications",
            "hobbies",
        ),
    ),
    "minimal": LayoutStyle(
        caps=False,
        colon=False,
        underline="-",
        styled_headings=False,
        separator=" to ",
        ongoing="Current",
        role_style=RoleStyle.INLINE,
        bullet="*",
        skills="list",
        headline=False,
        sections=(
            "summary",
            "experience",
            "skills",
            "education",
            "coursework",
            "projects",
            "certifications",
            "hobbies",
        ),
    ),
}


# --- Text helpers -------------------------------------------------------------------------


def heading_text(section: str, persona: Persona) -> str:
    """The section heading exactly as ``persona``'s layout renders it."""
    style = STYLES[persona.render.layout]
    text = HEADER_VARIANTS[persona.render.header_variant][section]
    if style.caps:
        text = text.upper()
    return f"{text}:" if style.colon else text


def format_month(year_month: str, date_format: DateFormat) -> str:
    year, month = year_month.split("-")
    match date_format:
        case DateFormat.MONTH_NAME:
            return f"{MONTHS[int(month) - 1]} {year}"
        case DateFormat.ISO:
            return year_month
        case DateFormat.NUMERIC:
            return f"{month}/{year}"
        case DateFormat.YEAR_ONLY:
            return year


def date_range(role: Role, persona: Persona) -> str:
    """A role's dates exactly as ``persona``'s layout renders them."""
    style = STYLES[persona.render.layout]
    fmt = persona.render.date_format
    end = style.ongoing if role.end == PRESENT else format_month(role.end, fmt)
    return f"{format_month(role.start, fmt)}{style.separator}{end}"


def skill_label(persona: Persona, skill: str) -> str:
    return persona.render.skill_labels.get(skill, skill)


def present_sections(persona: Persona) -> set[str]:
    present = {
        "summary": bool(persona.summary),
        "skills": bool(persona.skills_section),
        "experience": bool(persona.experience),
        "projects": bool(persona.projects),
        "education": bool(persona.education),
        "coursework": any(e.coursework for e in persona.education),
        "certifications": bool(persona.certifications),
        "hobbies": bool(persona.hobbies),
    }
    return {section for section, shown in present.items() if shown}


def section_order(persona: Persona) -> list[str]:
    """The sections ``persona``'s resume shows, in rendered order."""
    style = STYLES[persona.render.layout]
    present = present_sections(persona)
    return [s for s in (*style.panel, *style.sections) if s in present]


def expected_headings(persona: Persona) -> list[str]:
    return [heading_text(section, persona) for section in section_order(persona)]


# --- Section builders ---------------------------------------------------------------------


def _skill_groups(persona: Persona) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for skill in persona.skills_section:
        group = SKILL_GROUPS[get_skill(skill).category]
        groups.setdefault(group, []).append(skill_label(persona, skill))
    return list(groups.items())


def _skills(persona: Persona, style: LayoutStyle) -> list[Block]:
    match style.skills:
        case "table":
            return [
                Table(tuple((group, ", ".join(items)) for group, items in _skill_groups(persona)))
            ]
        case "grouped":
            return [
                Paragraph(f"{group}: {', '.join(items)}") for group, items in _skill_groups(persona)
            ]
        case _:
            return [Paragraph(", ".join(skill_label(persona, s) for s in persona.skills_section))]


def _experience(persona: Persona, style: LayoutStyle) -> list[Block]:
    blocks: list[Block] = []
    for index, role in enumerate(persona.experience):
        if index:
            blocks.append(Spacer())
        blocks.append(
            RoleHeader(role.title, role.company, date_range(role, persona), style=style.role_style)
        )
        blocks.append(BulletList(role.bullets, marker=style.bullet))
    return blocks


def _education(persona: Persona, style: LayoutStyle) -> list[Block]:
    blocks: list[Block] = []
    for edu in persona.education:
        details = f"{edu.institution}, {edu.start_year}{style.separator}{edu.end_year}"
        if edu.grade:
            details += f" | Grade: {edu.grade}"
        blocks += [Paragraph(f"{edu.degree} in {edu.field}", bold=True), Paragraph(details)]
    return blocks


def _coursework(persona: Persona, style: LayoutStyle) -> list[Block]:
    return [
        Paragraph(f"{edu.degree} in {edu.field}: {', '.join(edu.coursework)}")
        for edu in persona.education
        if edu.coursework
    ]


def _projects(persona: Persona, style: LayoutStyle) -> list[Block]:
    blocks: list[Block] = []
    for project in persona.projects:
        blocks += [Paragraph(project.name, bold=True), Paragraph(project.description)]
    return blocks


_BUILDERS: dict[str, Callable[[Persona, LayoutStyle], list[Block]]] = {
    "summary": lambda p, s: [Paragraph(p.summary)],
    "skills": _skills,
    "experience": _experience,
    "projects": _projects,
    "education": _education,
    "coursework": _coursework,
    "certifications": lambda p, s: [BulletList(p.certifications, marker=s.bullet)],
    "hobbies": lambda p, s: [Paragraph(", ".join(p.hobbies))],
}


def _section(section: str, persona: Persona, style: LayoutStyle) -> list[Block]:
    heading = Heading(
        heading_text(section, persona), underline=style.underline, styled=style.styled_headings
    )
    return [heading, *_BUILDERS[section](persona, style)]


def build(persona: Persona) -> list[Block]:
    """The blocks for ``persona`` in its own layout (``render.layout``)."""
    style = STYLES[persona.render.layout]
    present = present_sections(persona)
    blocks: list[Block] = [Heading(persona.name, level=0, styled=style.styled_headings)]
    if style.headline:
        blocks.append(Paragraph(persona.headline))
    if style.panel:
        panel: list[Block] = [
            Paragraph(persona.email),
            Paragraph(persona.phone),
            Paragraph(persona.location),
        ]
        for section in style.panel:
            if section in present:
                panel += _section(section, persona, style)
        blocks.append(Panel(tuple(panel)))
    else:
        blocks.append(Paragraph(f"{persona.email} | {persona.phone} | {persona.location}"))
    for section in style.sections:
        if section in present:
            blocks += _section(section, persona, style)
    return blocks


def _check(persona: Persona, layout: str) -> Persona:
    if persona.render.layout != layout:
        raise ValueError(f"{persona.id} uses layout {persona.render.layout!r}, not {layout!r}")
    return persona


def classic(persona: Persona) -> list[Block]:
    """Blocks for a persona in the ``classic`` layout."""
    return build(_check(persona, "classic"))


def modern(persona: Persona) -> list[Block]:
    """Blocks for a persona in the ``modern`` layout."""
    return build(_check(persona, "modern"))


def compact(persona: Persona) -> list[Block]:
    """Blocks for a persona in the ``compact`` layout."""
    return build(_check(persona, "compact"))


def sidebar(persona: Persona) -> list[Block]:
    """Blocks for a persona in the ``sidebar`` layout."""
    return build(_check(persona, "sidebar"))


def table_skills(persona: Persona) -> list[Block]:
    """Blocks for a persona in the ``table-skills`` layout."""
    return build(_check(persona, "table-skills"))


def academic(persona: Persona) -> list[Block]:
    """Blocks for a persona in the ``academic`` layout."""
    return build(_check(persona, "academic"))


def plain(persona: Persona) -> list[Block]:
    """Blocks for a persona in the ``plain`` layout."""
    return build(_check(persona, "plain"))


def minimal(persona: Persona) -> list[Block]:
    """Blocks for a persona in the ``minimal`` layout."""
    return build(_check(persona, "minimal"))


LAYOUT_FUNCTIONS: dict[str, Callable[[Persona], list[Block]]] = {
    "classic": classic,
    "modern": modern,
    "compact": compact,
    "sidebar": sidebar,
    "table-skills": table_skills,
    "academic": academic,
    "plain": plain,
    "minimal": minimal,
}
