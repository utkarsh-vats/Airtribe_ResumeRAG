"""rag.text, rag.dates and rag.sections: unit tests plus the 35-resume corpus."""

import ast
import itertools
import json
import re

import pytest

import fs_tools
from rag.dates import find_date_range, is_date_range
from rag.sections import (
    HEADINGS,
    ROLE_HEADER_FORMS,
    Section,
    SectionKind,
    heading_kind,
    parse_resume,
    split_roles,
)
from rag.text import clean_lines
from scripts.build_manifest import MANIFEST_PATH
from scripts.layouts import STYLES, _skills, date_range, section_order
from scripts.persona_schema import GROUND_TRUTH_DIR, HEADER_VARIANTS, ROOT, Persona, load_personas

PERSONAS = load_personas()
BY_ID = {p.id: p for p in PERSONAS}
any_persona = pytest.mark.parametrize("persona", PERSONAS, ids=[p.id for p in PERSONAS])


def collapse(text: str) -> str:
    return " ".join(text.split())


# --- rag/ stays independent of dataset tooling ----------------------------------------------


def test_rag_never_imports_scripts_or_tests():
    # Test helpers (skill_mentions, jd_sections, ...) are importable by bare name too.
    forbidden = {"scripts", "tests"} | {p.stem for p in (ROOT / "tests").glob("*.py")}
    for path in (ROOT / "rag").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n.split(".")[0] in forbidden for n in names), (path, names)


def test_heading_vocabulary_covers_every_rendered_variant():
    for variant in HEADER_VARIANTS.values():
        for section, heading in variant.items():
            assert heading_kind(heading) is SectionKind(section), heading
    readme = (GROUND_TRUTH_DIR / "README.md").read_text(encoding="utf-8")
    table = readme.split("### Section headings", 1)[1].split("\n\n")[2]
    rows = [r for r in table.splitlines() if r.startswith("| ") and not r.startswith("| Section")]
    assert len(rows) == len(HEADER_VARIANTS["standard"])
    for row in rows:
        section, *headings = [c.strip() for c in row.strip("|").split("|")]
        for heading in headings:
            assert heading_kind(heading) is SectionKind(section), heading
            assert heading_kind(heading.upper() + ":") is SectionKind(section), heading


def test_vocabulary_has_no_duplicates():
    forms = [h.casefold() for hs in HEADINGS.values() for h in hs]
    assert len(forms) == len(set(forms))


# --- rag.text -----------------------------------------------------------------------------


def test_clean_lines_removes_rendering_artefacts():
    text = (
        "John Doe\n"
        "SKILLS\n======\n"
        "   \x7f Led the move\n"  # PDF bullet after fs_tools sanitising: spaces
        "\u2022 Built things\n"
        "\u2022Tight bullet\n"
        "- Dash bullet\n"
        "* Star bullet\n"
        "-5% churn stays\n"
        "Title | Company\tJan 2020 - Present\n"
        "Lots    of     spaces\n"
        "John Doe \u00b7 Page 1 of 2\n"
        "\n\f\n"
        "Experience\n----------\n"
    )
    assert clean_lines(text) == [
        "John Doe",
        "SKILLS",
        "\x7f Led the move",
        "Built things",
        "Tight bullet",
        "Dash bullet",
        "Star bullet",
        "-5% churn stays",
        "Title | Company | Jan 2020 - Present",
        "Lots of spaces",
        "Experience",
    ]


def test_clean_lines_on_sanitised_pdf_bullets():
    # fs_tools turns the DEL that pypdf extracts for a bullet into a space.
    assert clean_lines("   Led the move\n  Built it\n") == ["Led the move", "Built it"]


# --- Headings -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("EXPERIENCE", SectionKind.EXPERIENCE),
        ("Experience", SectionKind.EXPERIENCE),
        ("Work History:", SectionKind.EXPERIENCE),
        ("TECHNICAL SKILLS:", SectionKind.SKILLS),
        ("  skills & tools  ", SectionKind.SKILLS),
        ("Relevant Coursework", SectionKind.COURSEWORK),
        ("OUTSIDE WORK", SectionKind.HOBBIES),
        ("VOLUNTEERING", SectionKind.OTHER),  # unknown ALL-CAPS fallback
        ("AWARDS & HONOURS", SectionKind.OTHER),
        ("Volunteering", None),  # unknown and not caps
        ("AWS, SQL", None),  # punctuation: a list, not a heading
        ("SEE MORE OF MY WORK HERE", None),  # more than four words
        ("TOP 10", None),  # digits
        ("Led the platform team", None),
        ("Experience: 5 years", None),
    ],
)
def test_heading_kind(line, kind):
    assert heading_kind(line) is kind


def test_underlined_txt_headings_split():
    parsed = parse_resume("Zoe\nzoe@example.com\n\nSummary\n-------\nHi.\n\nSKILLS\n======\nGit\n")
    assert parsed.header == ("Zoe", "zoe@example.com")
    assert [(s.kind, s.heading, s.lines) for s in parsed.sections] == [
        (SectionKind.SUMMARY, "Summary", ("Hi.",)),
        (SectionKind.SKILLS, "SKILLS", ("Git",)),
    ]
    first = parsed.sections[0]
    assert parsed.lines[first.start_line] == "Summary"
    assert first.end_line == parsed.sections[1].start_line


def test_document_without_headings():
    parsed = parse_resume("Ann Lee\nann@example.com | +1 555-0100\nI build things.\nMore text.\n")
    assert parsed.header == ("Ann Lee", "ann@example.com | +1 555-0100")
    assert [(s.kind, s.heading, s.lines) for s in parsed.sections] == [
        (SectionKind.OTHER, None, ("I build things.", "More text.")),
    ]
    no_contact = parse_resume("Ann Lee\nI build things.\n")
    assert no_contact.header == ("Ann Lee",)
    assert parse_resume("").sections == ()


# --- Dates --------------------------------------------------------------------------------

POINTS = {"month_name": "Jan 2020", "iso": "2020-01", "numeric": "01/2020", "year_only": "2020"}
ENDS = {"month_name": "Mar 2023", "iso": "2023-03", "numeric": "03/2023", "year_only": "2023"}
SEPARATORS = (" \u2013 ", " - ", " to ", "\u2013", "-")
ONGOING = ("Present", "present", "Current")


@pytest.mark.parametrize(
    ("fmt", "separator", "end"),
    [
        (fmt, sep, end)
        for fmt, sep in itertools.product(POINTS, SEPARATORS)
        if sep.strip() or fmt == "year_only" or sep == "\u2013"
        for end in (ENDS[fmt], *ONGOING)
    ],
)
def test_date_ranges_detected(fmt, separator, end):
    text = f"{POINTS[fmt]}{separator}{end}"
    assert is_date_range(text), text
    match = find_date_range(f"Acme Ltd | {text}")
    assert match and match.group("start") == POINTS[fmt] and match.group("end") == end


@pytest.mark.parametrize(
    "text",
    ["2020", "Jan 2020", "version 2.0 - 3.1", "15 years", "B.Sc. 2016", "100-200 ms", "2020-13"],
)
def test_non_ranges_not_detected(text):
    assert not is_date_range(text)


def test_range_inside_longer_text_is_found_but_not_exact():
    line = "Hill Country University, 2014 \u2013 2018 | Grade: 3.6"
    assert find_date_range(line).group() == "2014 \u2013 2018"
    assert not is_date_range(line)


# --- Roles --------------------------------------------------------------------------------


def experience(*lines: str) -> Section:
    return Section(SectionKind.EXPERIENCE, "Experience", tuple(lines), 0, len(lines) + 1)


def roles_of(*lines: str) -> list[tuple]:
    return [(r.title, r.company, r.date_text, r.lines) for r in split_roles(experience(*lines))]


def test_every_role_form_has_a_function():
    assert [f.__name__ for f in ROLE_HEADER_FORMS] == ["_inline", "_stacked", "_split"]


def test_stacked_roles():
    assert roles_of(
        "Senior Engineer",
        "Acme | Jan 2020 \u2013 Present",
        "Led things.",
        "Engineer",
        "Beta Corp, London | 2018 - 2019",
        "Built things.",
        "More things.",
    ) == [
        ("Senior Engineer", "Acme", "Jan 2020 \u2013 Present", ("Led things.",)),
        ("Engineer", "Beta Corp, London", "2018 - 2019", ("Built things.", "More things.")),
    ]


def test_inline_roles_and_modern_docx_after_tab_normalisation():
    lines = clean_lines(
        "Engineer | Acme | 2025-05 to Current\nBuilt it.\nLead | Beta\t01/2020 - 03/2023\nRan it.\n"
    )
    assert roles_of(*lines) == [
        ("Engineer", "Acme", "2025-05 to Current", ("Built it.",)),
        ("Lead", "Beta", "01/2020 - 03/2023", ("Ran it.",)),
    ]


def test_split_roles_modern_pdf():
    assert roles_of(
        "ML Engineer | Lumen",
        "2023-03 - Present",
        "Trained models,",
        "wrapped onto two lines.",
        "CV Engineer | Harbor",
        "2020-10 - 2023-02",
        "Counted containers.",
    ) == [
        (
            "ML Engineer",
            "Lumen",
            "2023-03 - Present",
            ("Trained models,", "wrapped onto two lines."),
        ),
        ("CV Engineer", "Harbor", "2020-10 - 2023-02", ("Counted containers.",)),
    ]


def test_lines_before_the_first_role_belong_to_no_role():
    assert roles_of("A short intro.", "Engineer", "Acme | 2020 - 2021", "Did it.") == [
        ("Engineer", "Acme", "2020 - 2021", ("Did it.",))
    ]
    assert roles_of("No roles here.", "Just text.") == []


def test_bullet_with_a_pipe_is_not_a_role():
    assert roles_of("Engineer", "Acme | 2020 - 2021", "Cut p95 | p99 latency by half.") == [
        ("Engineer", "Acme", "2020 - 2021", ("Cut p95 | p99 latency by half.",))
    ]


# --- Corpus -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parsed() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = {c["candidate_id"]: c for c in json.load(handle)["candidates"]}
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WORKSPACE_ROOT", str(ROOT / "data"))
        for persona in PERSONAS:
            result = fs_tools.read_file(manifest[persona.id]["resume_path"])
            assert result["success"], result["error"]
            out[persona.id] = parse_resume(result["data"]["content"])
    return out


def roles_for(parsed_resume) -> list:
    return [r for s in parsed_resume.of_kind(SectionKind.EXPERIENCE) for r in split_roles(s)]


@any_persona
def test_detected_sections_match_rendered_sections(persona: Persona, parsed):
    resume = parsed[persona.id]
    assert [s.kind.value for s in resume.sections] == section_order(persona)
    assert resume.header[0] == persona.name


def test_no_skills_section_persona(parsed):
    assert parsed["nina_kowalski"].of_kind(SectionKind.SKILLS) == []


@pytest.mark.parametrize("pid", ["elena_rodriguez", "grace_kim"])
def test_skills_table_rows_inside_skills_section(pid: str, parsed):
    persona = BY_ID[pid]
    (skills,) = parsed[pid].of_kind(SectionKind.SKILLS)
    rows = [" | ".join(row) for row in _skills(persona, STYLES["table-skills"])[0].rows]
    assert list(skills.lines) == rows


@any_persona
def test_roles_match_persona(persona: Persona, parsed):
    roles = roles_for(parsed[persona.id])
    assert len(roles) == len(persona.experience)
    for found, role in zip(roles, persona.experience, strict=True):
        assert (found.title, found.company) == (role.title, role.company)
        assert found.date_text == date_range(role, persona)
        body = collapse(" ".join(found.lines))
        for bullet in role.bullets:
            assert collapse(bullet) in body, bullet
    for found, following in itertools.pairwise(zip(roles, persona.experience, strict=True)):
        (role_found, _), (_, next_role) = found, following
        body = collapse(" ".join(role_found.lines))
        assert next_role.title not in role_found.lines
        assert collapse(next_role.bullets[0]) not in body


@any_persona
def test_no_footer_text_anywhere(persona: Persona, parsed):
    resume = parsed[persona.id]
    assert not any(re.search(r"Page \d+ of \d+", line) for line in resume.lines)


def test_john_doe_education_after_page_break(parsed):
    (education,) = parsed["john_doe"].of_kind(SectionKind.EDUCATION)
    assert education.heading == "EDUCATION"
    assert education.lines[0] == "B.S. in Computer Science"
    assert parsed["john_doe"].sections[-1].kind is SectionKind.CERTIFICATIONS


def test_peter_hughes_python_only_in_hobbies(parsed):
    resume = parsed["peter_hughes"]
    hits = {
        s.kind
        for s in resume.sections
        if any("python" in line.casefold() for line in (s.heading or "", *s.lines))
    }
    assert hits == {SectionKind.HOBBIES}
    assert not any("python" in line.casefold() for line in resume.header)


def test_corpus_files_exist():
    assert len(list((ROOT / "data" / "resumes").iterdir())) == len(PERSONAS)
