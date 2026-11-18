"""Resume generator: layouts, renderers and the committed files in data/resumes/."""

import json
import re
from pathlib import Path

import pytest
from docx import Document
from skill_mentions import mentioned_skills

import fs_tools
from scripts import generate_resumes
from scripts.build_manifest import MANIFEST_PATH
from scripts.generate_resumes import DATA_DIR, SUPPORTED, generate, render
from scripts.layouts import (
    LAYOUT_FUNCTIONS,
    SKILL_GROUPS,
    STYLES,
    _skills,
    build,
    date_range,
    expected_headings,
    heading_text,
    section_order,
    skill_label,
)
from scripts.persona_schema import (
    GROUND_TRUTH_DIR,
    HEADER_VARIANTS,
    LAYOUTS,
    PERSONA_DIR,
    Persona,
    ResumeFormat,
    load_personas,
)
from scripts.render_txt import render_txt
from scripts.resume_doc import Heading, RoleHeader, block_texts

PERSONAS = load_personas()
BY_ID = {p.id: p for p in PERSONAS}
RENDERED = [p for p in PERSONAS if p.render.format in SUPPORTED]
README = (GROUND_TRUTH_DIR / "README.md").read_text(encoding="utf-8")
JAVA = re.compile(r"(?<!\w)java(?!\w)", re.IGNORECASE)

rendered_persona = pytest.mark.parametrize("persona", RENDERED, ids=[p.id for p in RENDERED])
any_persona = pytest.mark.parametrize("persona", PERSONAS, ids=[p.id for p in PERSONAS])


def read(root: Path, relpath: str) -> str:
    """Extract a file's text with fs_tools.read_file, the loader the RAG pipeline uses."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WORKSPACE_ROOT", str(root))
        result = fs_tools.read_file(relpath)
    assert result["success"], result["error"]
    return result["data"]["content"]


def collapse(text: str) -> str:
    return " ".join(text.split())


def lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()]


@pytest.fixture(scope="module")
def manifest() -> dict[str, dict]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return {c["candidate_id"]: c for c in json.load(handle)["candidates"]}


@pytest.fixture(scope="module")
def committed(manifest) -> dict[str, str]:
    """Extracted text of every committed TXT/DOCX resume, keyed by persona id."""
    return {p.id: read(DATA_DIR, manifest[p.id]["resume_path"]) for p in RENDERED}


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory) -> tuple[Path, Path]:
    first, second = tmp_path_factory.mktemp("run1"), tmp_path_factory.mktemp("run2")
    generate(PERSONAS, first)
    generate(PERSONAS, second)
    return first, second


# --- Encoding guard -----------------------------------------------------------------------


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


@any_persona
def test_persona_text_encodes_in_cp1252(persona: Persona):
    # reportlab's standard Helvetica (step 3b) can only show cp1252 characters.
    data = json.loads((PERSONA_DIR / f"{persona.id}.json").read_text(encoding="utf-8"))
    bad = []
    for text in [*_strings(data), *block_texts(build(persona))]:
        try:
            text.encode("cp1252")
        except UnicodeEncodeError:
            bad.append(text)
    assert bad == []


# --- Layouts ------------------------------------------------------------------------------


def test_every_layout_has_a_function_and_style():
    assert set(LAYOUT_FUNCTIONS) == set(STYLES) == set(LAYOUTS)


def test_layout_function_rejects_other_layouts():
    zoe = BY_ID["zoe_turner"]
    assert LAYOUT_FUNCTIONS["minimal"](zoe) == build(zoe)
    with pytest.raises(ValueError, match="minimal"):
        LAYOUT_FUNCTIONS["classic"](zoe)


def test_readme_layout_table_matches_code():
    table = README.split("<!-- layout-table", 1)[1].split("\n\n", 1)[0]
    rows = [r for r in table.splitlines()[1:] if r.startswith("| `")]
    separators = {"en dash": " \N{EN DASH} ", "hyphen": " - ", "to": " to "}
    documented = {}
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        documented[cells[0].strip("`")] = cells[1:]
    assert set(documented) == set(STYLES)
    for name, style in STYLES.items():
        case, underline, docx_heading, role, sep, ongoing, skills, order = documented[name]
        assert case == ("caps" if style.caps else "title") + (" + colon" if style.colon else "")
        assert underline == (style.underline or "none"), name
        assert docx_heading == ("styled" if style.styled_headings else "bold"), name
        assert role == style.role_style.value, name
        assert separators[sep] == style.separator, name
        assert ongoing == style.ongoing, name
        assert skills == style.skills, name
        expected = ", ".join(style.sections)
        if style.panel:
            expected = f"[panel: {', '.join(style.panel)}], {expected}"
        assert order == expected, name


def test_readme_lists_every_heading_string():
    for variant in HEADER_VARIANTS.values():
        for heading in variant.values():
            assert f"| {heading} |" in README or f" {heading} |" in README, heading


def test_heading_styles():
    assert heading_text("experience", BY_ID["john_doe"]) == "EXPERIENCE"  # classic, standard
    assert heading_text("skills", BY_ID["emily_clarke"]) == "TECHNICAL SKILLS:"  # compact
    assert heading_text("education", BY_ID["ishaan_gupta"]) == "Education:"  # academic
    assert heading_text("hobbies", BY_ID["tomas_novak"]) == "OUTSIDE WORK"  # plain, employment


def test_date_ranges_per_format():
    def first(pid: str) -> str:
        return date_range(BY_ID[pid].experience[0], BY_ID[pid])

    assert first("elena_rodriguez") == "Mar 2023 - Current"  # table-skills, month_name
    assert first("noah_williams") == "02/2023 - present"  # plain, numeric
    assert first("zoe_turner") == "2025-05 to Current"  # minimal, iso
    assert first("david_okonkwo") == "2020 \N{EN DASH} Present"  # classic, year_only


def test_academic_puts_education_first_and_compact_skills_first():
    assert section_order(BY_ID["ishaan_gupta"])[:2] == ["education", "coursework"]
    assert section_order(BY_ID["emily_clarke"])[0] == "skills"


@any_persona
def test_skills_section_mentions_exactly_its_skills(persona: Persona):
    # The rendered Skills section must read back as the persona's skills_section, so group
    # names and separators add no mentions of their own.
    if not persona.skills_section:
        return
    text = "\n".join(block_texts(_skills(persona, STYLES[persona.render.layout])))
    assert mentioned_skills(text) == set(persona.skills_section)


def test_skill_group_names_are_not_skills():
    for name in SKILL_GROUPS.values():
        assert mentioned_skills(name) == set(), name


@any_persona
def test_blocks_carry_every_bullet_verbatim(persona: Persona):
    texts = block_texts(build(persona))
    for role in persona.experience:
        for bullet in role.bullets:
            assert bullet in texts
    for project in persona.projects:
        assert project.description in texts


# --- Committed files ----------------------------------------------------------------------


@rendered_persona
def test_file_exists_at_manifest_path(persona: Persona, manifest):
    relpath = manifest[persona.id]["resume_path"]
    path = DATA_DIR / relpath
    assert path.is_file(), f"missing {relpath}; run: python -m scripts.generate_resumes"
    assert path.suffix == f".{persona.render.format.value}"


def test_resume_dir_holds_only_generated_files(manifest):
    expected = {Path(manifest[p.id]["resume_path"]).name for p in RENDERED}
    actual = {p.name for p in (DATA_DIR / "resumes").iterdir() if p.name != ".gitkeep"}
    assert actual == expected


@rendered_persona
def test_committed_file_is_up_to_date(persona: Persona, two_runs, committed):
    fresh = read(two_runs[0], f"{persona.id}.{persona.render.format.value}")
    assert committed[persona.id] == fresh, "stale; run: python -m scripts.generate_resumes"


@rendered_persona
def test_extracted_text_has_every_rendered_field(persona: Persona, committed):
    text = committed[persona.id]
    flat = collapse(text)
    assert persona.name in flat
    assert collapse(persona.summary) in flat
    for role in persona.experience:
        assert date_range(role, persona) in flat, role.company
        for bullet in role.bullets:
            assert collapse(bullet) in flat, bullet
    for project in persona.projects:
        assert collapse(project.description) in flat
    for skill in persona.skills_section:
        assert skill_label(persona, skill) in flat, skill
    # Every heading the layout should emit, each alone on its line, in order.
    expected = expected_headings(persona)
    assert [line for line in lines(text) if line in expected] == expected


def test_no_skills_section_persona_has_no_skills_heading(committed):
    for persona in RENDERED:
        if "no_skills_section" not in persona.edge_cases:
            continue
        every_skills_heading = {
            form
            for variant in HEADER_VARIANTS.values()
            for base in (variant["skills"], variant["skills"].upper())
            for form in (base, f"{base}:")
        }
        assert set(lines(committed[persona.id])).isdisjoint(every_skills_heading)
        assert "skills" not in section_order(persona)
        break
    else:
        pytest.fail("no rendered persona is tagged no_skills_section")


def test_js_heavy_persona_has_no_java_token(committed):
    assert not JAVA.search(committed["emily_clarke"])


def test_peter_hughes_traps_in_rendered_text():
    # peter_hughes is a PDF persona (step 3b); check his layout's text through the TXT
    # renderer so the trap guards already hold for the blocks the PDF will draw.
    peter = BY_ID["peter_hughes"]
    blocks = build(peter)
    text = render_txt(blocks)
    role_lines = {f"{b.company} | {b.date_range}" for b in blocks if isinstance(b, RoleHeader)} | {
        b.title for b in blocks if isinstance(b, RoleHeader)
    }
    spark_lines = [line for line in lines(text) if "Spark" in line or "spark" in line.lower()]
    assert spark_lines and all(line in role_lines for line in spark_lines), spark_lines
    hobbies = heading_text("hobbies", peter)
    before, _, after = text.partition(f"\n{hobbies}\n")
    assert after, "no hobbies section"
    assert "python" not in before.lower() and "Python" in after


@pytest.mark.parametrize("pid", ["elena_rodriguez", "grace_kim"])
def test_skills_table_is_a_real_docx_table(pid: str, manifest):
    persona = BY_ID[pid]
    document = Document(str(DATA_DIR / manifest[pid]["resume_path"]))
    assert len(document.tables) == 1
    cells = " ".join(c.text for row in document.tables[0].rows for c in row.cells)
    for skill in persona.skills_section:
        assert skill_label(persona, skill) in cells, skill
    styles = {p.style.name for p in document.paragraphs}
    assert {"Title", "Heading 1", "List Bullet"} <= styles


def test_docx_extraction_keeps_tables_in_document_order(committed):
    # Documents fs_tools' behaviour (iter_inner_content): table rows come out where the table
    # sits, so the skills rows land between the skills heading and the next heading.
    for pid in ("elena_rodriguez", "grace_kim"):
        persona = BY_ID[pid]
        text_lines = lines(committed[pid])
        order = section_order(persona)
        skills_at = text_lines.index(heading_text("skills", persona))
        next_at = text_lines.index(heading_text(order[order.index("skills") + 1], persona))
        rows = [" | ".join(row) for row in _skills(persona, STYLES["table-skills"])[0].rows]
        assert text_lines[skills_at + 1 : next_at] == rows, pid


def test_docx_core_properties_fixed(manifest):
    document = Document(str(DATA_DIR / manifest["elena_rodriguez"]["resume_path"]))
    props = document.core_properties
    assert props.author == "Elena Rodriguez"
    assert props.created.isoformat().startswith("2026-11-01T00:00:00")
    assert props.modified == props.created
    assert props.revision == 1


def test_styled_headings_only_where_layout_says(manifest):
    for persona in RENDERED:
        if persona.render.format is not ResumeFormat.DOCX:
            continue
        document = Document(str(DATA_DIR / manifest[persona.id]["resume_path"]))
        headings = [p.text for p in document.paragraphs if p.style.name == "Heading 1"]
        styled = STYLES[persona.render.layout].styled_headings
        assert headings == (expected_headings(persona) if styled else []), persona.id


# --- Determinism and CLI ------------------------------------------------------------------


@rendered_persona
def test_generation_is_deterministic(persona: Persona, two_runs):
    name = f"{persona.id}.{persona.render.format.value}"
    first, second = (root / name for root in two_runs)
    assert read(two_runs[0], name) == read(two_runs[1], name)
    if persona.render.format is ResumeFormat.TXT:
        assert first.read_bytes() == second.read_bytes()
        assert b"\r" not in first.read_bytes()


def test_cli_only_renders_one_persona(tmp_path: Path, capsys):
    assert generate_resumes.main(["--only", "zoe_turner", "--out-dir", str(tmp_path)]) == 0
    assert [p.name for p in tmp_path.iterdir()] == ["zoe_turner.txt"]


def test_cli_skips_pdf_with_message(tmp_path: Path, capsys):
    assert generate_resumes.main(["--only", "john_doe", "--out-dir", str(tmp_path)]) == 0
    assert list(tmp_path.iterdir()) == []
    assert "step 3b" in capsys.readouterr().err


def test_cli_rejects_unknown_id(tmp_path: Path):
    with pytest.raises(SystemExit):
        generate_resumes.main(["--only", "nobody", "--out-dir", str(tmp_path)])


def test_render_refuses_pdf(tmp_path: Path):
    with pytest.raises(ValueError, match="3b"):
        render(BY_ID["john_doe"], tmp_path)


def test_name_is_the_first_block():
    for persona in PERSONAS:
        first = build(persona)[0]
        assert isinstance(first, Heading) and first.level == 0 and first.text == persona.name
