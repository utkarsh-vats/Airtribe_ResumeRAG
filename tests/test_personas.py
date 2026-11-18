"""Validation of the persona files, the persona plan and the ground-truth manifest maths."""

import json
import re
import unicodedata
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from skill_mentions import find_mentions, mentioned_skills

from rag.skills import _key, canonicalize, expand_implied, get_skill
from scripts.build_manifest import (
    MANIFEST_PATH,
    build_manifest,
    candidate_skills,
    highest_education,
    merged_months,
    month_index,
    months_to_years,
    render_manifest,
    skill_years,
    total_years,
)
from scripts.jd_themes import load_jd_themes, unmet_must_haves
from scripts.persona_plan import RELEVANCE, Slot, load_plan
from scripts.persona_schema import (
    AS_OF,
    EDGE_CASES,
    FORMAT_LAYOUTS,
    LAYOUTS,
    PERSONA_DIR,
    PRESENT,
    DateFormat,
    Education,
    EmploymentType,
    Persona,
    PersonaError,
    Project,
    ResumeFormat,
    Role,
    load_persona,
    load_personas,
    persona_from_dict,
)

PERSONA_FILES = sorted(PERSONA_DIR.glob("*.json"))
PLAN_SIZE = 35
THEMES = {theme.id: theme for theme in load_jd_themes()}
JDS = tuple(THEMES)
REQUIRED_EDGE_CASES = (
    "python_in_hobbies",
    "js_for_java_jd",
    "java_for_frontend_jd",
    "no_skills_section",
    "skills_table",
    "aliases_only",
    "overlapping_roles",
    "career_gap",
    "career_changer",
    "coursework_only_skill",
    "listed_never_used",
    "duplicate_name",
    "two_page",
    "minimal",
    "internships_only",
)
MILESTONE1_IDS = {
    "john_doe",
    "priya_sharma",
    "marcus_chen",
    "elena_rodriguez",
    "aisha_okafor",
    "liam_walsh",
    "sofia_rossi",
    "kenji_tanaka",
}


# --- Helpers ------------------------------------------------------------------------------


def slug(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def skill_texts(persona: Persona) -> list[tuple[str, tuple[str, ...], str]]:
    """``(where, skills_used, text)`` for every role's bullets and project's description."""
    texts = [(f"role {r.company}", r.skills_used, "\n".join(r.bullets)) for r in persona.experience]
    texts += [(f"project {p.name}", p.skills_used, p.description) for p in persona.projects]
    return texts


def all_skill_refs(persona: Persona) -> list[str]:
    refs = list(persona.skills_section)
    for role in persona.experience:
        refs.extend(role.skills_used)
    for project in persona.projects:
        refs.extend(project.skills_used)
    return refs


@pytest.fixture(scope="module")
def personas() -> list[Persona]:
    return load_personas()


@pytest.fixture(scope="module")
def plan() -> dict[str, Slot]:
    return load_plan()


@pytest.fixture(scope="module")
def manifest(personas) -> dict[str, dict]:
    return {c["candidate_id"]: c for c in build_manifest(personas)["candidates"]}


@pytest.fixture(scope="module")
def committed_manifest() -> dict[str, dict]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return {c["candidate_id"]: c for c in json.load(handle)["candidates"]}


persona_file = pytest.mark.parametrize("path", PERSONA_FILES, ids=[p.stem for p in PERSONA_FILES])


# --- Persona files ------------------------------------------------------------------------


def test_milestone1_personas_present():
    assert {p.stem for p in PERSONA_FILES} >= MILESTONE1_IDS


@persona_file
def test_persona_loads_and_id_matches_filename(path: Path):
    assert load_persona(path).id == path.stem


def test_ids_unique(personas):
    counts = Counter(p.id for p in personas)
    assert [i for i, n in counts.items() if n > 1] == []


@persona_file
def test_id_is_name_slug(path: Path):
    persona = load_persona(path)
    base = slug(persona.name)
    if persona.id != base:
        match = re.fullmatch(rf"{re.escape(base)}_(\d+)", persona.id)
        assert match and int(match.group(1)) >= 2, f"{persona.id} is not a slug of {persona.name}"
        assert (PERSONA_DIR / f"{base}.json").exists(), "a _N suffix needs the base persona"


def test_contact_details_fictional_and_unique(personas):
    for p in personas:
        assert p.email.endswith("@example.com"), p.id
        assert re.fullmatch(r"\+\d{1,3} 555-01\d\d", p.phone), p.id
    for field in ("email", "phone"):
        counts = Counter(getattr(p, field) for p in personas)
        assert [v for v, n in counts.items() if n > 1] == [], field


@persona_file
def test_all_skills_canonical(path: Path):
    persona = load_persona(path)
    for ref in all_skill_refs(persona):
        assert canonicalize(ref) == ref, f"{ref!r} is not a canonical skill name"
    assert len(set(persona.skills_section)) == len(persona.skills_section), "duplicate skill"
    for skill, label in persona.render.skill_labels.items():
        assert skill in persona.skills_section, f"label for unlisted skill {skill!r}"
        assert canonicalize(label) == skill, f"label {label!r} does not mean {skill!r}"


@persona_file
def test_skills_section_entries_are_recognised(path: Path):
    # What the Skills section will display must be read back as exactly that skill.
    persona = load_persona(path)
    for skill in persona.skills_section:
        shown = persona.render.skill_labels.get(skill, skill)
        assert mentioned_skills(shown) == {skill}, f"{shown!r} is not recognised as {skill!r}"


@persona_file
def test_dates_valid_and_ordered(path: Path):
    persona = load_persona(path)
    as_of = month_index(PRESENT)
    for role in persona.experience:
        start, end = month_index(role.start), month_index(role.end)
        assert start <= end <= as_of, f"{role.company}: {role.start}..{role.end}"
    for edu in persona.education:
        assert edu.start_year <= edu.end_year <= AS_OF.year, edu.institution


@persona_file
def test_year_only_dates_hide_no_months(path: Path):
    persona = load_persona(path)
    if persona.render.date_format is not DateFormat.YEAR_ONLY:
        return
    for role in persona.experience:
        assert role.start.endswith("-01"), f"{role.company} start {role.start}"
        assert role.end == PRESENT or role.end.endswith("-12"), f"{role.company} end {role.end}"


@persona_file
def test_roles_list_skills_and_have_bullets(path: Path):
    persona = load_persona(path)
    for role in persona.experience:
        low, high = (1, 5) if persona.id in MILESTONE1_IDS else (3, 5)
        assert low <= len(role.bullets) <= high, f"{role.company}: {len(role.bullets)} bullets"
        if role.employment_type is not EmploymentType.INTERNSHIP:
            assert role.skills_used, f"{role.company}: non-internship role lists no skills"


@persona_file
def test_skills_used_are_mentioned(path: Path):
    for where, used, text in skill_texts(load_persona(path)):
        evidenced = expand_implied(mentioned_skills(text))
        missing = sorted(set(used) - evidenced)
        assert missing == [], f"{where}: never mentions {missing}"


@persona_file
def test_no_mentions_beyond_skills_used(path: Path):
    for where, used, text in skill_texts(load_persona(path)):
        extra = sorted(mentioned_skills(text) - expand_implied(used))
        assert extra == [], f"{where}: mentions {extra} but skills_used does not cover them"


@persona_file
def test_aliases_only_persona_never_writes_canonical_names(path: Path):
    persona = load_persona(path)
    if "aliases_only" not in persona.edge_cases:
        return
    for skill in persona.skills_section:
        if get_skill(skill).aliases:
            label = persona.render.skill_labels.get(skill, skill)
            assert _key(label) != _key(skill), f"{skill} is shown by its canonical name"
    for where, _used, text in skill_texts(persona):
        canonical = [
            m.text for m in find_mentions(text) if not m.via_alias and get_skill(m.skill).aliases
        ]
        assert canonical == [], f"{where}: canonical names {canonical}"


@persona_file
def test_skills_released_before_role_start(path: Path):
    # Projects carry no dates, so only roles can be checked.
    for role in load_persona(path).experience:
        start_year = int(role.start[:4])
        early = {
            name: released
            for name in expand_implied(role.skills_used)
            if (released := get_skill(name).first_release_year) and released > start_year
        }
        assert early == {}, f"{role.company} starts {role.start} but uses {early}"


def persona_strings(persona: Persona) -> list[str]:
    """Every free-text string in the persona file (for whole-resume token checks)."""

    def walk(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from walk(item)

    return list(walk(json.loads((PERSONA_DIR / f"{persona.id}.json").read_text("utf-8"))))


def test_js_heavy_persona_never_writes_java(personas):
    java = re.compile(r"(?<!\w)java(?!\w)", re.IGNORECASE)
    for persona in personas:
        if "js_for_java_jd" in persona.edge_cases:
            hits = [s for s in persona_strings(persona) if java.search(s)]
            assert hits == [], f"{persona.id}: {hits}"


def test_peter_hughes_traps_stay_where_planned(personas):
    peter = next(p for p in personas if p.id == "peter_hughes")
    python = re.compile(r"(?<!\w)python(?!\w)", re.IGNORECASE)
    spark = re.compile(r"(?<!\w)spark(?!\w)", re.IGNORECASE)
    hobby_text = set(peter.hobbies)
    company_text = {r.company for r in peter.experience}
    strings = persona_strings(peter)
    assert any(python.search(h) for h in hobby_text)
    assert [s for s in strings if python.search(s) and s not in hobby_text] == []
    assert any(spark.search(c) for c in company_text)
    assert [s for s in strings if spark.search(s) and s not in company_text] == []
    bullets = "\n".join(b for r in peter.experience for b in r.bullets)
    assert "airflow" in bullets and "Airflow" not in bullets
    assert not re.search(r"(?<!\w)ml(?!\w)", bullets, re.IGNORECASE)


@persona_file
def test_edge_case_tags_consistent(path: Path):
    persona = load_persona(path)
    tags = set(persona.edge_cases)
    render = persona.render
    assert render.layout in FORMAT_LAYOUTS[render.format]
    assert ("no_skills_section" in tags) == (not persona.skills_section)
    assert ("skills_table" in tags) == (render.layout == "table-skills")
    if "internships_only" in tags:
        assert persona.experience
        assert all(r.employment_type is EmploymentType.INTERNSHIP for r in persona.experience)
    if "duplicate_name" in tags:
        assert re.search(r"_\d+$", persona.id)


def test_manifest_is_up_to_date(personas):
    expected = render_manifest(build_manifest(personas))
    assert MANIFEST_PATH.read_text(encoding="utf-8") == expected, (
        "manifest.json is stale; run: python -m scripts.build_manifest"
    )


def test_milestone1_facts_still_hold(manifest):
    m1 = {cid: c for cid, c in manifest.items() if cid in MILESTONE1_IDS}

    def having(skill: str) -> set[str]:
        return {cid for cid, c in m1.items() if skill in c["skills"]}

    assert having("Python") == {"john_doe", "kenji_tanaka", "liam_walsh", "priya_sharma"}
    assert having("Kubernetes") == {"marcus_chen", "kenji_tanaka"}
    assert having("Java") == {"elena_rodriguez"}
    assert {cid for cid, c in m1.items() if c["total_years"] >= 5} == {
        "elena_rodriguez",
        "john_doe",
        "kenji_tanaka",
        "marcus_chen",
        "priya_sharma",
    }


# --- Persona plan -------------------------------------------------------------------------


def test_plan_rows_valid(plan):
    assert len(plan) == PLAN_SIZE
    for slot in plan.values():
        assert slot.id == slug(slot.id), slot.id
        fmt = ResumeFormat(slot.format)
        assert slot.layout in LAYOUTS and slot.layout in FORMAT_LAYOUTS[fmt], slot.id
        DateFormat(slot.date_format)
        assert slot.edge_cases <= EDGE_CASES.keys(), slot.id
        assert set(slot.relevance.values()) <= set(RELEVANCE), slot.id


def test_plan_meets_coverage_rules(plan):
    slots = list(plan.values())
    for jd in JDS:
        counts = Counter(s.relevance[jd] for s in slots)
        assert counts["strong"] >= 3, jd
        assert counts["partial"] >= 3, jd
        assert counts["distractor"] >= 2, jd
    formats = Counter(s.format for s in slots)
    for fmt, target in (("pdf", 14), ("docx", 12), ("txt", 9)):
        assert abs(formats[fmt] - target) <= 1, formats
    assert {s.date_format for s in slots} == {d.value for d in DateFormat}
    non_tech = [s for s in slots if "non_tech" in s.edge_cases]
    assert len(non_tech) >= 2
    assert all(set(s.relevance.values()) == {"none"} for s in non_tech)
    assert any("internships_only" in s.edge_cases and s.years == 0 for s in slots)
    assert max(s.years for s in slots) >= 15
    placed = set().union(*(s.edge_cases for s in slots))
    assert set(REQUIRED_EDGE_CASES) <= placed, set(REQUIRED_EDGE_CASES) - placed


@persona_file
def test_persona_matches_its_plan_slot(path: Path, plan, manifest):
    persona = load_persona(path)
    slot = plan.get(persona.id)
    assert slot is not None, f"{persona.id} has no slot in persona_plan.md"
    assert persona.render.format == slot.format
    assert persona.render.layout == slot.layout
    assert persona.render.date_format == slot.date_format
    assert set(persona.edge_cases) == slot.edge_cases
    assert abs(manifest[persona.id]["total_years"] - slot.years) <= 1.0


# --- Labels vs JD must-haves --------------------------------------------------------------


def test_jd_themes_match_plan_columns(plan):
    assert JDS == (
        "python_backend",
        "ml_engineer",
        "devops_sre",
        "frontend",
        "java_backend",
        "data_engineer",
    )
    assert all(tuple(slot.relevance) == JDS for slot in plan.values())


def _labelled(plan, committed_manifest, jd: str, labels: set[str]):
    for cid, candidate in committed_manifest.items():
        if plan[cid].relevance[jd] in labels:
            yield cid, unmet_must_haves(THEMES[jd], candidate)


@pytest.mark.parametrize("jd", JDS)
def test_strong_candidates_meet_every_must_have(jd, plan, committed_manifest):
    failing = {cid: u for cid, u in _labelled(plan, committed_manifest, jd, {"strong"}) if u}
    assert failing == {}


@pytest.mark.parametrize("jd", JDS)
def test_distractors_fail_some_must_have(jd, plan, committed_manifest):
    passing = [cid for cid, u in _labelled(plan, committed_manifest, jd, {"distractor"}) if not u]
    assert passing == []


@pytest.mark.parametrize("jd", JDS)
def test_candidates_meeting_every_must_have_are_strong_or_partial(jd, plan, committed_manifest):
    mislabelled = [
        f"{cid} ({plan[cid].relevance[jd]})"
        for cid, u in _labelled(plan, committed_manifest, jd, {"distractor", "none"})
        if not u
    ]
    assert mislabelled == []


def test_unmet_must_haves_rules():
    theme = THEMES["python_backend"]
    base = {
        "total_years": 6.0,
        "skills": ["Django", "PostgreSQL", "Python", "SQL"],
        "skill_years": {"Django": 0.0, "PostgreSQL": 1.0, "Python": 5.0, "SQL": 1.0},
    }
    assert unmet_must_haves(theme, base) == []
    short = {**base, "skill_years": {**base["skill_years"], "Python": 4.9}}
    assert unmet_must_haves(theme, short) == ["5+ yrs Python"]
    no_framework = {**base, "skills": ["PostgreSQL", "Python", "SQL"]}
    assert unmet_must_haves(theme, no_framework) == ["Django or FastAPI"]
    devops = THEMES["devops_sre"]
    junior = {
        "total_years": 3.9,
        "skills": ["AWS", "Kubernetes", "Terraform"],
        "skill_years": {"AWS": 3.9, "Kubernetes": 3.9, "Terraform": 3.9},
    }
    assert unmet_must_haves(devops, junior) == ["4+ total yrs"]


def test_every_plan_slot_has_a_persona(personas, plan, manifest):
    if len(personas) < PLAN_SIZE:
        pytest.skip(f"{len(personas)}/{PLAN_SIZE} personas authored; the rest arrive in step 2c")
    assert {p.id for p in personas} == set(plan)
    assert Counter(p.render.format.value for p in personas) == {"pdf": 14, "docx": 12, "txt": 9}
    assert Counter(p.render.date_format.value for p in personas) == {
        "month_name": 13,
        "iso": 8,
        "numeric": 7,
        "year_only": 7,
    }
    for jd in JDS:
        counts = Counter(plan[p.id].relevance[jd] for p in personas)
        assert counts["strong"] >= 3 and counts["partial"] >= 3 and counts["distractor"] >= 2, jd
    by_id = {p.id: p for p in personas}
    years = [c["total_years"] for c in manifest.values()]
    assert max(years) >= 15
    for pid, p in by_id.items():
        if "internships_only" in p.edge_cases:
            assert manifest[pid]["total_years"] == 0.0
        if "non_tech" in p.edge_cases:
            assert all(get_skill(s).category == "business" for s in manifest[pid]["skills"])


# --- Schema loader ------------------------------------------------------------------------


def _raw_persona() -> dict:
    return {
        "id": "test_person",
        "name": "Test Person",
        "email": "test.person@example.com",
        "phone": "+1 555-0199",
        "location": "Nowhere",
        "headline": "Engineer",
        "summary": "Builds things.",
        "skills_section": ["Python"],
        "experience": [
            {
                "title": "Engineer",
                "company": "Acme",
                "employment_type": "full_time",
                "start": "2020-01",
                "end": "present",
                "skills_used": ["Python"],
                "bullets": ["Wrote Python."],
            }
        ],
        "education": [],
        "render": {
            "format": "txt",
            "layout": "plain",
            "header_variant": "standard",
            "date_format": "iso",
        },
    }


def test_loader_accepts_minimal_persona():
    persona = persona_from_dict(_raw_persona())
    assert persona.projects == () and persona.hobbies == () and persona.edge_cases == ()
    assert persona.render.skill_labels == {}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.pop("summary"), "missing keys"),
        (lambda d: d.update(nickname="x"), "unknown keys"),
        (lambda d: d["experience"][0].update(start="2020-13"), "YYYY-MM"),
        (lambda d: d["experience"][0].update(start="present"), "YYYY-MM"),
        (lambda d: d["experience"][0].update(employment_type="gig"), "not one of"),
        (lambda d: d["render"].update(layout="fancy"), "not one of"),
        (lambda d: d["render"].update(format="rtf"), "not one of"),
        (lambda d: d.update(edge_cases=["haunted"]), "not one of"),
        (lambda d: d.update(skills_section="Python"), "expected a list"),
    ],
)
def test_loader_rejects_schema_violations(mutate, message):
    data = _raw_persona()
    mutate(data)
    with pytest.raises(PersonaError, match=message):
        persona_from_dict(data)


# --- Manifest maths -----------------------------------------------------------------------


def role(start: str, end: str, skills=("Python",), kind=EmploymentType.FULL_TIME) -> Role:
    return Role("Engineer", "Acme", kind, start, end, tuple(skills), ("Did things.",))


def edu(degree: str, end_year: int, field: str = "Computer Science") -> Education:
    return Education(degree, field, f"Uni {end_year}", end_year - 3, end_year, None)


BASE = persona_from_dict(_raw_persona())


def make(*roles: Role, **changes) -> Persona:
    return replace(
        BASE, experience=roles, skills_section=changes.pop("skills_section", ()), **changes
    )


def test_month_index_present_is_as_of_month():
    assert month_index(PRESENT) == month_index("2026-11") == 2026 * 12 + 10
    assert month_index("2020-01") == 2020 * 12


@pytest.mark.parametrize(
    ("months", "years"), [(0, 0.0), (1, 0.1), (6, 0.5), (11, 0.9), (13, 1.1), (107, 8.9)]
)
def test_months_to_years(months, years):
    assert months_to_years(months) == years


@pytest.mark.parametrize(
    ("intervals", "months"),
    [
        ([], 0),
        ([(0, 0)], 1),
        ([(0, 11), (6, 17)], 18),  # overlap counted once
        ([(0, 5), (6, 11)], 12),  # adjacent
        ([(0, 59), (12, 17)], 60),  # contained
        ([(24, 35), (0, 11)], 24),  # gap, unsorted input
    ],
)
def test_merged_months(intervals, months):
    assert merged_months(intervals) == months


def test_role_months_inclusive():
    assert total_years(make(role("2020-01", "2020-01"))) == 0.1
    assert total_years(make(role("2020-01", "2020-12"))) == 1.0


def test_overlapping_roles_merged_not_summed():
    persona = make(
        role("2020-01", "2020-12"), role("2020-07", "2021-06", kind=EmploymentType.FREELANCE)
    )
    assert total_years(persona) == 1.5


def test_career_gap_not_counted():
    assert total_years(make(role("2018-01", "2018-12"), role("2021-01", "2021-12"))) == 2.0


def test_present_runs_to_as_of():
    assert total_years(make(role("2026-01", PRESENT))) == 0.9


def test_internships_excluded_from_years_but_not_skills():
    persona = make(
        role("2019-01", "2019-12", skills=["PyTorch"], kind=EmploymentType.INTERNSHIP),
        role("2020-01", "2020-12", skills=["SQL"]),
    )
    assert total_years(persona) == 1.0
    years = skill_years(persona)
    assert years["PyTorch"] == 0.0 and years["Python"] == 0.0
    assert years["SQL"] == 1.0
    assert {"PyTorch", "Python", "Deep Learning", "Machine Learning"} <= candidate_skills(persona)


def test_skill_years_expand_implications_and_merge():
    persona = make(
        role("2020-01", "2021-12", skills=["Django"]),
        role("2021-07", "2022-06", skills=["Python", "pandas"]),
        skills_section=("Java",),
    )
    years = skill_years(persona)
    assert years["Django"] == 2.0
    assert years["Python"] == 2.5  # 2020-01..2022-06 merged from both roles
    assert years["pandas"] == 1.0
    assert years["Java"] == 0.0  # listed, never used
    assert list(years) == sorted(years)


def test_hobbies_and_coursework_are_not_skills():
    persona = make(
        role("2020-01", "2020-12", skills=["Excel"]),
        hobbies=("Python scripting for a home weather station",),
        education=(replace(edu("M.Sc.", 2019), coursework=("PyTorch", "Machine Learning")),),
        projects=(Project("Budget", ("SQL",), "A SQL budget tracker."),),
    )
    assert candidate_skills(persona) == {"Excel", "SQL"}


def test_highest_education():
    assert highest_education(make()) is None
    persona = make(education=(edu("B.Sc.", 2019), edu("M.Sc.", 2021), edu("Diploma", 2023)))
    assert highest_education(persona) == {
        "degree": "M.Sc.",
        "field": "Computer Science",
        "institution": "Uni 2021",
        "end_year": 2021,
    }
    tie = make(education=(edu("B.Sc.", 2015), edu("B.Eng.", 2018)))
    assert highest_education(tie)["degree"] == "B.Eng."
    assert (
        highest_education(make(education=(edu("PhD", 2016), edu("MBA", 2022))))["degree"] == "PhD"
    )


def test_build_manifest_shape_and_determinism():
    b = replace(make(role("2020-01", "2020-12")), id="b_person")
    a = replace(b, id="a_person", render=replace(b.render, format=ResumeFormat.PDF))
    manifest = build_manifest([b, a])
    assert manifest["as_of"] == "2026-11-01"
    assert [c["candidate_id"] for c in manifest["candidates"]] == ["a_person", "b_person"]
    first = manifest["candidates"][0]
    assert first["resume_path"] == "resumes/a_person.pdf"
    assert first["format"] == "pdf"
    assert set(first) == {
        "candidate_id",
        "name",
        "resume_path",
        "format",
        "skills",
        "total_years",
        "skill_years",
        "highest_education",
        "edge_cases",
    }
    assert render_manifest(manifest) == render_manifest(build_manifest([a, b]))
    assert render_manifest(manifest).endswith("}\n")
