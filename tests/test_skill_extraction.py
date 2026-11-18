"""rag.skill_matcher and the skill half of rag.metadata, against the contract and the manifest."""

import json
from datetime import date

import pytest
from skill_mentions import mentioned_skills as reference_mentions

import fs_tools
from rag.metadata import CandidateMetadata, extract_metadata
from rag.sections import SectionKind, parse_resume, split_roles
from rag.skill_matcher import find_mentions, mentioned_skills
from scripts.build_manifest import MANIFEST_PATH
from scripts.persona_schema import ROOT, load_personas

AS_OF = date(2026, 11, 1)
PERSONAS = load_personas()
any_persona = pytest.mark.parametrize("persona", PERSONAS, ids=[p.id for p in PERSONAS])


def collapse(text: str) -> str:
    return " ".join(text.split())


def extract(doc: str, as_of: date = AS_OF) -> CandidateMetadata:
    return extract_metadata("resumes/test_person.txt", parse_resume(doc), as_of)


# --- Matcher ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Postgres and k8s", {"PostgreSQL", "Kubernetes"}),
        ("POSTGRESQL, KUBERNETES", {"PostgreSQL", "Kubernetes"}),  # non-ambiguous: any case
        ("python and Python3", {"Python"}),
        ("JS, ML and sklearn", {"JavaScript", "Machine Learning", "scikit-learn"}),
        ("Golang services", {"Go"}),
        ("We Go further", {"Go"}),  # ambiguous canonical, exact case
        ("let's go further", set()),
        ("a spark of curiosity", set()),
        ("Apache Spark and apache airflow", {"Spark", "Airflow"}),  # safe aliases, any case
        ("react to incidents; React and ReactJS", {"React"}),
        ("excel at it", set()),
        ("Microsoft Excel", {"Excel"}),
        ("rest assured, RestAssured", {"REST Assured"}),
    ],
)
def test_alias_and_case_rules(text, expected):
    assert mentioned_skills(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("C++ and C# on .NET", {"C++", "C#", ".NET"}),
        ("Node.js services", {"Node.js"}),
        ("JavaScript only", {"JavaScript"}),  # Java never inside JavaScript
        ("Java, then JavaScript", {"Java", "JavaScript"}),
        ("CI/CD with A/B tests", {"CI/CD", "A/B Testing"}),
        ("pythonic code", set()),
        ("C-suite", {"C"}),  # a documented limitation of exact-case single letters
    ],
)
def test_symbol_aware_boundaries(text, expected):
    assert mentioned_skills(text) == expected


def test_overlap_policy_longest_then_earliest():
    assert [m.canonical for m in find_mentions("Apache Spark jobs")] == ["Spark"]
    assert [m.canonical for m in find_mentions("AWS Lambda and AWS")] == ["AWS Lambda", "AWS"]
    assert [m.canonical for m in find_mentions("React Native app")] == ["React Native"]
    assert [m.canonical for m in find_mentions("Tailwind CSS")] == ["Tailwind CSS"]
    first = find_mentions("Spring Boot")[0]
    assert (first.canonical, first.matched_text, first.span) == (
        "Spring Boot",
        "Spring Boot",
        (0, 11),
    )


def test_names_match_across_line_wraps():
    # A space in a multi-word name matches any whitespace, so a PDF line wrap inside
    # "GitHub Actions" still counts (the reference matcher sees only single-line text).
    assert mentioned_skills("Docker, GitHub\nActions") == {"Docker", "GitHub Actions"}


# --- Metadata: skills, months, evidence, warnings -----------------------------------------

DOC = """Test Person
Python enthusiast | test@example.com
SUMMARY
Python and Kubernetes expert.
EXPERIENCE
Python Developer
Django Works Ltd | Jan 2020 - Dec 2021
Built Django services.
Senior Engineer
Acme | Jul 2021 - Jun 2022
Ran Postgres and Python jobs.
Machine Learning Intern
Beta | Jun 2019 - Aug 2019
Trained PyTorch models.
SKILLS
Terraform, Python
PROJECTS
side-project
A FastAPI toy.
EDUCATION
B.Sc. in Python Studies
Uni of Ruby, 2015 - 2018
COURSEWORK
B.Sc. in Python Studies: Machine Learning, Spark
CERTIFICATIONS
AWS Certified Developer
INTERESTS
Python scripting, React hobby apps
"""


@pytest.fixture(scope="module")
def doc_metadata() -> CandidateMetadata:
    return extract(DOC)


def test_skill_sources_and_exclusions(doc_metadata):
    assert set(doc_metadata.skills) == {
        "Django",  # role body
        "Python",  # implied by Django, plus role bodies, skills and projects
        "PostgreSQL",
        "SQL",  # implied by PostgreSQL
        "PyTorch",  # internship body
        "Deep Learning",
        "Machine Learning",  # implied by PyTorch, not from the title or coursework
        "Terraform",  # skills section only
        "FastAPI",  # projects only
    }
    # Never from summary (Kubernetes), certifications (AWS), hobbies (React), coursework
    # (Spark), the header, titles or company names.
    for absent in ("Kubernetes", "AWS", "React", "Spark"):
        assert absent not in doc_metadata.skills


def test_skill_months(doc_metadata):
    months = doc_metadata.skill_months
    assert months["Django"] == 24  # Jan 2020 - Dec 2021
    assert months["PostgreSQL"] == months["SQL"] == 12  # Jul 2021 - Jun 2022
    assert months["Python"] == 30  # both roles, overlap merged: Jan 2020 - Jun 2022
    assert months["PyTorch"] == months["Machine Learning"] == 0  # internship: no months
    assert months["Terraform"] == 0  # skills section only
    assert months["FastAPI"] == 0  # projects only
    assert doc_metadata.skill_years == {
        "Deep Learning": 0.0,
        "Django": 2.0,
        "FastAPI": 0.0,
        "Machine Learning": 0.0,
        "PostgreSQL": 1.0,
        "PyTorch": 0.0,
        "Python": 2.5,
        "SQL": 1.0,
        "Terraform": 0.0,
    }
    assert list(doc_metadata.skill_months) == sorted(doc_metadata.skill_months)


def test_skill_evidence(doc_metadata):
    evidence = doc_metadata.skill_evidence
    # The internship counts as evidence (PyTorch implies Python) even though it adds no months.
    assert evidence["Python"] == (
        "skills",
        "experience:0",
        "experience:1",
        "experience:2",
        "projects",
    )
    assert evidence["Terraform"] == ("skills",)
    assert evidence["PyTorch"] == ("experience:2",)
    assert evidence["FastAPI"] == ("projects",)
    assert set(evidence) == set(doc_metadata.skills)


def test_title_and_company_never_give_skills():
    doc = "A\nEXPERIENCE\nPython Developer\nDjango Works | 2020 - 2021\nWrote reports.\n"
    metadata = extract(doc)
    assert metadata.skills == ()
    assert [r.title for r in metadata.roles] == ["Python Developer"]


def test_warnings_for_bad_and_clamped_dates():
    doc = (
        "A\nEXPERIENCE\n"
        "Engineer\nAcme | Mar 2023 - Jan 2023\nUsed Docker.\n"
        "Lead\nBeta | Jan 2025 - Dec 2027\nUsed Go.\n"
        "Dev\nGamma | 2020 - 2021\nUsed Rust lang.\n"
    )
    metadata = extract(doc)
    assert metadata.warnings == (
        "Engineer: could not parse dates 'Mar 2023 - Jan 2023'",
        "Lead: end date after 2026-11-01; clamped",
    )
    assert metadata.skill_months["Docker"] == 0  # unparseable dates give no months
    assert metadata.skill_months["Go"] == 23  # Jan 2025 .. Nov 2026
    assert metadata.skill_months["Rust"] == 24


def test_round_trip(doc_metadata):
    data = doc_metadata.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert CandidateMetadata.from_dict(json.loads(json.dumps(data))) == doc_metadata
    assert "test@example.com" not in json.dumps(data)


# --- Cross-check against the reference matcher --------------------------------------------


@pytest.fixture(scope="module")
def parsed() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = {c["candidate_id"]: c for c in json.load(handle)["candidates"]}
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WORKSPACE_ROOT", str(ROOT / "data"))
        for persona in PERSONAS:
            path = manifest[persona.id]["resume_path"]
            result = fs_tools.read_file(path)
            assert result["success"], result["error"]
            out[persona.id] = (path, parse_resume(result["data"]["content"]))
    return out


def source_units(parsed) -> list[tuple[str, str, str]]:
    """Every role body, skills section and project section, as (persona, where, text)."""
    units = []
    for pid, (_, resume) in parsed.items():
        for section in resume.sections:
            if section.kind in (SectionKind.SKILLS, SectionKind.PROJECTS):
                units.append((pid, section.kind.value, "\n".join(section.lines)))
            elif section.kind is SectionKind.EXPERIENCE:
                for i, role in enumerate(split_roles(section)):
                    units.append((pid, f"role {i}", "\n".join(role.lines)))
    return units


def test_cross_check_with_reference_matcher_on_resumes(parsed):
    # The reference matcher was only ever run on single-line persona text, so both sides get
    # the text with whitespace collapsed (line wraps undone).
    units = source_units(parsed)
    assert len(units) == 117  # 79 role bodies + 34 skills sections + 4 project sections
    disagreements = [
        (pid, where, mentioned_skills(text) ^ reference_mentions(collapse(text)))
        for pid, where, text in units
        if mentioned_skills(text) != reference_mentions(collapse(text))
    ]
    assert disagreements == []


def test_cross_check_with_reference_matcher_on_jds():
    for path in sorted((ROOT / "data" / "job_descriptions").glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        assert mentioned_skills(text) == reference_mentions(text), path.name
        for line in text.splitlines():
            assert mentioned_skills(line) == reference_mentions(line), (path.name, line)


def test_raw_wrap_cases_are_the_only_raw_differences(parsed):
    # Documents the one known difference: on raw PDF text the reference matcher misses a
    # multi-word name wrapped across lines. Collapsing whitespace removes it.
    raw = sorted(
        (pid, where, tuple(sorted(mentioned_skills(text) - reference_mentions(text))))
        for pid, where, text in source_units(parsed)
        if mentioned_skills(text) != reference_mentions(text)
    )
    assert raw == [
        ("fatima_al_sayed", "skills", ("Machine Learning",)),
        ("john_doe", "skills", ("GitHub Actions",)),
    ]


# --- Corpus against the manifest ----------------------------------------------------------


@pytest.fixture(scope="module")
def corpus(parsed) -> dict[str, CandidateMetadata]:
    return {pid: extract_metadata(path, resume, AS_OF) for pid, (path, resume) in parsed.items()}


@pytest.fixture(scope="module")
def manifest() -> dict[str, dict]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return {c["candidate_id"]: c for c in json.load(handle)["candidates"]}


@any_persona
def test_skills_and_years_match_manifest(persona, corpus, manifest):
    found = corpus[persona.id]
    assert list(found.skills) == manifest[persona.id]["skills"]
    assert found.skill_years == manifest[persona.id]["skill_years"]
    assert found.warnings == ()


@any_persona
def test_corpus_round_trip(persona, corpus):
    metadata = corpus[persona.id]
    data = json.loads(json.dumps(metadata.to_dict()))
    assert CandidateMetadata.from_dict(data) == metadata
    assert persona.email not in json.dumps(data) and persona.phone not in json.dumps(data)


def test_named_traps(corpus):
    assert corpus["peter_hughes"].skills == ("Excel",)
    hannah = set(corpus["hannah_berg"].skills)
    assert not hannah & {"PyTorch", "Machine Learning", "Deep Learning"}
    omar = corpus["omar_haddad"]
    assert {"Python", "Django"} <= set(omar.skills)
    assert omar.skill_months["Python"] == omar.skill_months["Django"] == 0
    noah = corpus["noah_williams"]
    assert {"PostgreSQL", "Kubernetes", "JavaScript", "Machine Learning"} <= set(noah.skills)
    nina = corpus["nina_kowalski"]
    assert {"SQL", "Spark", "Airflow"} <= set(nina.skills)
    assert all(not loc.startswith(("skills", "projects")) for loc in nina.skill_evidence["Spark"])
    arjun = corpus["arjun_nair"]
    assert {"PySpark", "Spark", "Python"} <= set(arjun.skills)
    assert arjun.skill_months["Spark"] == arjun.skill_months["PySpark"] > 0
    assert "Java" not in corpus["emily_clarke"].skills
    assert corpus["mateo_garcia"].skill_years["Python"] == 4.3


@pytest.mark.parametrize("pid", ["elena_rodriguez", "grace_kim"])
def test_skills_tables_give_exactly_the_skills_section(pid, parsed):
    persona = next(p for p in PERSONAS if p.id == pid)
    _, resume = parsed[pid]
    (skills,) = resume.of_kind(SectionKind.SKILLS)
    assert mentioned_skills("\n".join(skills.lines)) == set(persona.skills_section)


def test_kenji_lowercase_python(parsed, corpus):
    _, resume = parsed["kenji_tanaka"]
    (skills,) = resume.of_kind(SectionKind.SKILLS)
    assert "python" in skills.lines[0].split(", ")
    assert "Python" in corpus["kenji_tanaka"].skills
    assert corpus["kenji_tanaka"].skill_evidence["Python"][0] == "skills"
