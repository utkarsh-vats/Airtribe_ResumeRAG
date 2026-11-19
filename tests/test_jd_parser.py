"""rag.jd_parser: sections, requirement units, years, grouping, fallbacks and the six JDs."""

import json
import re

import pytest
from jd_sections import HEADER, NICE_TO_HAVE, OTHER, REQUIREMENTS, split_jd

from rag.jd_parser import (
    HEADINGS,
    NO_REQUIREMENTS_WARNING,
    JDSectionKind,
    JobDescriptionError,
    Requirement,
    heading_kind,
    load_job_description,
    parse_job_description,
    skill_groups,
    split_sentences,
    units,
    years_in,
)
from scripts.persona_schema import ROOT

JD_DIR = ROOT / "data" / "job_descriptions"
THEMES = json.loads((ROOT / "data" / "ground_truth" / "jd_themes.json").read_text("utf-8"))
K = JDSectionKind


def jd(requirements: str = "", nice: str = "", extra: str = "") -> str:
    parts = ["Backend Engineer", "Acme - Remote", ""]
    if extra:
        parts += [extra, ""]
    if requirements:
        parts += ["Requirements", requirements, ""]
    if nice:
        parts += ["Nice to have", nice, ""]
    return "\n".join(parts)


def must(text: str) -> list[tuple[tuple[str, ...], int | None]]:
    return [(r.any_of, r.min_years) for r in parse_job_description(text).must_have]


# --- Years --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phrase", "years"),
    [
        ("5+ years of Python", 5),
        ("at least 3 years of Python", 3),
        ("5 or more years of Python", 5),
        ("A minimum of 4 years with Python", 4),
        ("Python (3+ yrs) in production", 3),
        ("3 years' experience with Python", 3),
        ("2 year of Python", 2),
        ("Python, 3-5 years", 3),
        ("five years of Python", 5),
    ],
)
def test_every_years_pattern(phrase, years):
    assert years_in(phrase) == [years]
    assert must(jd(f"- {phrase}")) == [(("Python",), years)]


def test_no_years_in_other_numbers():
    assert years_in("Python 3.12, 30 days of holiday, 2,000 euros, 18,000 shops") == []


def test_years_without_a_skill_set_total_experience():
    req = parse_job_description(jd("- A minimum of 4 years in DevOps, SRE or infrastructure roles"))
    assert req.must_have == () and req.min_total_years == 4
    assert req.unparsed_requirements == ()


def test_years_apply_to_each_skill_in_an_and_unit():
    assert must(jd("- 4+ years with Kubernetes and Terraform")) == [
        (("Kubernetes",), 4),
        (("Terraform",), 4),
    ]


def test_largest_years_used_with_a_warning():
    req = parse_job_description(jd("- 3+ years of Python, ideally 5 years"))
    assert [(r.any_of, r.min_years) for r in req.must_have] == [(("Python",), 5)]
    assert any("several year figures" in w for w in req.warnings)


# --- Grouping -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "groups"),
    [
        ("Production experience with Django or FastAPI.", [["Django", "FastAPI"]]),
        ("Django, FastAPI or Flask", [["Django", "FastAPI", "Flask"]]),
        ("Django, FastAPI, or Flask", [["Django", "FastAPI", "Flask"]]),
        ("Java/Kotlin services", [["Java", "Kotlin"]]),
        ("Kubernetes and Terraform", [["Kubernetes"], ["Terraform"]]),
        ("Kubernetes, Terraform and AWS", [["Kubernetes"], ["Terraform"], ["AWS"]]),
        ("Kubernetes, Terraform, AWS", [["Kubernetes"], ["Terraform"], ["AWS"]]),
        ("Python and Django or FastAPI", [["Python"], ["Django", "FastAPI"]]),
        ("Docker; we also use Redis or Celery", [["Docker"], ["Redis", "Celery"]]),
    ],
)
def test_skill_groups(unit, groups):
    assert skill_groups(unit) == groups


def test_any_of_vs_separate_must_haves():
    assert must(jd("- Django or FastAPI\n- PostgreSQL and Redis")) == [
        (("Django", "FastAPI"), None),
        (("PostgreSQL",), None),
        (("Redis",), None),
    ]


def test_aliases_are_canonical_and_not_implied():
    req = parse_job_description(jd("- Postgres and k8s\n- Django"))
    assert req.critical_skills == ("PostgreSQL", "Kubernetes", "Django")  # no Python, no SQL


def test_duplicate_requirements_merge_keeping_the_largest_years():
    req = parse_job_description(jd("- Python\n- 5+ years of Python"))
    assert [(r.any_of, r.min_years, r.source_text) for r in req.must_have] == [
        (("Python",), 5, "Python")
    ]


# --- Units --------------------------------------------------------------------------------


def test_bullets_are_units_and_wrapped_lines_join():
    assert units(
        ["- First item", "  continues here.", "* Second", "\u2022 Third", "1. Fourth"]
    ) == [
        "First item continues here.",
        "Second",
        "Third",
        "Fourth",
    ]


def test_prose_is_split_into_sentences():
    prose = (
        "You write SQL fluently. You have built batch jobs on Apache Spark, and you have\n"
        "scheduled pipelines with Apache Airflow. You care about quality, e.g. tests!"
    )
    assert units(prose.splitlines()) == [
        "You write SQL fluently.",
        "You have built batch jobs on Apache Spark, and you have scheduled pipelines with "
        "Apache Airflow.",
        "You care about quality, e.g. tests!",
    ]
    assert split_sentences("Use Python 3.12. Then Go.") == ["Use Python 3.12.", "Then Go."]


def test_prose_requirements():
    req = parse_job_description(
        jd("You write SQL. You run Apache Spark, and Apache Airflow. You care about quality.")
    )
    assert [r.any_of for r in req.must_have] == [("SQL",), ("Spark",), ("Airflow",)]
    assert req.unparsed_requirements == ("You care about quality.",)


def test_unparsed_soft_requirements_are_kept():
    req = parse_job_description(
        jd("- Python\n- Comfortable owning a service end to end, from design to production.")
    )
    assert req.unparsed_requirements == (
        "Comfortable owning a service end to end, from design to production.",
    )


# --- Sources and traps --------------------------------------------------------------------


def test_years_and_skills_outside_requirement_sections_are_ignored():
    text = (
        "Platform Engineer\nAcme - Dublin\n\n"
        "About us\nWe've been around for 12 years and use Java everywhere.\n\n"
        "The role\nYou'll work with our React team for 2 years at least.\n\n"
        "Requirements\n- Kubernetes\n\n"
        "Benefits\n- 30 days of paid holiday\n- A 5 year Python training plan\n"
    )
    req = parse_job_description(text)
    assert [(r.any_of, r.min_years) for r in req.must_have] == [(("Kubernetes",), None)]
    assert req.min_total_years is None and req.nice_to_have == ()


def test_nice_to_have_contributes_skills_only():
    req = parse_job_description(
        jd(
            "- Python",
            "- Next.js or another server-rendering framework.\n- 3+ years of Redux\n- Helm",
        )
    )
    assert req.nice_to_have == ("Helm", "Next.js", "Redux")
    assert req.min_total_years is None and [r.any_of for r in req.must_have] == [("Python",)]


def test_nice_to_have_drops_must_have_skills():
    req = parse_job_description(jd("- Kubernetes", "- Kubernetes operators\n- Helm"))
    assert req.nice_to_have == ("Helm",)


def test_ambiguous_skills_in_jds():
    req = parse_job_description(
        jd(
            "- Apache Spark at scale\n- Experience with spark plugs",
            "- Some Go, for operators or small internal tools.\n- Go-getter attitude\n- go fast",
        )
    )
    assert [r.any_of for r in req.must_have] == [("Spark",)]
    assert req.unparsed_requirements == ("Experience with spark plugs",)
    assert "Go" in req.nice_to_have  # exact case only; "Go-getter" is a known false positive
    assert any("go fast" in w for w in req.warnings)


def test_no_requirements_section_fallback():
    text = (
        "Data Person\nAcme\n\nAbout us\nWe love Java and have done for 10 years.\n\n"
        "The role\nYou'll build pipelines with Apache Spark.\n\n"
        "Nice to have\n- 5+ years of Python\n\nBenefits\n- Free Rust lessons\n"
    )
    req = parse_job_description(text)
    assert req.must_have == () and req.critical_skills == ()
    assert req.min_total_years is None
    assert req.nice_to_have == ("Python", "Spark")  # not Java (about) or Rust (benefits)
    assert NO_REQUIREMENTS_WARNING in req.warnings
    assert not req.has_requirements_section


def test_text_without_headings():
    req = parse_job_description("Python developer\nWe need Django and Postgres, 5+ years.")
    assert req.must_have == () and req.nice_to_have == ("Django", "PostgreSQL", "Python")
    assert set(req.warnings) == {"no section headings found", NO_REQUIREMENTS_WARNING}
    assert req.query_text == "Python developer\nWe need Django and Postgres, 5+ years."


def test_query_text_excludes_about_and_benefits():
    text = (
        "Backend Engineer\nAcme - Leeds\n\nAbout us\nWe insure homes.\n\n"
        "The role\nOwn the claims API.\n\nWhat you'll do\n- Build services.\n\n"
        "Requirements\n- Python\n\nNice to have\n- Docker\n\nBenefits\n- 28 days of holiday\n"
    )
    req = parse_job_description(text)
    assert req.query_text == (
        "Backend Engineer\n\nOwn the claims API.\n\n- Build services.\n\n- Python\n\n- Docker"
    )
    assert "insure" not in req.query_text and "holiday" not in req.query_text
    assert "Acme - Leeds" not in req.query_text
    assert req.full_text == text


def test_title_is_first_non_empty_line():
    assert parse_job_description("\n\n  Staff Engineer  \nAcme\n").title == "Staff Engineer"


def test_requirements_without_skills_warn():
    req = parse_job_description(jd("- Good communication"))
    assert req.must_have == () and "requirements section names no known skill or years" in (
        req.warnings
    )


def test_to_dict():
    data = parse_job_description(jd("- Django or FastAPI")).to_dict()
    assert data["must_have"] == [
        {"any_of": ["Django", "FastAPI"], "min_years": None, "source_text": "Django or FastAPI"}
    ]
    assert "full_text" not in data and "lines" not in data["sections"][0]
    assert "full_text" in parse_job_description("x").to_dict(include_text=True)
    assert Requirement(("A",), 1, "s").to_dict()["any_of"] == ["A"]


# --- Headings -----------------------------------------------------------------------------


def _readme_headings() -> dict[str, list[str]]:
    readme = (JD_DIR / "README.md").read_text(encoding="utf-8")
    rows = {}
    for label in ("Requirements (must-haves)", "Nice-to-haves", "Everything else"):
        row = next(line for line in readme.splitlines() if line.startswith(f"| {label} |"))
        rows[label] = re.findall(r"`([^`]+)`", row.split("|")[2])
    return rows


def test_heading_vocabulary_covers_the_readme():
    rows = _readme_headings()
    assert sum(len(v) for v in rows.values()) == 21
    for heading in rows["Requirements (must-haves)"]:
        assert heading_kind(heading) is K.REQUIREMENTS, heading
    for heading in rows["Nice-to-haves"]:
        assert heading_kind(heading) is K.NICE_TO_HAVE, heading
    for heading in rows["Everything else"]:
        assert heading_kind(heading) in {
            K.ABOUT,
            K.OVERVIEW,
            K.RESPONSIBILITIES,
            K.BENEFITS,
        }, heading
    listed = {h for headings in HEADINGS.values() for h in headings}
    assert {h for v in rows.values() for h in v} <= listed


def test_heading_matching_rules():
    assert heading_kind("  what YOU'LL need:  ") is K.REQUIREMENTS
    assert heading_kind("Requirements::") is None  # one trailing colon only
    assert heading_kind("Requirements for the role") is None


# --- Loading ------------------------------------------------------------------------------


def test_load_from_workspace_path(monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(ROOT / "data"))
    text, origin = load_job_description("job_descriptions/devops_sre.txt")
    assert origin == "job_descriptions/devops_sre.txt"
    assert text.startswith("DevOps / Site Reliability Engineer")


def test_load_docx_jd(tmp_path, monkeypatch):
    import docx

    document = docx.Document()
    for line in ("Backend Engineer", "Requirements", "- Django or FastAPI"):
        document.add_paragraph(line)
    document.save(tmp_path / "jd.docx")
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    text, origin = load_job_description("jd.docx")
    assert origin == "jd.docx"
    assert must(text) == [(("Django", "FastAPI"), None)]


def test_load_literal_text():
    assert load_job_description("Python developer with Django") == (
        "Python developer with Django",
        "<text>",
    )
    multi = "Title\nRequirements\n- foo.txt"
    assert load_job_description(multi) == (multi, "<text>")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("job_descriptions/missing.txt", "FILE_NOT_FOUND"),
        ("../outside.txt", "PATH"),
        ("   ", "empty"),
    ],
)
def test_load_errors(source, message, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(ROOT / "data"))
    with pytest.raises(JobDescriptionError, match=message):
        load_job_description(source)


# --- The six JDs --------------------------------------------------------------------------

theme_param = pytest.mark.parametrize("theme", THEMES, ids=[t["id"] for t in THEMES])


def parsed(theme):
    return parse_job_description((JD_DIR / f"{theme['id']}.txt").read_text(encoding="utf-8"))


@theme_param
def test_corpus_requirements_match_themes(theme):
    req = parsed(theme)
    assert req.title == theme["title"]
    assert [{"any_of": list(r.any_of), "min_years": r.min_years} for r in req.must_have] == theme[
        "must_have"
    ]
    assert req.min_total_years == theme["min_total_years"]
    assert sorted(req.nice_to_have) == sorted(theme["nice_to_have"])
    assert req.critical_skills == tuple(s for r in theme["must_have"] for s in r["any_of"])
    assert req.warnings == ()


@theme_param
def test_corpus_sections_agree_with_reference_splitter(theme):
    text = (JD_DIR / f"{theme['id']}.txt").read_text(encoding="utf-8")
    coarse = {K.REQUIREMENTS: REQUIREMENTS, K.NICE_TO_HAVE: NICE_TO_HAVE}
    ours = [
        (HEADER if s.heading is None else coarse.get(s.kind, OTHER), s.heading, s.text)
        for s in parse_job_description(text).sections
    ]
    assert ours == [(s.kind, s.heading, s.text) for s in split_jd(text)]


@theme_param
def test_corpus_query_text(theme):
    req = parsed(theme)
    assert req.query_text.startswith(theme["title"] + "\n\n")
    for section in req.sections:
        if section.kind in (K.ABOUT, K.BENEFITS, K.OTHER):
            assert section.text not in req.query_text
        else:
            assert section.text in req.query_text


def test_corpus_unparsed_requirements():
    assert {t["id"]: parsed(t).unparsed_requirements for t in THEMES} == {
        "python_backend": (
            "Comfortable owning a service end to end, from design document to production.",
        ),
        "ml_engineer": ("Clear writing: you'll explain model behaviour to doctors.",),
        "devops_sre": (),
        "frontend": ("An eye for accessibility, layout and detail.",),
        "java_backend": ("A habit of writing tests and readable code.",),
        "data_engineer": (
            "You care about data quality, you test your transformations, and you document "
            "what you build so the next person doesn't have to guess.",
        ),
    }
