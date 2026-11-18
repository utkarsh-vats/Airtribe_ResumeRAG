"""The six JD files against jd_themes.json and the JD contract in their README."""

import itertools
import re
from pathlib import Path

import pytest
from jd_sections import HEADER, HEADINGS, NICE_TO_HAVE, OTHER, REQUIREMENTS, split_jd, text_of
from skill_mentions import find_mentions, mentioned_skills

from scripts.jd_themes import JDTheme, load_jd_themes
from scripts.persona_schema import ROOT

JD_DIR = ROOT / "data" / "job_descriptions"
THEMES = load_jd_themes()
BY_ID = {t.id: t for t in THEMES}
README = (JD_DIR / "README.md").read_text(encoding="utf-8")

# A year phrase: "5+ years", "3 years'", "4 years", "5 or more years", "(3+ yrs)".
YEARS = re.compile(r"(?<!\w)(\d+)\s*(?:\+|or more)?\s*(?:years?|yrs)(?!\w)")

theme_param = pytest.mark.parametrize("theme", THEMES, ids=[t.id for t in THEMES])


def jd_text(theme: JDTheme) -> str:
    return (JD_DIR / f"{theme.id}.txt").read_text(encoding="utf-8")


def sentences(text: str) -> list[str]:
    """Lines, further split into sentences."""
    return [s for line in text.splitlines() for s in re.split(r"(?<=[.!?])\s+", line) if s]


def must_have_skills(theme: JDTheme) -> set[str]:
    return {skill for req in theme.must_have for skill in req.any_of}


def test_every_theme_has_a_jd_and_nothing_else():
    assert {p.stem for p in JD_DIR.glob("*.txt")} == set(BY_ID)


@theme_param
def test_jd_is_ascii_lf_and_200_to_450_words(theme: JDTheme):
    raw = (JD_DIR / f"{theme.id}.txt").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    raw.decode("ascii")
    assert 200 <= len(raw.split()) <= 450


def test_lengths_and_headings_vary():
    lengths = [len(jd_text(t).split()) for t in THEMES]
    assert max(lengths) - min(lengths) >= 100, lengths
    req, nice = set(), set()
    for theme in THEMES:
        for section in split_jd(jd_text(theme)):
            if section.kind == REQUIREMENTS:
                req.add(section.heading.removesuffix(":"))
            elif section.kind == NICE_TO_HAVE:
                nice.add(section.heading.removesuffix(":"))
    assert len(req) >= 4 and len(nice) >= 3, (req, nice)


@theme_param
def test_jd_structure(theme: JDTheme):
    sections = split_jd(jd_text(theme))
    kinds = [s.kind for s in sections]
    assert kinds[0] == HEADER and sections[0].text.splitlines()[0] == theme.title
    assert kinds.count(REQUIREMENTS) == 1 and kinds.count(NICE_TO_HAVE) == 1
    assert kinds.count(OTHER) >= 3  # company blurb, role / responsibilities, location / benefits
    assert all(s.text for s in sections), "empty section"


def test_readme_documents_every_heading():
    for headings in HEADINGS.values():
        for heading in headings:
            assert f"`{heading}`" in README, heading


@theme_param
def test_requirements_mention_exactly_the_must_haves(theme: JDTheme):
    text = text_of(split_jd(jd_text(theme)), REQUIREMENTS)
    mentioned = mentioned_skills(text)
    assert must_have_skills(theme) - mentioned == set(), "must-have not mentioned"
    assert mentioned - must_have_skills(theme) == set(), "extra skill in requirements"


@theme_param
def test_nice_to_have_mentions_exactly_nice_to_have(theme: JDTheme):
    text = text_of(split_jd(jd_text(theme)), NICE_TO_HAVE)
    assert mentioned_skills(text) == set(theme.nice_to_have)


@theme_param
def test_year_phrases_match_min_years(theme: JDTheme):
    # Every year phrase in the requirements belongs to the skill(s) in its sentence; a year
    # phrase in a sentence with no skill is the total-experience minimum.
    text = text_of(split_jd(jd_text(theme)), REQUIREMENTS)
    skill_years: dict[str, int] = {}
    total_years: list[int] = []
    for sentence in sentences(text):
        found = [int(n) for n in YEARS.findall(sentence)]
        if not found:
            continue
        assert len(found) == 1, sentence
        skills = mentioned_skills(sentence)
        if skills:
            skill_years.update(dict.fromkeys(skills, found[0]))
        else:
            total_years.append(found[0])
    expected = {s: r.min_years for r in theme.must_have if r.min_years for s in r.any_of}
    assert skill_years == expected
    assert total_years == ([theme.min_total_years] if theme.min_total_years else [])


@theme_param
def test_min_total_years_phrasing(theme: JDTheme):
    text = text_of(split_jd(jd_text(theme)), REQUIREMENTS)
    phrase = re.compile(r"\bA minimum of (\d+) years in [^.\n]+ roles\b")
    found = [int(n) for n in phrase.findall(text)]
    assert found == ([theme.min_total_years] if theme.min_total_years else [])


@theme_param
def test_any_of_groups_are_joined_by_or(theme: JDTheme):
    text = text_of(split_jd(jd_text(theme)), REQUIREMENTS)
    for req in theme.must_have:
        if len(req.any_of) < 2:
            continue
        for sentence in sentences(text):
            mentions = [m for m in find_mentions(sentence) if m.skill in req.any_of]
            if {m.skill for m in mentions} != set(req.any_of):
                continue
            gaps = [sentence[a.end : b.start] for a, b in itertools.pairwise(mentions)]
            assert all(re.fullmatch(r",?\s+or\s+|,\s+", g) for g in gaps), gaps
            assert gaps[-1].strip(", ").strip() == "or"
            break
        else:
            pytest.fail(f"{req.any_of} never appear together in one sentence")


@theme_param
def test_noise_skills_outside_the_requirement_sections(theme: JDTheme):
    outside = mentioned_skills(text_of(split_jd(jd_text(theme)), HEADER, OTHER))
    noise = outside - must_have_skills(theme) - set(theme.nice_to_have)
    assert len(noise) >= 2, noise


def test_data_engineer_requirements_are_prose():
    text = text_of(split_jd(jd_text(BY_ID["data_engineer"])), REQUIREMENTS)
    assert not any(line.lstrip().startswith(("-", "*")) for line in text.splitlines())


def test_reverse_check_catches_an_extra_requirement(tmp_path: Path):
    # Mutation check: a stray skill in the requirements is caught.
    theme = BY_ID["python_backend"]
    broken = jd_text(theme).replace(
        "- Production experience with Django or FastAPI.",
        "- Production experience with Django or FastAPI, deployed on Kubernetes.",
    )
    assert broken != jd_text(theme)
    mentioned = mentioned_skills(text_of(split_jd(broken), REQUIREMENTS))
    assert mentioned - must_have_skills(theme) == {"Kubernetes"}


def test_splitter_basics():
    sections = split_jd("Title\nCo\n\nAbout us\nHi.\nMust have:\n- Go\nperks\nCake\n")
    assert [(s.kind, s.heading) for s in sections] == [
        (HEADER, None),
        (OTHER, "About us"),
        (REQUIREMENTS, "Must have:"),
        (OTHER, "perks"),
    ]
    assert sections[2].text == "- Go"
