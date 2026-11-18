from collections import Counter

import pytest

from rag.skills import (
    SKILLS,
    SKILLS_BY_NAME,
    Category,
    _build_lookup,
    _key,
    canonicalize,
    expand_implied,
    get_skill,
)
from rag.skills import Skill as S


def test_taxonomy_size_and_categories():
    assert 100 <= len(SKILLS) <= 150
    assert {skill.category for skill in SKILLS} == set(Category)


def test_canonical_names_unique():
    counts = Counter(_key(skill.name) for skill in SKILLS)
    assert [name for name, n in counts.items() if n > 1] == []


def test_every_form_maps_to_exactly_one_canonical():
    # Names and aliases share one namespace: no alias may repeat a name or another alias.
    counts = Counter(_key(form) for skill in SKILLS for form in (skill.name, *skill.aliases))
    assert [form for form, n in counts.items() if n > 1] == []
    for skill in SKILLS:
        for form in (skill.name, *skill.aliases):
            assert canonicalize(form) == skill.name


def test_build_lookup_rejects_collisions():
    with pytest.raises(ValueError, match="maps to both"):
        _build_lookup([S("Java", Category.LANGUAGE), S("JavaScript", Category.LANGUAGE, ("java",))])


def test_implies_targets_exist_and_are_not_self():
    for skill in SKILLS:
        for target in skill.implies:
            assert target in SKILLS_BY_NAME, f"{skill.name} implies unknown {target!r}"
            assert target != skill.name


def test_implies_has_no_cycles():
    for skill in SKILLS:
        assert skill.name not in expand_implied(skill.implies), f"cycle through {skill.name}"


def test_java_and_javascript_are_distinct():
    java, js = get_skill("Java"), get_skill("JavaScript")
    assert java is not js
    java_forms = {_key(f) for f in (java.name, *java.aliases)}
    js_forms = {_key(f) for f in (js.name, *js.aliases)}
    assert java_forms.isdisjoint(js_forms)
    assert canonicalize("java") == "Java"
    assert canonicalize("JS") == "JavaScript"
    assert "JavaScript" not in expand_implied(["Java", "Spring Boot", "JUnit"])
    assert "Java" not in expand_implied(["JavaScript", "TypeScript", "Node.js", "React"])


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("k8s", "Kubernetes"),
        ("K8S", "Kubernetes"),
        ("Postgres", "PostgreSQL"),
        ("psql", "PostgreSQL"),
        ("  postgresql ", "PostgreSQL"),
        ("js", "JavaScript"),
        ("ML", "Machine Learning"),
        ("machine   learning", "Machine Learning"),
        ("Golang", "Go"),
        ("sklearn", "scikit-learn"),
        ("c++", "C++"),
        ("c#", "C#"),
        ("Apache Spark", "Spark"),
    ],
)
def test_canonicalize_aliases_case_and_whitespace(raw, canonical):
    assert canonicalize(raw) == canonical


@pytest.mark.parametrize("raw", ["", "Javascripts", "CV", "Node", "Cobol"])
def test_canonicalize_unknown_returns_none(raw):
    assert canonicalize(raw) is None


def test_get_skill_unknown_raises():
    with pytest.raises(KeyError, match="Cobol"):
        get_skill("Cobol")


def test_ambiguous_flags():
    for name in ("Go", "R", "C", "Excel", "Spark", "Airflow"):
        assert get_skill(name).ambiguous, name
    for name in ("Python", "Java", "Kubernetes", "PostgreSQL"):
        assert not get_skill(name).ambiguous, name


@pytest.mark.parametrize(
    ("skills", "expected"),
    [
        (["Next.js"], {"Next.js", "React", "JavaScript"}),
        (["Spring Cloud"], {"Spring Cloud", "Spring Boot", "Java"}),
        (["Django REST Framework"], {"Django REST Framework", "Django", "Python"}),
        (["PySpark"], {"PySpark", "Spark", "Python"}),
        (["EKS"], {"EKS", "AWS", "Kubernetes"}),
        (["PyTorch"], {"PyTorch", "Python", "Deep Learning", "Machine Learning"}),
        (
            ["Large Language Models"],
            {"Large Language Models", "Natural Language Processing", "Machine Learning"},
        ),
        (["postgres", "MySQL"], {"PostgreSQL", "MySQL", "SQL"}),
        (["Kubernetes"], {"Kubernetes"}),
        ([], set()),
    ],
)
def test_expand_implied_is_transitive(skills, expected):
    assert expand_implied(skills) == expected


def test_expand_implied_is_idempotent_and_canonical():
    once = expand_implied(["express js", "k8s", "Tailwind"])
    assert expand_implied(once) == once
    assert once == {"Express.js", "Node.js", "JavaScript", "Kubernetes", "Tailwind CSS", "CSS"}


def test_expand_implied_unknown_raises():
    with pytest.raises(KeyError):
        expand_implied(["Python", "Cobol"])
