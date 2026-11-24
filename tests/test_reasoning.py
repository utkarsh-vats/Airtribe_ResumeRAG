"""rag.reasoning: the templated explanation."""

from match_helpers import PY_JD, hit, meta, top

from rag.calibration import Calibration
from rag.jd_parser import parse_job_description
from rag.reasoning import join_words, reasoning
from rag.requirements_check import check_candidate
from rag.scoring import score_candidate

CAL = Calibration(0.6, 0.8, "collection")
CHUNKS = [top("experience", 0), top("skills")]


def scored(skills, job=PY_JD, total_months=85, chunks=CHUNKS, roles=None):
    kwargs = {"roles": roles} if roles else {}
    check = check_candidate(meta(skills=skills, total_months=total_months, **kwargs), job)
    return score_candidate(hit(chunks=chunks), check, job, CAL)


def test_strict_match():
    match = scored({"Python": 85, "FastAPI": 52, "PostgreSQL": 85, "Docker": 30, "AWS": 12})
    assert reasoning(match, PY_JD) == (
        "Meets all 3 must-haves: Python 7.1 yrs (5+ required), FastAPI 4.3 yrs (Django or "
        "FastAPI), PostgreSQL 7.1 yrs. Has 2 of 4 nice-to-haves (Docker, AWS). 7.1 years' "
        "total experience. Strongest evidence in Experience (Senior Backend Engineer at "
        "Harbourline Freight) and Skills."
    )


def test_soft_match_with_misses():
    match = scored({"Python": 52, "FastAPI": 52}, total_months=52)
    assert reasoning(match, PY_JD, show_misses=True) == (
        "Meets 1 of 3 must-haves: FastAPI 4.3 yrs (Django or FastAPI). "
        "Misses: Python 5+ yrs (has 4.3); PostgreSQL (missing). "
        "Has none of the 4 nice-to-haves. 4.3 years' total experience. "
        "Strongest evidence in Experience (Senior Backend Engineer at Harbourline Freight) "
        "and Skills."
    )
    assert "Misses" not in reasoning(match, PY_JD)  # strict never shows them


def test_jd_without_nice_to_haves_and_with_total_years():
    job = parse_job_description(
        "SRE\n\nMust have\n- A minimum of 4 years in infrastructure roles\n- Kubernetes\n"
    )
    match = scored({"Kubernetes": 0}, job=job, total_months=60, chunks=[top("certifications")])
    assert reasoning(match, job) == (
        "Meets the must-have: Kubernetes (no counted experience). "
        "5.0 years' total experience (4+ required). Strongest evidence in Certifications."
    )


def test_no_must_haves_and_several_roles():
    job = parse_job_description("Dev\n\nThe role\nPython work.\n")
    roles = (("Engineer", "A"), ("Developer", "B"), ("Intern", "C"))
    chunks = [top("experience", 1), top("experience", 0), top("projects")]
    match = scored({"Python": 10}, job=job, total_months=10, chunks=chunks, roles=roles)
    assert reasoning(match, job) == (
        "Has 1 of 1 nice-to-have (Python). 0.8 years' total experience. Strongest evidence "
        "in Experience (Developer at B, Engineer at A) and Projects."
    )


def test_join_words():
    assert [join_words(x) for x in ([], ["a"], ["a", "b"], ["a", "b", "c"])] == [
        "",
        "a",
        "a and b",
        "a, b and c",
    ]


def test_self_employed_roles():
    roles = (("Freelance Python Developer", "Self-employed"), ("Backend Engineer", "Pampa"))
    chunks = [top("experience", 0), top("experience", 1)]
    match = scored({"Python": 85}, chunks=chunks, roles=roles)
    assert reasoning(match, PY_JD).endswith(
        "Strongest evidence in Experience (Freelance Python Developer (self-employed), "
        "Backend Engineer at Pampa)."
    )
    for company in ("Freelance", "self employed", "Freelancer"):
        check = check_candidate(meta(roles=(("Designer", company),)), PY_JD)
        assert check.role_labels == ("Designer (self-employed)",)
