"""rag.requirements_check: must-haves, year boundaries, unmet strings, and qrels agreement."""

import json

import pytest
from match_helpers import PY_JD, meta

from rag.jd_parser import parse_job_description
from rag.metadata import CandidateMetadata
from rag.requirements_check import check_candidate
from scripts.persona_schema import ROOT

DEVOPS = parse_job_description(
    "SRE\n\nMust have\n- A minimum of 4 years in infrastructure roles\n- Kubernetes\n"
)


def test_all_met():
    check = check_candidate(
        meta(skills={"Python": 85, "FastAPI": 52, "PostgreSQL": 85, "Docker": 20}), PY_JD
    )
    assert check.eligible and check.unmet == ()
    python, web, db = check.results
    assert (python.skill, python.have_months, python.need_months) == ("Python", 85, 60)
    assert (web.skill, web.need_months, web.credit) == ("FastAPI", None, 1.0)
    assert db.met and db.credit == 1.0
    assert check.nice_present == ("Docker",)
    assert check.nice_missing == ("AWS", "Celery", "Redis")


def test_any_of_takes_any_listed_skill():
    for skills in ({"Django": 1}, {"FastAPI": 1}, {"Django": 1, "FastAPI": 2}):
        web = check_candidate(meta(skills=skills), PY_JD).results[1]
        assert web.met and web.skill in skills
    web = check_candidate(meta(skills={"Flask": 30}), PY_JD).results[1]
    assert not web.met and web.skill is None and web.credit == 0.0


@pytest.mark.parametrize(
    ("months", "met", "unmet", "credit"),
    [
        (59, False, "Python 5+ yrs (has 4.9)", 59 / 60),
        (60, True, None, 1.0),
        (61, True, None, 1.0),
        (0, False, "Python 5+ yrs (has 0.0)", 0.0),
    ],
)
def test_years_boundary(months, met, unmet, credit):
    python = check_candidate(meta(skills={"Python": months}), PY_JD).results[0]
    assert python.met is met
    assert python.unmet_text == unmet
    assert python.credit == pytest.approx(credit)


def test_any_of_with_years_uses_the_skill_that_meets_it():
    job = parse_job_description("X\n\nRequirements\n- 3+ years of Django or FastAPI\n")
    result = check_candidate(meta(skills={"Django": 20, "FastAPI": 40}), job).results[0]
    assert result.met and result.skill == "FastAPI"
    result = check_candidate(meta(skills={"Django": 20, "FastAPI": 30}), job).results[0]
    assert not result.met and result.skill == "FastAPI"  # the best present one
    assert result.unmet_text == "Django or FastAPI 3+ yrs (has 2.5)"


def test_total_years_checked_first():
    check = check_candidate(meta(skills={"Kubernetes": 10}, total_months=47), DEVOPS)
    assert not check.eligible
    assert check.unmet == ("4+ total yrs (has 3.9)",)
    total = check.total_result
    assert total.is_total and total.credit == pytest.approx(47 / 48)
    assert check_candidate(meta(skills={"Kubernetes": 10}, total_months=48), DEVOPS).eligible


def test_unmet_strings_and_order():
    check = check_candidate(meta(skills={"Python": 52}, total_months=40), PY_JD)
    assert check.unmet == (
        "Python 5+ yrs (has 4.3)",
        "Django or FastAPI (missing)",
        "PostgreSQL (missing)",
    )
    assert check_candidate(meta(skills={}), PY_JD).unmet[0] == "Python 5+ yrs (missing)"


def test_no_must_haves_is_eligible():
    job = parse_job_description("Generalist\n\nThe role\nAnything.\n")
    check = check_candidate(meta(), job)
    assert check.eligible and check.results == ()


def test_role_labels_and_identity():
    check = check_candidate(meta(), PY_JD)
    assert check.role_labels == ("Senior Backend Engineer at Harbourline Freight",)
    assert (check.candidate_id, check.name, check.resume_path) == (
        "noah",
        "Noah",
        "resumes/noah.txt",
    )


def test_agrees_with_qrels_for_all_pairs(fake_corpus):
    qrels = json.loads((ROOT / "data" / "ground_truth" / "qrels.json").read_text("utf-8"))
    metas = {
        cid: CandidateMetadata.from_dict(record.metadata)
        for cid, record in fake_corpus.sidecar.candidates.items()
    }
    pairs = 0
    for jd in qrels["jds"]:
        job = parse_job_description((ROOT / "data" / jd["jd_path"]).read_text("utf-8"))
        for judgement in jd["judgements"]:
            check = check_candidate(metas[judgement["candidate_id"]], job)
            assert check.eligible == judgement["eligible"], (jd["jd_id"], judgement)
            assert list(check.unmet) == judgement["unmet"], (jd["jd_id"], judgement)
            pairs += 1
    assert pairs == 210
