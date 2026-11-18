"""qrels.json: built from persona_plan.md labels + manifest/jd_themes eligibility."""

import json

import pytest

from scripts.build_manifest import MANIFEST_PATH
from scripts.build_qrels import GRADES, QRELS_PATH, build_qrels, load_manifest, render_qrels
from scripts.jd_themes import load_jd_themes, unmet_details, unmet_must_haves
from scripts.persona_plan import load_plan
from scripts.persona_schema import ROOT

THEMES = {t.id: t for t in load_jd_themes()}


@pytest.fixture(scope="module")
def qrels() -> dict:
    with QRELS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def manifest() -> dict[str, dict]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return {c["candidate_id"]: c for c in json.load(handle)["candidates"]}


def judgements(qrels: dict):
    for jd in qrels["jds"]:
        for j in jd["judgements"]:
            yield jd["jd_id"], j


def test_qrels_is_up_to_date():
    expected = render_qrels(build_qrels(load_manifest()))
    assert QRELS_PATH.read_text(encoding="utf-8") == expected, (
        "qrels.json is stale; run: python -m scripts.build_qrels"
    )


def test_covers_every_jd_and_candidate(qrels, manifest):
    assert [jd["jd_id"] for jd in qrels["jds"]] == list(THEMES)
    for jd in qrels["jds"]:
        assert (ROOT / "data" / jd["jd_path"]).is_file()
        ids = [j["candidate_id"] for j in jd["judgements"]]
        assert ids == sorted(manifest)
        for j in jd["judgements"]:
            assert j["resume_path"] == manifest[j["candidate_id"]]["resume_path"]


def test_labels_and_grades_come_from_the_plan(qrels):
    plan = load_plan()
    short = {"strong": "S", "partial": "P", "distractor": "D", "none": "none"}
    for jd_id, j in judgements(qrels):
        assert j["label"] == short[plan[j["candidate_id"]].relevance[jd_id]]
        assert j["grade"] == GRADES[j["label"]]
    assert GRADES == {"S": 2, "P": 1, "D": 0, "none": 0}


def test_eligibility_uses_the_must_have_rule(qrels, manifest):
    for jd_id, j in judgements(qrels):
        candidate = manifest[j["candidate_id"]]
        assert j["eligible"] == (not unmet_must_haves(THEMES[jd_id], candidate))
        assert j["unmet"] == unmet_details(THEMES[jd_id], candidate)
        assert len(j["unmet"]) == len(unmet_must_haves(THEMES[jd_id], candidate))


def test_strong_implies_eligible_and_eligible_implies_relevant(qrels):
    for jd_id, j in judgements(qrels):
        if j["label"] == "S":
            assert j["eligible"], (jd_id, j["candidate_id"])
        if j["eligible"]:
            assert j["grade"] >= 1, (jd_id, j["candidate_id"])


def test_dataset_property_eligible_equals_strong(qrels):
    # Documented in data/ground_truth/README.md ("Relevance semantics"); evaluation relies on it.
    for jd_id, j in judgements(qrels):
        assert j["eligible"] == (j["label"] == "S"), (jd_id, j["candidate_id"])


def test_label_counts_per_jd(qrels):
    for jd in qrels["jds"]:
        labels = [j["label"] for j in jd["judgements"]]
        assert labels.count("S") >= 3 and labels.count("P") >= 3 and labels.count("D") >= 2


def test_unmet_details_wording():
    theme = THEMES["python_backend"]
    candidate = {
        "total_years": 6.0,
        "skills": ["Python", "SQL"],
        "skill_years": {"Python": 4.3, "SQL": 1.0},
    }
    assert unmet_details(theme, candidate) == [
        "Python 5+ yrs (has 4.3)",
        "Django or FastAPI (missing)",
        "PostgreSQL (missing)",
    ]
    devops = THEMES["devops_sre"]
    junior = {
        "total_years": 3.9,
        "skills": ["AWS", "Kubernetes"],
        "skill_years": {"AWS": 3.9, "Kubernetes": 3.9},
    }
    assert unmet_details(devops, junior) == ["4+ total yrs (has 3.9)", "Terraform (missing)"]
