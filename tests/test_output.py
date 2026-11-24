"""rag.output: the brief's exact structure and the full format's extras."""

import json
from datetime import date

import pytest

from rag.jd_parser import parse_job_description
from rag.matcher import match
from rag.output import BRIEF_MATCH_KEYS, build_output
from scripts.persona_schema import ROOT

JD_PATH = ROOT / "data" / "job_descriptions" / "python_backend.txt"


@pytest.fixture(scope="module")
def job():
    return parse_job_description(JD_PATH.read_text("utf-8"))


def run(fake_corpus, job, filter_mode="strict"):
    return match(
        job,
        fake_corpus.store,
        fake_corpus.keyword_index,
        fake_corpus.embedder,
        fake_corpus.sidecar,
        filter_mode=filter_mode,
        as_of=date(2026, 11, 1),
    )


def test_brief_has_exactly_the_brief_keys(fake_corpus, job):
    out = build_output(run(fake_corpus, job), job, "brief")
    assert list(out) == ["job_description", "top_matches"]
    assert out["job_description"] == job.full_text
    assert out["top_matches"]
    for entry in out["top_matches"]:
        assert tuple(entry) == BRIEF_MATCH_KEYS
        assert isinstance(entry["match_score"], int) and 0 <= entry["match_score"] <= 100
        assert all(isinstance(e, str) for e in entry["relevant_excerpts"])
    json.dumps(out)


def test_full_adds_the_extras(fake_corpus, job):
    result = run(fake_corpus, job)
    out = build_output(result, job, source="job_descriptions/python_backend.txt", collection="c")
    assert list(out)[:2] == ["job_description", "top_matches"]
    assert set(out) == {
        "job_description",
        "top_matches",
        "job",
        "filter",
        "config",
        "notes",
        "warnings",
        "timings_ms",
    }
    entry = out["top_matches"][0]
    assert tuple(entry)[:6] == BRIEF_MATCH_KEYS
    assert set(entry) - set(BRIEF_MATCH_KEYS) == {"candidate_id", "score_breakdown"}
    assert set(entry["score_breakdown"]) == {"components", "weights"}
    assert out["job"]["source"] == "job_descriptions/python_backend.txt"
    assert out["job"]["must_have"][1] == {"any_of": ["Django", "FastAPI"], "min_years": None}
    assert out["job"]["not_assessed"] == list(job.unparsed_requirements)
    assert out["filter"]["mode"] == "strict"
    assert out["filter"]["eligible_count"] == 3 and out["filter"]["filtered_count"] == 32
    assert {f["candidate_id"] for f in out["filter"]["filtered"]} == set(result.filtered)
    assert out["config"]["collection"] == "c"
    assert out["config"]["retrieval"]["keyword_mode"] == "skills"
    assert out["config"]["calibration"]["source"] == "generic"
    assert any("no semantic calibration" in w for w in out["warnings"])
    assert set(out["timings_ms"]) == {"filter", "retrieval", "scoring", "total"}
    json.dumps(out)


def test_soft_full_output_keeps_unmet(fake_corpus, job):
    out = build_output(run(fake_corpus, job, "soft"), job)
    assert all("unmet" in entry for entry in out["top_matches"])
    assert any(entry["unmet"] for entry in out["top_matches"])
    assert out["filter"]["filtered"] == [] and out["filter"]["eligible_count"] == 3


def test_nobody_eligible_still_builds(fake_corpus):
    job = parse_job_description("Veteran\n\nRequirements\n- 40+ years of Python\n")
    out = build_output(run(fake_corpus, job), job)
    assert out["top_matches"] == [] and out["config"]["retrieval"]["mode"] == "hybrid"
