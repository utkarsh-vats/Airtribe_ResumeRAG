"""rag.matcher: strict filter-then-retrieve, soft mode, notes, warnings, k and tie-breaks."""

import json
from dataclasses import replace
from datetime import date

import pytest

import rag.matcher as matcher
from rag.jd_parser import parse_job_description
from rag.matcher import FilterMode, match
from scripts.persona_schema import ROOT

QRELS = json.loads((ROOT / "data" / "ground_truth" / "qrels.json").read_text("utf-8"))
AS_OF = date(2026, 11, 1)


def jd(jd_id):
    return parse_job_description(
        (ROOT / "data" / "job_descriptions" / f"{jd_id}.txt").read_text("utf-8")
    )


def judgements(jd_id):
    entry = next(j for j in QRELS["jds"] if j["jd_id"] == jd_id)
    return {j["candidate_id"]: j for j in entry["judgements"]}


def run(fake_corpus, job, **kwargs):
    kwargs.setdefault("as_of", AS_OF)
    return match(
        job,
        fake_corpus.store,
        fake_corpus.keyword_index,
        fake_corpus.embedder,
        fake_corpus.sidecar,
        **kwargs,
    )


@pytest.fixture
def spy(monkeypatch):
    calls = []
    original = matcher.retrieve

    def wrapper(*args, candidate_ids=None, **kwargs):
        calls.append(sorted(candidate_ids))
        return original(*args, candidate_ids=candidate_ids, **kwargs)

    monkeypatch.setattr(matcher, "retrieve", wrapper)
    return calls


@pytest.mark.parametrize("jd_id", ["python_backend", "devops_sre", "data_engineer"])
def test_strict_retrieves_over_exactly_the_eligible(fake_corpus, spy, jd_id):
    truth = judgements(jd_id)
    eligible = sorted(c for c, j in truth.items() if j["eligible"])
    result = run(fake_corpus, jd(jd_id), k=35)
    assert spy == [eligible]
    assert {m.candidate_id for m in result.matches} <= set(eligible)
    assert all(m.check.eligible and m.unmet == () for m in result.matches)
    assert result.filtered_count == 35 - len(eligible) == len(result.filtered)
    assert result.filtered == {c: tuple(j["unmet"]) for c, j in truth.items() if not j["eligible"]}
    assert result.candidates_total == 35 and result.filter_mode is FilterMode.STRICT


def test_soft_scores_everyone_and_keeps_unmet(fake_corpus, spy):
    truth = judgements("python_backend")
    result = run(fake_corpus, jd("python_backend"), filter_mode="soft", k=35)
    assert spy == [sorted(truth)]
    assert result.filtered_count == 0 and result.filtered == {}
    ineligible = [m for m in result.matches if not m.check.eligible]
    assert ineligible
    for m in ineligible:
        assert list(m.unmet) == truth[m.candidate_id]["unmet"]
        assert "Misses: " in m.reasoning


def test_no_must_have_jd_runs_like_soft(fake_corpus, spy):
    job = parse_job_description("Data Person\n\nThe role\nApache Spark and dbt pipelines.\n")
    result = run(fake_corpus, job)
    assert any("no must-haves" in note for note in result.notes)
    assert result.filtered_count == 0 and len(spy[0]) == 35


def test_nobody_eligible(fake_corpus, spy):
    job = parse_job_description("Veteran\n\nRequirements\n- 40+ years of Python\n")
    result = run(fake_corpus, job)
    assert result.matches == () and result.filtered_count == 35 and spy == []
    assert "no candidate meets every must-have" in result.notes


def test_warnings_for_as_of_and_missing_calibration(fake_corpus):
    result = run(fake_corpus, jd("frontend"), as_of=date(2027, 1, 1))
    assert any("as_of 2026-11-01, not 2027-01-01" in w for w in result.warnings)
    assert any("no semantic calibration" in w for w in result.warnings)
    assert not any("as_of" in w for w in run(fake_corpus, jd("frontend")).warnings)


def test_k_truncation_and_order(fake_corpus):
    result = run(fake_corpus, jd("python_backend"), filter_mode="soft", k=3)
    assert len(result.matches) == 3 and 3 <= result.scored_count <= 35
    scores = [m.match_score for m in run(fake_corpus, jd("java_backend"), k=35).matches]
    assert scores == sorted(scores, reverse=True)
    with pytest.raises(ValueError):
        run(fake_corpus, jd("java_backend"), k=0)


def test_ties_broken_by_retrieval_rank_then_id(fake_corpus, monkeypatch):
    original = matcher.score_candidate

    def flat(*args, **kwargs):
        return replace(original(*args, **kwargs), match_score=50)

    monkeypatch.setattr(matcher, "score_candidate", flat)
    result = run(fake_corpus, jd("python_backend"), filter_mode="soft", k=35)
    assert [m.retrieval_rank for m in result.matches] == list(range(1, len(result.matches) + 1))
    assert [m.candidate_id for m in result.matches] == [
        h.candidate_id for h in result.retrieval.hits
    ]


def test_result_details(fake_corpus):
    job = jd("python_backend")
    result = run(fake_corpus, job)
    assert result.not_assessed == job.unparsed_requirements
    t = result.timings
    assert t.filter > 0 and t.scoring > 0 and t.retrieval.embed_query >= 0
    assert t.total >= t.filter + t.scoring
    first = result.matches[0]
    assert first.reasoning.startswith("Meets all 3 must-haves: ")
    assert 0 <= first.match_score <= 100


def test_warning_when_calibration_percentiles_differ(fake_corpus):
    from rag.calibration import Calibration, calibration_settings
    from rag.retrieval import RetrievalConfig

    def sidecar_with(floor_pct, ceiling_pct):
        cal = Calibration(
            0.0,
            1.0,
            "collection",
            retrieval=calibration_settings(RetrievalConfig()),
            floor_pct=floor_pct,
            ceiling_pct=ceiling_pct,
        )
        return replace(fake_corpus.sidecar, calibration=cal.to_dict())

    def warnings(sidecar, **kwargs):
        return match(
            jd("frontend"),
            fake_corpus.store,
            fake_corpus.keyword_index,
            fake_corpus.embedder,
            sidecar,
            as_of=AS_OF,
            **kwargs,
        ).warnings

    assert warnings(sidecar_with(50.0, 99.0)) == ()
    assert warnings(sidecar_with(50.0, 95.0)) == (
        "the calibration used percentiles 50/95, not 50/99; re-run calibrate",
    )
    assert warnings(sidecar_with(50.0, 95.0), ceiling_pct=95) == ()
    assert "unrecorded" in warnings(sidecar_with(None, None))[0]
