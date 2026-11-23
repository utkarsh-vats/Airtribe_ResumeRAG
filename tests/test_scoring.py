"""rag.scoring: components, dropping and renormalising, rounding, matched skills."""

import pytest
from match_helpers import PY_JD, hit, meta, top

from rag.calibration import Calibration
from rag.jd_parser import parse_job_description
from rag.requirements_check import check_candidate
from rag.scoring import ScoreWeights, renormalised, round_half_up, score_candidate

CAL = Calibration(0.60, 0.80, "collection")
NOAH = {"Python": 85, "FastAPI": 52, "PostgreSQL": 85, "Docker": 30, "AWS": 12}


def score(skills=NOAH, aggregate=0.7, job=PY_JD, total_months=85, weights=None):
    check = check_candidate(meta(skills=skills, total_months=total_months), job)
    return score_candidate(
        hit(aggregate=aggregate, chunks=[top("skills")]), check, job, CAL, weights
    )


@pytest.mark.parametrize(("aggregate", "expected"), [(0.5, 0.0), (0.7, 0.5), (0.95, 1.0)])
def test_semantic_component_is_calibrated_and_clamped(aggregate, expected):
    assert score(aggregate=aggregate).components["semantic"] == pytest.approx(expected)


def test_all_components_for_a_strong_candidate():
    match = score()
    assert match.components == pytest.approx(
        {"semantic": 0.5, "must_have": 1.0, "nice_to_have": 0.5, "experience": 1.0}
    )
    assert match.weights == pytest.approx(
        {"semantic": 0.40, "must_have": 0.30, "nice_to_have": 0.15, "experience": 0.15}
    )
    # 0.4*0.5 + 0.3*1 + 0.15*0.5 + 0.15*1 = 0.725
    assert match.match_score == 73 and isinstance(match.match_score, int)


def test_partial_credit_and_experience():
    match = score(skills={"Python": 30, "Django": 10})  # Python 2.5 of 5 yrs, no PostgreSQL
    assert match.components["must_have"] == pytest.approx((0.5 + 1 + 0) / 3)
    assert match.components["experience"] == pytest.approx(0.5)
    assert match.components["nice_to_have"] == 0.0
    assert match.unmet == ("Python 5+ yrs (has 2.5)", "PostgreSQL (missing)")


def test_total_years_count_in_must_have_and_experience():
    job = parse_job_description(
        "SRE\n\nMust have\n- A minimum of 4 years in infrastructure roles\n"
        "- 3+ years of Kubernetes\n\nPreferred\n- Helm\n"
    )
    match = score(skills={"Kubernetes": 18}, total_months=24, job=job)
    assert match.components["must_have"] == pytest.approx((0.5 + 0.5) / 2)
    assert match.components["experience"] == pytest.approx((0.5 + 0.5) / 2)


def test_components_dropped_and_weights_renormalised():
    job = parse_job_description("Dev\n\nRequirements\n- Python\n")  # no nice, no years
    match = score(job=job)
    assert set(match.components) == {"semantic", "must_have"}
    assert match.weights == pytest.approx({"semantic": 0.4 / 0.7, "must_have": 0.3 / 0.7})
    assert sum(match.weights.values()) == pytest.approx(1.0)
    assert match.match_score == round_half_up(100 * (0.4 / 0.7 * 0.5 + 0.3 / 0.7 * 1.0))


def test_no_requirements_jd_scores_semantic_and_nice_only():
    job = parse_job_description("Dev\n\nThe role\nPython and Docker work.\n")
    match = score(job=job)
    assert set(match.components) == {"semantic", "nice_to_have"}
    assert match.components["nice_to_have"] == 1.0


def test_rounding_is_half_up():
    assert [round_half_up(v) for v in (72.5, 73.5, 72.49, 0.5, 99.5)] == [73, 74, 72, 1, 100]
    assert round_half_up(100 * (0.2 + 0.3 + 0.075 + 0.15)) == 73  # 72.49999999999999


def test_weights_validation_and_custom_weights():
    with pytest.raises(ValueError):
        ScoreWeights(semantic=-0.1)
    with pytest.raises(ValueError):
        ScoreWeights(0, 0, 0, 0)
    with pytest.raises(ValueError, match="weight 0"):
        renormalised(ScoreWeights(1, 0, 0, 0), ["must_have"])
    match = score(weights=ScoreWeights(1, 0, 0, 0))
    assert match.match_score == 50  # semantic only


def test_matched_skills_order():
    match = score(skills={"Python": 85, "Django": 10, "FastAPI": 52, "AWS": 1, "Redis": 1})
    # must-haves in JD order (both any_of skills), then nice-to-haves in JD order, not sorted
    assert match.matched_skills == ("Python", "Django", "FastAPI", "Redis", "AWS")


def test_identity_and_passthrough():
    match = score()
    assert (match.candidate_id, match.name, match.resume_path) == (
        "noah",
        "Noah",
        "resumes/noah.txt",
    )
    assert match.matched_sections == ("skills",) and match.top_chunks[0].section == "skills"
    with pytest.raises(ValueError, match="hit is"):
        score_candidate(hit("other"), check_candidate(meta(), PY_JD), PY_JD, CAL)
