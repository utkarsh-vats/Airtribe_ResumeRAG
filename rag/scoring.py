"""The 0-100 match score for one retrieved candidate.

Four components, each 0-1:

- **semantic**: the candidate's semantic aggregate from retrieval, calibrated:
  ``clamp((aggregate - floor) / (ceiling - floor))`` (``rag.calibration``);
- **must_have**: mean partial credit over the must-haves and the total-years minimum
  (``rag.requirements_check``): 1 for a present skill, ``min(1, have/need)`` months for a
  year minimum, 0 for a missing skill;
- **nice_to_have**: the fraction of nice-to-haves present;
- **experience**: mean over every year requirement (per-skill and total) of
  ``min(1, have/need)``.

``match_score = round_half_up(100 x sum(w_i x c_i))`` with default weights 0.40 / 0.30 /
0.15 / 0.15. A component that doesn't apply (no must-haves, no nice-to-haves, no year
requirements) is dropped and the remaining weights are renormalised to sum to 1.
"""

import math
from dataclasses import dataclass, fields
from decimal import ROUND_HALF_UP, Decimal

from rag.calibration import Calibration
from rag.jd_parser import JobRequirements
from rag.requirements_check import RequirementCheck
from rag.retrieval import CandidateHit, TopChunk

COMPONENTS = ("semantic", "must_have", "nice_to_have", "experience")


@dataclass(frozen=True)
class ScoreWeights:
    """Component weights; they needn't sum to 1 (the used ones are renormalised)."""

    semantic: float = 0.40
    must_have: float = 0.30
    nice_to_have: float = 0.15
    experience: float = 0.15

    def __post_init__(self):
        """Weights must be finite and >= 0, and not all 0."""
        values = [getattr(self, f.name) for f in fields(self)]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"score weights must be finite and >= 0: {values}")
        if not any(values):
            raise ValueError("at least one score weight must be > 0")


@dataclass(frozen=True)
class ScoredMatch:
    """One candidate's score and everything that explains it."""

    candidate_id: str
    name: str
    resume_path: str
    match_score: int  # 0-100
    components: dict[str, float]  # only the components that apply
    weights: dict[str, float]  # the renormalised weights used
    matched_skills: tuple[str, ...]  # JD must-have skills present, then nice-to-haves
    unmet: tuple[str, ...]
    matched_sections: tuple[str, ...]
    top_chunks: tuple[TopChunk, ...]
    check: RequirementCheck
    fused_score: float
    retrieval_rank: int  # position in the retrieval pool, 1-based
    reasoning: str = ""


def round_half_up(value: float) -> int:
    """``value`` rounded to the nearest integer, halves up (72.5 -> 73).

    Rounded to 9 decimals first, so float noise (0.725 * 100 = 72.49999999999999) doesn't
    turn a half into a round-down.
    """
    return int(Decimal(repr(round(value, 9))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def components(
    hit: CandidateHit, check: RequirementCheck, job: JobRequirements, calibration: Calibration
) -> dict[str, float]:
    """The components that apply to this JD, each 0-1."""
    values = {"semantic": calibration.normalise(hit.semantic.aggregate)}
    if check.results:
        values["must_have"] = sum(r.credit for r in check.results) / len(check.results)
    if job.nice_to_have:
        values["nice_to_have"] = len(check.nice_present) / len(job.nice_to_have)
    years = [r for r in check.results if r.has_years]
    if years:
        values["experience"] = sum(
            1.0 if not r.need_months else min(1.0, r.have_months / r.need_months) for r in years
        ) / len(years)
    return values


def renormalised(weights: ScoreWeights, used: list[str]) -> dict[str, float]:
    """The weights of ``used`` components, scaled to sum to 1."""
    raw = {name: getattr(weights, name) for name in used}
    total = sum(raw.values())
    if total == 0:
        raise ValueError(f"every applicable component has weight 0: {used}")
    return {name: value / total for name, value in raw.items()}


def matched_skills(check: RequirementCheck, job: JobRequirements) -> tuple[str, ...]:
    """Must-have skills the candidate has (JD order), then nice-to-haves present (JD order)."""
    have = set(check.skills)
    skills: list[str] = []
    for req in job.must_have:
        skills += [s for s in req.any_of if s in have and s not in skills]
    present = set(check.nice_present)
    skills += [s for s in job.nice_to_have_in_jd_order() if s in present and s not in skills]
    return tuple(skills)


def score_candidate(
    hit: CandidateHit,
    check: RequirementCheck,
    job: JobRequirements,
    calibration: Calibration,
    weights: ScoreWeights | None = None,
    retrieval_rank: int = 0,
) -> ScoredMatch:
    """Score one retrieved candidate against ``job``."""
    if hit.candidate_id != check.candidate_id:
        raise ValueError(f"hit is {hit.candidate_id}, check is {check.candidate_id}")
    values = components(hit, check, job, calibration)
    used = renormalised(weights or ScoreWeights(), [c for c in COMPONENTS if c in values])
    total = sum(used[name] * values[name] for name in used)
    return ScoredMatch(
        candidate_id=check.candidate_id,
        name=check.name,
        resume_path=check.resume_path,
        match_score=round_half_up(100 * total),
        components=values,
        weights=used,
        matched_skills=matched_skills(check, job),
        unmet=check.unmet,
        matched_sections=hit.matched_sections,
        top_chunks=hit.top_chunks,
        check=check,
        fused_score=hit.fused_score,
        retrieval_rank=retrieval_rank,
    )
