"""The JSON output of a match, in the brief's format.

``build_output(result, job, format)``:

- ``brief``: exactly the brief's structure::

      {"job_description": <the full JD text>,
       "top_matches": [{"candidate_name", "resume_path", "match_score", "matched_skills",
                        "relevant_excerpts", "reasoning"}, ...]}

  with the match keys in that order and nothing else.
- ``full`` (default): the same keys first, then per match ``candidate_id``,
  ``score_breakdown`` ({components, weights}) and, in soft mode, ``unmet``. Top level adds
  ``job`` (title, source, must-haves, nice-to-haves, total-years minimum, requirements not
  assessed), ``filter`` (mode and counts, filtered candidates with their unmet
  requirements), ``config`` (retrieval settings, collection, calibration used), ``notes``,
  ``warnings`` and ``timings_ms``.

Floats in the extras are rounded (4 decimals for scores, 1 for milliseconds) so the files
stay readable; ``match_score`` is already an integer.
"""

from enum import StrEnum
from typing import Any

from rag.jd_parser import JobRequirements
from rag.matcher import FilterMode, MatchResult
from rag.scoring import ScoredMatch

BRIEF_MATCH_KEYS = (
    "candidate_name",
    "resume_path",
    "match_score",
    "matched_skills",
    "relevant_excerpts",
    "reasoning",
)


class OutputFormat(StrEnum):
    """Which JSON shape to build."""

    FULL = "full"
    BRIEF = "brief"


def _brief_match(match: ScoredMatch) -> dict[str, Any]:
    return {
        "candidate_name": match.name,
        "resume_path": match.resume_path,
        "match_score": match.match_score,
        "matched_skills": list(match.matched_skills),
        "relevant_excerpts": list(match.relevant_excerpts),
        "reasoning": match.reasoning,
    }


def _rounded(values: dict[str, float], digits: int = 4) -> dict[str, float]:
    return {k: round(v, digits) for k, v in values.items()}


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def _full_match(match: ScoredMatch, soft: bool) -> dict[str, Any]:
    data = _brief_match(match) | {
        "candidate_id": match.candidate_id,
        "score_breakdown": {
            "components": _rounded(match.components),
            "weights": _rounded(match.weights),
        },
    }
    if soft:
        data["unmet"] = list(match.unmet)
    return data


def timings_ms(result: MatchResult) -> dict[str, Any]:
    """The match timings in milliseconds, retrieval with its stages."""
    retrieval = result.timings.retrieval
    stages = ("embed_query", "semantic_search", "keyword_search", "aggregate_fuse", "backfill")
    return {
        "filter": _ms(result.timings.filter),
        "retrieval": {stage: _ms(getattr(retrieval, stage)) for stage in stages}
        | {"total": _ms(retrieval.total)},
        "scoring": _ms(result.timings.scoring),
        "total": _ms(result.timings.total),
    }


def build_output(
    result: MatchResult,
    job: JobRequirements,
    format: OutputFormat | str = OutputFormat.FULL,
    *,
    source: str | None = None,
    collection: str | None = None,
) -> dict[str, Any]:
    """The output dict for one JD; ``source`` is where the JD came from (path or ``<text>``)."""
    format = OutputFormat(format)
    if format is OutputFormat.BRIEF:
        return {
            "job_description": job.full_text,
            "top_matches": [_brief_match(m) for m in result.matches],
        }
    soft = result.filter_mode is FilterMode.SOFT
    used = result.retrieval.config if result.retrieval else result.config
    retrieval_config = used.to_dict() if used else None
    return {
        "job_description": job.full_text,
        "top_matches": [_full_match(m, soft) for m in result.matches],
        "job": {
            "title": job.title,
            "source": source,
            "must_have": [
                {"any_of": list(r.any_of), "min_years": r.min_years} for r in job.must_have
            ],
            "nice_to_have": list(job.nice_to_have),
            "min_total_years": job.min_total_years,
            "not_assessed": list(result.not_assessed),
        },
        "filter": {
            "mode": result.filter_mode.value,
            "candidates_total": result.candidates_total,
            "eligible_count": result.eligible_count,
            "filtered_count": result.filtered_count,
            "scored_count": result.scored_count,
            "filtered": [
                {"candidate_id": cid, "unmet": list(unmet)}
                for cid, unmet in sorted(result.filtered.items())
            ],
        },
        "config": {
            "collection": collection,
            "retrieval": retrieval_config,
            "calibration": result.calibration.to_dict(),
        },
        "notes": list(result.notes),
        "warnings": list(result.warnings),
        "timings_ms": timings_ms(result),
    }
