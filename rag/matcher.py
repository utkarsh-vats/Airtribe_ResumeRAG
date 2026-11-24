"""Match a JD against an indexed collection: filter -> retrieve -> score -> reason.

- **strict** (default): every candidate in the sidecar is checked against the must-haves
  (``rag.requirements_check``); retrieval then runs over the eligible candidates only
  (``retrieve(..., candidate_ids=eligible)``), so the brief's metadata filtering happens
  before ranking and no eligible candidate can be lost to the retrieval pool size. Filtered
  candidates are counted, with their unmet requirements.
- **soft**: retrieval runs over every candidate and all are scored; unmet requirements stay on
  each match and appear in its reasoning (``Misses: ...``).

A JD with no must-haves and no total-years minimum has nothing to filter on, so strict acts
like soft (with a note). The semantic list is widened to every chunk so every candidate
considered is ranked and scored. The top ``k`` by ``match_score`` are returned, ties broken
by retrieval rank and then candidate id.
"""

import time
from dataclasses import dataclass, field, replace
from datetime import date
from enum import StrEnum
from typing import Any

from rag.calibration import (
    DEFAULT_CEILING_PCT,
    DEFAULT_FLOOR_PCT,
    Calibration,
    calibration_settings,
    load_calibration,
)
from rag.embedders.base import Embedder
from rag.excerpts import select_excerpts
from rag.jd_parser import JobRequirements
from rag.metadata import CandidateMetadata
from rag.reasoning import reasoning
from rag.requirements_check import RequirementCheck, check_candidate
from rag.retrieval import RetrievalConfig, RetrievalResult, RetrievalTimings, retrieve
from rag.scoring import ScoredMatch, ScoreWeights, score_candidate

DEFAULT_K = 10


class FilterMode(StrEnum):
    """How must-haves are applied."""

    STRICT = "strict"
    SOFT = "soft"


@dataclass
class MatchTimings:
    """Seconds per stage; ``retrieval`` keeps its own sub-timings."""

    filter: float = 0.0
    retrieval: RetrievalTimings = field(default_factory=RetrievalTimings)
    scoring: float = 0.0

    @property
    def total(self) -> float:
        """All stages together."""
        return self.filter + self.retrieval.total + self.scoring


@dataclass(frozen=True)
class MatchResult:
    """The top ``k`` matches for one JD and how they were produced."""

    job: JobRequirements
    filter_mode: FilterMode
    matches: tuple[ScoredMatch, ...]
    candidates_total: int  # candidates in the collection
    eligible_count: int  # candidates meeting every must-have (in either mode)
    filtered_count: int  # removed by the must-have filter (strict only)
    filtered: dict[str, tuple[str, ...]]  # candidate id -> unmet requirements
    scored_count: int  # candidates retrieved and scored
    not_assessed: tuple[str, ...]  # requirement lines the parser couldn't turn into checks
    calibration: Calibration
    notes: tuple[str, ...]
    warnings: tuple[str, ...]
    timings: MatchTimings
    retrieval: RetrievalResult | None
    config: RetrievalConfig | None = None  # as requested (retrieval.config is as run)


def _has_hard_filters(job: JobRequirements) -> bool:
    return bool(job.must_have) or job.min_total_years is not None


def match(
    job: JobRequirements,
    store: Any,
    keyword_index: Any,
    embedder: Embedder,
    sidecar: Any,
    config: RetrievalConfig | None = None,
    filter_mode: FilterMode | str = FilterMode.STRICT,
    k: int = DEFAULT_K,
    *,
    as_of: date | None = None,
    weights: ScoreWeights | None = None,
    floor_pct: float = DEFAULT_FLOOR_PCT,
    ceiling_pct: float = DEFAULT_CEILING_PCT,
    calibration: Calibration | None = None,
) -> MatchResult:
    """The top ``k`` candidates for ``job``; see the module docstring."""
    filter_mode = FilterMode(filter_mode)
    if k < 1:
        raise ValueError("k must be >= 1")
    config = config or RetrievalConfig()
    timings = MatchTimings()
    notes: list[str] = []
    warnings: list[str] = []
    if as_of is not None and sidecar.as_of != as_of.isoformat():
        warnings.append(
            f"the index was built with as_of {sidecar.as_of}, not {as_of.isoformat()}; "
            "durations may be off (re-run index with --as-of)"
        )
    if calibration is None:  # the sidecar's, else a fallback (with a warning)
        calibration, calibration_warnings = load_calibration(sidecar, embedder)
        warnings += calibration_warnings
    if calibration.source == "collection":
        if calibration.retrieval and calibration.retrieval != calibration_settings(config):
            warnings.append(
                "the calibration was computed with other retrieval settings; re-run calibrate"
            )
        stored = (calibration.floor_pct, calibration.ceiling_pct)
        if stored != (float(floor_pct), float(ceiling_pct)):
            shown = "unrecorded" if None in stored else f"{stored[0]:g}/{stored[1]:g}"
            warnings.append(
                f"the calibration used percentiles {shown}, not {floor_pct:g}/{ceiling_pct:g}; "
                "re-run calibrate"
            )

    start = time.perf_counter()
    checks: dict[str, RequirementCheck] = {
        cid: check_candidate(CandidateMetadata.from_dict(record.metadata), job)
        for cid, record in sorted(sidecar.candidates.items())
    }
    strict = filter_mode is FilterMode.STRICT
    if strict and not _has_hard_filters(job):
        notes.append("the JD has no must-haves; strict filtering keeps every candidate")
        strict = False
    filtered = {cid: c.unmet for cid, c in checks.items() if strict and not c.eligible}
    pool_ids = [cid for cid in checks if cid not in filtered]
    timings.filter = time.perf_counter() - start

    retrieval = None
    scored: list[ScoredMatch] = []
    if pool_ids:
        retrieval = retrieve(
            job,
            store,
            keyword_index,
            embedder,
            # Every chunk in the semantic list, so every candidate considered is ranked there
            # and scored (228 chunks for the corpus; cheap).
            replace(
                config,
                pool_size=len(pool_ids),
                n_semantic=max(config.n_semantic, store.count()),
            ),
            candidate_ids=pool_ids,
        )
        timings.retrieval = retrieval.timings
        notes += retrieval.notes
        start = time.perf_counter()
        for rank, hit in enumerate(retrieval.hits, start=1):
            scored.append(
                score_candidate(hit, checks[hit.candidate_id], job, calibration, weights, rank)
            )
        scored.sort(key=lambda m: (-m.match_score, m.retrieval_rank, m.candidate_id))
        scored = [
            replace(
                m,
                reasoning=reasoning(m, job, show_misses=not strict),
                relevant_excerpts=tuple(select_excerpts(m, job)),
            )
            for m in scored[:k]
        ]
        timings.scoring = time.perf_counter() - start
    else:
        notes.append("no candidate meets every must-have")

    return MatchResult(
        job=job,
        filter_mode=filter_mode,
        matches=tuple(scored),
        candidates_total=len(checks),
        eligible_count=sum(1 for c in checks.values() if c.eligible),
        filtered_count=len(filtered),
        filtered=filtered,
        scored_count=len(retrieval.hits) if retrieval else 0,
        not_assessed=job.unparsed_requirements,
        calibration=calibration,
        notes=tuple(notes),
        warnings=tuple(warnings),
        timings=timings,
        retrieval=retrieval,
        config=config,
    )
