"""Semantic calibration: the range that maps a candidate's semantic aggregate onto 0-1.

Cosine similarities from one model sit in a narrow, model-specific band (bge-small rarely
goes below 0.5 for any resume against any JD), so a raw similarity says little on its own.
Calibration fixes a **floor** and a **ceiling** per collection, and scoring uses
``clamp((aggregate - floor) / (ceiling - floor))``:

- for every JD in a set and every candidate in the collection, run retrieval in semantic mode
  over all candidates and take the candidate-level semantic aggregate (best weighted chunk +
  ``second_chunk_weight`` x second best);
- floor = the ``floor_pct`` percentile of those values (default 50, the median), ceiling =
  the ``ceiling_pct`` percentile (default 99), with linear interpolation.

The median floor means a candidate no closer than the typical (mostly unrelated) pair scores
0 on this component. The ceiling sits near the top of all pairs, so the best matches still
differ from each other: at the 95th percentile most strong pairs saturated at 1.0.
**No labels are read**: only JDs and the indexed resumes. It does use the same corpus the
evaluation runs on, so it is fitted to this dataset (see DESIGN.md).

The result is stored in the collection's sidecar under ``calibration``. A collection without
one uses a per-model fallback from ``MODEL_CONFIGS``, or a generic one, with a warning.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rag.embedders.base import Embedder
from rag.jd_parser import JobRequirements
from rag.retrieval import RetrievalConfig, RetrievalMode, retrieve

DEFAULT_FLOOR_PCT = 50.0
DEFAULT_CEILING_PCT = 99.0
GENERIC_FALLBACK = (0.0, 1.0)


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100) with linear interpolation between closest ranks."""
    if not values:
        raise ValueError("percentile of no values")
    if not 0 <= q <= 100:
        raise ValueError("q must be within 0-100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True)
class Calibration:
    """Floor and ceiling of the semantic aggregate, and where they came from."""

    floor: float
    ceiling: float
    source: str  # "collection" (computed), "model" (MODEL_CONFIGS) or "generic"
    embedder_id: str = ""
    jd_files: tuple[str, ...] = ()
    pairs: int = 0
    computed_at: str | None = None
    retrieval: dict = field(default_factory=dict)  # the settings the aggregates depend on
    floor_pct: float | None = None  # percentiles used; None for fallbacks and old sidecars
    ceiling_pct: float | None = None

    def __post_init__(self):
        """Reject a range that can't be used."""
        if not (math.isfinite(self.floor) and math.isfinite(self.ceiling)):
            raise ValueError("calibration floor and ceiling must be finite")

    def normalise(self, value: float) -> float:
        """``clamp((value - floor) / (ceiling - floor))`` to 0-1."""
        span = self.ceiling - self.floor
        if span <= 0:
            return 1.0 if value >= self.ceiling else 0.0
        return min(1.0, max(0.0, (value - self.floor) / span))

    def to_dict(self) -> dict:
        """JSON form (stored in the sidecar)."""
        return {
            "floor": self.floor,
            "ceiling": self.ceiling,
            "source": self.source,
            "embedder_id": self.embedder_id,
            "jd_files": list(self.jd_files),
            "pairs": self.pairs,
            "computed_at": self.computed_at,
            "retrieval": dict(self.retrieval),
            "floor_pct": self.floor_pct,
            "ceiling_pct": self.ceiling_pct,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Calibration:
        """Inverse of ``to_dict``."""
        return cls(
            floor=float(data["floor"]),
            ceiling=float(data["ceiling"]),
            source=data.get("source", "collection"),
            embedder_id=data.get("embedder_id", ""),
            jd_files=tuple(data.get("jd_files", ())),
            pairs=int(data.get("pairs", 0)),
            computed_at=data.get("computed_at"),
            retrieval=dict(data.get("retrieval", {})),
            floor_pct=data.get("floor_pct"),
            ceiling_pct=data.get("ceiling_pct"),
        )


def calibration_settings(config: RetrievalConfig) -> dict:
    """The retrieval settings a semantic aggregate depends on."""
    return {
        "query_source": config.query_source.value,
        "second_chunk_weight": config.second_chunk_weight,
        "section_weights": dict(config.section_weights),
    }


def check_percentiles(floor_pct: float, ceiling_pct: float) -> None:
    """Require ``0 <= floor_pct < ceiling_pct <= 100``."""
    if not (0 <= floor_pct < ceiling_pct <= 100):
        raise ValueError(
            f"need 0 <= floor_pct < ceiling_pct <= 100, got {floor_pct} and {ceiling_pct}"
        )


def semantic_aggregates(
    jobs: Mapping[str, JobRequirements],
    store: Any,
    keyword_index: Any,
    embedder: Embedder,
    config: RetrievalConfig | None = None,
) -> dict[tuple[str, str], float]:
    """``{(jd key, candidate id): semantic aggregate}`` for every pair, label-free."""
    if not jobs:
        raise ValueError("calibration needs at least one JD")
    base = config or RetrievalConfig()
    candidates = store.list_candidates()
    if not candidates:
        raise ValueError("the collection has no candidates")
    semantic = RetrievalConfig(
        **{f: getattr(base, f) for f in ("query_source", "section_weights", "second_chunk_weight")},
        keyword_mode=base.keyword_mode,
        mode=RetrievalMode.SEMANTIC,
        n_semantic=max(1, store.count()),  # every chunk, so every candidate is ranked
        pool_size=len(candidates),
    )
    values: dict[tuple[str, str], float] = {}
    for key, job in jobs.items():
        result = retrieve(job, store, keyword_index, embedder, semantic)
        for hit in result.hits:
            values[(key, hit.candidate_id)] = hit.semantic.aggregate
    return values


def compute_calibration(
    jobs: Mapping[str, JobRequirements],
    store: Any,
    keyword_index: Any,
    embedder: Embedder,
    config: RetrievalConfig | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    floor_pct: float = DEFAULT_FLOOR_PCT,
    ceiling_pct: float = DEFAULT_CEILING_PCT,
) -> Calibration:
    """Calibrate over every (JD, candidate) pair; ``jobs`` maps a JD's file to its parse."""
    check_percentiles(floor_pct, ceiling_pct)
    base = config or RetrievalConfig()
    values = list(semantic_aggregates(jobs, store, keyword_index, embedder, base).values())
    return calibration_from_values(
        values,
        floor_pct,
        ceiling_pct,
        embedder_id=embedder.embedder_id,
        jd_files=tuple(sorted(jobs)),
        computed_at=now().isoformat(timespec="seconds"),
        retrieval=calibration_settings(base),
    )


def calibration_from_values(
    values: Sequence[float],
    floor_pct: float = DEFAULT_FLOOR_PCT,
    ceiling_pct: float = DEFAULT_CEILING_PCT,
    source: str = "collection",
    **details: Any,
) -> Calibration:
    """A calibration from already computed semantic aggregates."""
    check_percentiles(floor_pct, ceiling_pct)
    return Calibration(
        floor=percentile(values, floor_pct),
        ceiling=percentile(values, ceiling_pct),
        source=source,
        pairs=len(values),
        floor_pct=float(floor_pct),
        ceiling_pct=float(ceiling_pct),
        **details,
    )


def fallback_calibration(embedder: Embedder) -> tuple[Calibration, str]:
    """A per-model fallback (``MODEL_CONFIGS``) or the generic one, and the warning to show."""
    model_config = getattr(embedder, "config", None)
    floor = getattr(model_config, "semantic_floor", None)
    ceiling = getattr(model_config, "semantic_ceiling", None)
    if floor is not None and ceiling is not None:
        calibration = Calibration(floor, ceiling, "model", embedder.embedder_id)
        where = f"the {embedder.model} fallback"
    else:
        calibration = Calibration(*GENERIC_FALLBACK, "generic", embedder.embedder_id)
        where = "the generic fallback"
    warning = (
        f"the collection has no semantic calibration; using {where} "
        f"(floor {calibration.floor:.3f}, ceiling {calibration.ceiling:.3f}). "
        "Run: python resume_rag.py calibrate"
    )
    return calibration, warning


def load_calibration(sidecar: Any, embedder: Embedder) -> tuple[Calibration, list[str]]:
    """The sidecar's calibration, else a fallback; plus any warnings."""
    data = getattr(sidecar, "calibration", None)
    if data:
        return Calibration.from_dict(data), []
    calibration, warning = fallback_calibration(embedder)
    return calibration, [warning]


def save_calibration(store: Any, calibration: Calibration) -> None:
    """Store ``calibration`` in the collection's sidecar (which must exist)."""
    sidecar = store.load_sidecar()
    if sidecar is None:
        raise ValueError(f"no sidecar for {store.name}; index the collection first")
    sidecar.calibration = calibration.to_dict()
    store.save_sidecar(sidecar)
