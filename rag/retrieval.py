"""Hybrid retrieval: semantic and BM25 chunk lists -> per-candidate rankings -> RRF fusion.

Flow for one ``JobRequirements``:

1. **Semantic list**: embed the query source (``query_text`` or ``full_text``) with
   ``embed_query`` and take the ``n_semantic`` nearest chunks from the store. Chroma's cosine
   distance is ``1 - cos`` (checked against chromadb 1.5.9), so similarity is
   ``1 - distance``.
2. **Keyword list**: ``keyword_index.search_skills`` over the JD's critical skills (optionally
   plus its nice-to-haves); a JD with no critical skills uses its nice-to-haves, with a note.
   The index must use ``keyword_mode`` (default ``skills``: skill tokens only, so a word that
   merely looks like a skill, such as "Spark Energy Ltd", earns nothing).
3. **Per list**: each chunk score is multiplied by its section weight (a negative cosine
   counts as 0), chunks are grouped by candidate, and a candidate's list score is its best
   weighted chunk plus ``second_chunk_weight`` times its second best. Candidates are ranked by
   that score (zero scores are not ranked); ties go to the smaller candidate id.
4. **Fusion**: weighted Reciprocal Rank Fusion over the candidate rankings,
   ``fused(c) = sum_l w_l / (rrf_k + rank_l(c))``; a candidate absent from a list gets 0 from
   it. ``semantic`` and ``keyword`` modes use only their own list. The top ``pool_size``
   candidates form the pool.
5. **Backfill**: a pool candidate with no semantic score gets one from a store query filtered
   to that candidate; one with no keyword score gets one from a filtered BM25 search, or an
   explicit 0.0 when BM25 finds nothing. Backfilled scores never change the ranking; they
   give step 12 both signals for every pool candidate.

``candidate_ids`` restricts both lists and the backfills to those candidates; the matcher
uses it to retrieve over must-have-eligible candidates only. Final scoring is in
``rag.scoring``.
"""

import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from rag.embedders.base import Embedder
from rag.jd_parser import JobRequirements
from rag.keyword_index import TokenizerMode
from rag.sections import SectionKind

SEMANTIC = "semantic"
KEYWORD = "keyword"
LISTS = (SEMANTIC, KEYWORD)
BACKFILL_CHUNKS = 50  # more than any resume has; a filtered query returns what exists

DEFAULT_SECTION_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "experience": 1.0,
        "projects": 0.8,
        "skills": 0.7,
        "summary": 0.6,
        "certifications": 0.5,
        "other": 0.5,
        "education": 0.4,
        "coursework": 0.2,
        "hobbies": 0.1,
    }
)
DEFAULT_LIST_WEIGHTS: Mapping[str, float] = MappingProxyType({SEMANTIC: 1.0, KEYWORD: 1.0})


class RetrievalMode(StrEnum):
    """Which ranked lists are fused."""

    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


class QuerySource(StrEnum):
    """Which JD text is embedded."""

    QUERY_TEXT = "query_text"
    FULL_TEXT = "full_text"


class KeywordSkills(StrEnum):
    """Which JD skills form the BM25 query."""

    CRITICAL = "critical"
    CRITICAL_NICE = "critical+nice"


def _enum(kind: type[StrEnum], value: Any, name: str) -> Any:
    try:
        return kind(value)
    except ValueError:
        raise ValueError(
            f"{name} must be one of {[m.value for m in kind]}, not {value!r}"
        ) from None


def _weights(
    given: Mapping[str, float], defaults: Mapping[str, float], allowed: Iterable[str], name: str
):
    unknown = set(given) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {name} {sorted(unknown)}; allowed: {sorted(allowed)}")
    merged = dict(defaults) | dict(given)
    for key, value in merged.items():
        if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name}[{key!r}] must be a finite number >= 0, not {value!r}")
    return MappingProxyType(merged)


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval knobs. ``section_weights`` and ``list_weights`` are merged over the defaults."""

    mode: RetrievalMode | str = RetrievalMode.HYBRID
    query_source: QuerySource | str = QuerySource.QUERY_TEXT
    n_semantic: int = 100  # chunks
    n_keyword: int = 100  # chunks
    keyword_skills: KeywordSkills | str = KeywordSkills.CRITICAL
    keyword_mode: TokenizerMode | str = TokenizerMode.SKILLS  # of the KeywordIndex used
    section_weights: Mapping[str, float] = DEFAULT_SECTION_WEIGHTS
    second_chunk_weight: float = 0.3
    rrf_k: float = 60
    list_weights: Mapping[str, float] = DEFAULT_LIST_WEIGHTS
    pool_size: int = 20  # candidates

    def __post_init__(self):
        """Validate and normalise every field; raises ``ValueError``."""
        set_ = object.__setattr__
        set_(self, "mode", _enum(RetrievalMode, self.mode, "mode"))
        set_(self, "query_source", _enum(QuerySource, self.query_source, "query_source"))
        set_(self, "keyword_skills", _enum(KeywordSkills, self.keyword_skills, "keyword_skills"))
        set_(self, "keyword_mode", _enum(TokenizerMode, self.keyword_mode, "keyword_mode"))
        sections = [k.value for k in SectionKind]
        set_(
            self,
            "section_weights",
            _weights(self.section_weights, DEFAULT_SECTION_WEIGHTS, sections, "section_weights"),
        )
        set_(
            self,
            "list_weights",
            _weights(self.list_weights, DEFAULT_LIST_WEIGHTS, LISTS, "list_weights"),
        )
        for name in ("n_semantic", "n_keyword", "pool_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, not {value!r}")
        if not math.isfinite(self.rrf_k) or self.rrf_k <= 0:
            raise ValueError(f"rrf_k must be > 0, not {self.rrf_k!r}")
        scw = self.second_chunk_weight
        if not isinstance(scw, int | float) or not math.isfinite(scw) or scw < 0:
            raise ValueError(f"second_chunk_weight must be >= 0, not {scw!r}")
        if all(self.list_weights[name] == 0 for name in self.lists):
            raise ValueError(f"every list used by mode {self.mode.value!r} has weight 0")

    @property
    def lists(self) -> tuple[str, ...]:
        """The lists this mode fuses."""
        return {
            RetrievalMode.HYBRID: LISTS,
            RetrievalMode.SEMANTIC: (SEMANTIC,),
            RetrievalMode.KEYWORD: (KEYWORD,),
        }[self.mode]

    def section_weight(self, section: str) -> float:
        """Weight of a section; sections without one (``header``) use the ``other`` weight."""
        return self.section_weights.get(section, self.section_weights["other"])

    def to_dict(self) -> dict:
        """JSON form."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        return {
            k: v.value if isinstance(v, StrEnum) else dict(v) if isinstance(v, Mapping) else v
            for k, v in data.items()
        }


# --- Building blocks ----------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkInfo:
    """What retrieval knows about a chunk."""

    chunk_id: str
    candidate_id: str
    section: str
    role_index: int | None
    text: str
    body_start: int = 0  # lines of ``text`` before the body (role-header lines)


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk's raw score in one list and its section-weighted score."""

    chunk: ChunkInfo
    raw: float
    weighted: float


@dataclass(frozen=True)
class CandidateListScore:
    """One candidate in one list: its chunks (best first) and aggregate score."""

    candidate_id: str
    score: float
    chunks: tuple[ScoredChunk, ...]

    @property
    def best_raw(self) -> float:
        """The highest raw chunk score."""
        return max(c.raw for c in self.chunks)

    @property
    def best_weighted(self) -> float:
        """The highest weighted chunk score."""
        return self.chunks[0].weighted

    def contributing(self, second_chunk_weight: float) -> tuple[ScoredChunk, ...]:
        """The chunks that make up ``score``: the best, and the second when it counts."""
        n = 2 if second_chunk_weight > 0 else 1
        return tuple(c for c in self.chunks[:n] if c.weighted > 0)


def weight_chunks(
    hits: Iterable[tuple[ChunkInfo, float]], config: RetrievalConfig
) -> list[ScoredChunk]:
    """Section-weighted scores; a negative raw score (cosine) counts as 0."""
    return [
        ScoredChunk(info, raw, max(raw, 0.0) * config.section_weight(info.section))
        for info, raw in hits
    ]


def aggregate(
    chunks: Iterable[ScoredChunk], second_chunk_weight: float
) -> dict[str, CandidateListScore]:
    """Per candidate: best weighted chunk + ``second_chunk_weight`` x second best."""
    grouped: dict[str, list[ScoredChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.chunk.candidate_id].append(chunk)
    out = {}
    for candidate_id, items in grouped.items():
        items.sort(key=lambda c: (-c.weighted, c.chunk.chunk_id))
        score = items[0].weighted + (
            second_chunk_weight * items[1].weighted if len(items) > 1 else 0
        )
        out[candidate_id] = CandidateListScore(candidate_id, score, tuple(items))
    return out


def rank_candidates(scores: Mapping[str, CandidateListScore]) -> list[str]:
    """Candidate ids by score (best first, ties by id); zero scores are left out."""
    ranked = [s for s in scores.values() if s.score > 0]
    return [s.candidate_id for s in sorted(ranked, key=lambda s: (-s.score, s.candidate_id))]


def rrf_fuse(
    rankings: Mapping[str, Sequence[str]], weights: Mapping[str, float], k: float
) -> dict[str, float]:
    """Weighted RRF: ``sum_l w_l / (k + rank)`` with 1-based ranks; absent means 0."""
    fused: dict[str, float] = defaultdict(float)
    for name, ranking in rankings.items():
        for rank, candidate_id in enumerate(ranking, start=1):
            fused[candidate_id] += weights[name] / (k + rank)
    return dict(fused)


def order_by_score(scores: Mapping[str, float]) -> list[str]:
    """Ids by score, best first, ties by id."""
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


# --- Results ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ListScore:
    """A pool candidate's standing in one list."""

    raw: float  # best raw chunk score (cosine similarity, or BM25)
    weighted: float  # best section-weighted chunk score
    aggregate: float  # best weighted + second_chunk_weight x second best
    rank: int | None  # 1-based rank in the list's candidate ranking; None if not ranked there
    backfilled: bool = False  # filled in after fusion; did not affect the ranking


@dataclass(frozen=True)
class TopChunk:
    """One of a candidate's best chunks, with what each list scored it."""

    chunk_id: str
    section: str
    role_index: int | None
    found_by: tuple[str, ...]
    semantic_raw: float | None
    semantic_weighted: float | None
    keyword_raw: float | None
    keyword_weighted: float | None
    text: str
    body_start: int = 0  # lines of ``text`` before the body (role-header lines)


@dataclass(frozen=True)
class CandidateHit:
    """One candidate in the retrieval pool."""

    candidate_id: str
    fused_score: float
    semantic: ListScore
    keyword: ListScore
    top_chunks: tuple[TopChunk, ...]
    matched_sections: tuple[str, ...]


@dataclass
class RetrievalTimings:
    """Seconds per stage."""

    embed_query: float = 0.0
    semantic_search: float = 0.0
    keyword_search: float = 0.0
    aggregate_fuse: float = 0.0
    backfill: float = 0.0

    @property
    def total(self) -> float:
        """All stages together."""
        return (
            self.embed_query
            + self.semantic_search
            + self.keyword_search
            + self.aggregate_fuse
            + self.backfill
        )


@dataclass(frozen=True)
class RetrievalResult:
    """The pool for one JD, best first, and how it was produced."""

    hits: tuple[CandidateHit, ...]
    config: RetrievalConfig
    query_text: str
    keyword_skills: tuple[str, ...]
    notes: tuple[str, ...]
    timings: RetrievalTimings
    ranked: Mapping[str, int] = field(default_factory=dict)  # candidates ranked per list

    def to_dict(self) -> dict:
        """JSON form."""
        return {
            "config": self.config.to_dict(),
            "query_text": self.query_text,
            "keyword_skills": list(self.keyword_skills),
            "notes": list(self.notes),
            "ranked": dict(self.ranked),
            "timings": asdict(self.timings) | {"total": self.timings.total},
            "hits": [asdict(h) for h in self.hits],
        }


# --- Retrieval ----------------------------------------------------------------------------


def _semantic_hits(store: Any, vector: Sequence[float], n: int, where: dict | None = None):
    found = store.query(vector, n, where=where)
    hits = []
    for chunk_id, text, meta, distance in zip(
        found["ids"][0],
        found["documents"][0],
        found["metadatas"][0],
        found["distances"][0],
        strict=True,
    ):
        info = ChunkInfo(
            chunk_id,
            meta["candidate_id"],
            meta["section"],
            meta.get("role_index"),
            text or "",
            meta.get("body_start", 0),
        )
        hits.append((info, 1.0 - float(distance)))
    return hits


def _keyword_hits(keyword_index: Any, skills: Sequence[str], n: int, where: dict | None = None):
    hits = []
    for chunk_id, score in keyword_index.search_skills(skills, n, where=where):
        doc = keyword_index.doc(chunk_id)
        hits.append(
            (
                ChunkInfo(
                    chunk_id,
                    doc.candidate_id,
                    doc.section,
                    doc.role_index,
                    doc.text,
                    doc.body_start,
                ),
                score,
            )
        )
    return hits


def keyword_query_skills(
    job: JobRequirements, config: RetrievalConfig
) -> tuple[list[str], list[str]]:
    """The skills for the BM25 query, and any notes about how they were chosen."""
    notes = []
    skills = list(job.critical_skills)
    if not skills:
        skills = list(job.nice_to_have)
        notes.append(
            "no critical skills (no must-haves); the keyword query uses the nice-to-have skills"
        )
    elif config.keyword_skills is KeywordSkills.CRITICAL_NICE:
        skills += [s for s in job.nice_to_have if s not in skills]
    if not skills:
        notes.append("no skills for the keyword query; the keyword list is empty")
    return skills, notes


class _Timer:
    def __init__(
        self,
        timings: RetrievalTimings,
        stage: str,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.timings, self.stage, self.clock = timings, stage, clock

    def __enter__(self):
        self.start = self.clock()

    def __exit__(self, *exc):
        elapsed = self.clock() - self.start
        setattr(self.timings, self.stage, getattr(self.timings, self.stage) + elapsed)


def retrieve(
    job: JobRequirements,
    store: Any,
    keyword_index: Any,
    embedder: Embedder,
    config: RetrievalConfig | None = None,
    candidate_ids: Iterable[str] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RetrievalResult:
    """The top ``pool_size`` candidates for ``job``, with both lists' scores for each.

    ``candidate_ids``, when given, restricts everything to those candidates; an empty set
    gives an empty result with a note.
    """
    config = config or RetrievalConfig()
    store_embedder = getattr(store, "embedder_id", None)
    if store_embedder is not None and store_embedder != embedder.embedder_id:
        raise ValueError(f"store holds {store_embedder} vectors, not {embedder.embedder_id}")
    index_mode = getattr(keyword_index, "mode", None)
    if index_mode is not None and index_mode != config.keyword_mode:
        raise ValueError(
            f"keyword index uses {index_mode.value!r} tokens; the config asks for "
            f"{config.keyword_mode.value!r} (build it with KeywordIndex.from_store(store, "
            "mode=config.keyword_mode))"
        )
    timings = RetrievalTimings()
    notes: list[str] = []
    query = job.query_text if config.query_source is QuerySource.QUERY_TEXT else job.full_text
    skills, skill_notes = keyword_query_skills(job, config)
    notes += skill_notes
    semantic_where = keyword_where = None
    if candidate_ids is not None:
        ids = sorted(set(candidate_ids))
        if not ids:
            notes.append("no candidates to retrieve from")
            return RetrievalResult((), config, query, tuple(skills), tuple(notes), timings)
        semantic_where = {"candidate_id": {"$in": ids}}
        keyword_where = {"candidate_ids": ids}

    with _Timer(timings, "embed_query", clock):  # also needed in keyword mode, for the backfill
        vector = embedder.embed_query(query)

    lists: dict[str, dict[str, CandidateListScore]] = {SEMANTIC: {}, KEYWORD: {}}
    raw_hits: dict[str, list[tuple[ChunkInfo, float]]] = {SEMANTIC: [], KEYWORD: []}
    if SEMANTIC in config.lists:
        with _Timer(timings, "semantic_search", clock):
            raw_hits[SEMANTIC] = _semantic_hits(
                store, vector, max(1, min(config.n_semantic, store.count())), semantic_where
            )
    if KEYWORD in config.lists and skills:
        with _Timer(timings, "keyword_search", clock):
            raw_hits[KEYWORD] = _keyword_hits(
                keyword_index, skills, config.n_keyword, keyword_where
            )

    with _Timer(timings, "aggregate_fuse", clock):
        for name in config.lists:
            lists[name] = aggregate(
                weight_chunks(raw_hits[name], config), config.second_chunk_weight
            )
        rankings = {name: rank_candidates(lists[name]) for name in config.lists}
        ranks = {
            name: {cid: i for i, cid in enumerate(rankings.get(name, []), start=1)}
            for name in LISTS
        }
        fused = rrf_fuse(rankings, config.list_weights, config.rrf_k)
        pool = order_by_score(fused)[: config.pool_size]

    backfilled: dict[str, set[str]] = {SEMANTIC: set(), KEYWORD: set()}
    with _Timer(timings, "backfill", clock):
        for candidate_id in pool:
            if candidate_id not in lists[SEMANTIC]:
                hits = _semantic_hits(
                    store, vector, BACKFILL_CHUNKS, where={"candidate_id": candidate_id}
                )
                found = aggregate(weight_chunks(hits, config), config.second_chunk_weight)
                lists[SEMANTIC].update(found)
                backfilled[SEMANTIC].add(candidate_id)
            if candidate_id not in lists[KEYWORD]:
                hits = (
                    _keyword_hits(
                        keyword_index, skills, config.n_keyword, {"candidate_ids": [candidate_id]}
                    )
                    if skills
                    else []
                )
                found = aggregate(weight_chunks(hits, config), config.second_chunk_weight)
                lists[KEYWORD].update(found)
                backfilled[KEYWORD].add(candidate_id)
                if found and KEYWORD in config.lists:
                    notes.append(
                        f"{candidate_id}: keyword score found by backfill (beyond n_keyword)"
                    )

    hits = tuple(_hit(cid, fused[cid], lists, ranks, backfilled, config) for cid in pool)
    return RetrievalResult(
        hits=hits,
        config=config,
        query_text=query,
        keyword_skills=tuple(skills),
        notes=tuple(notes),
        timings=timings,
        ranked={name: len(rankings.get(name, [])) for name in LISTS},
    )


def _list_score(scores: CandidateListScore | None, rank: int | None, backfilled: bool) -> ListScore:
    if scores is None:  # nothing found, even by the backfill
        return ListScore(0.0, 0.0, 0.0, None, backfilled)
    return ListScore(scores.best_raw, scores.best_weighted, scores.score, rank, backfilled)


def _hit(
    candidate_id: str,
    fused_score: float,
    lists: Mapping[str, Mapping[str, CandidateListScore]],
    ranks: Mapping[str, Mapping[str, int]],
    backfilled: Mapping[str, set[str]],
    config: RetrievalConfig,
) -> CandidateHit:
    per_list = {name: lists[name].get(candidate_id) for name in LISTS}
    # Chunks that make up each list's score, merged by id; ordered by a chunk-level RRF over
    # their position within the candidate in each list.
    merged: dict[str, dict[str, Any]] = {}
    for name in LISTS:
        scores = per_list[name]
        if scores is None:
            continue
        for position, chunk in enumerate(scores.contributing(config.second_chunk_weight), 1):
            entry = merged.setdefault(chunk.chunk.chunk_id, {"info": chunk.chunk, "rrf": 0.0})
            entry[name] = chunk
            entry["rrf"] += config.list_weights[name] / (config.rrf_k + position)
    ordered = sorted(merged.values(), key=lambda e: (-e["rrf"], e["info"].chunk_id))
    top = tuple(
        TopChunk(
            chunk_id=e["info"].chunk_id,
            section=e["info"].section,
            role_index=e["info"].role_index,
            found_by=tuple(name for name in LISTS if name in e),
            semantic_raw=e[SEMANTIC].raw if SEMANTIC in e else None,
            semantic_weighted=e[SEMANTIC].weighted if SEMANTIC in e else None,
            keyword_raw=e[KEYWORD].raw if KEYWORD in e else None,
            keyword_weighted=e[KEYWORD].weighted if KEYWORD in e else None,
            text=e["info"].text,
            body_start=e["info"].body_start,
        )
        for e in ordered[:3]
    )
    sections = sorted({e["info"].section for e in merged.values()})
    return CandidateHit(
        candidate_id=candidate_id,
        fused_score=fused_score,
        semantic=_list_score(
            per_list[SEMANTIC],
            ranks[SEMANTIC].get(candidate_id),
            candidate_id in backfilled[SEMANTIC],
        ),
        keyword=_list_score(
            per_list[KEYWORD],
            ranks[KEYWORD].get(candidate_id),
            candidate_id in backfilled[KEYWORD],
        ),
        top_chunks=top,
        matched_sections=tuple(sections),
    )
