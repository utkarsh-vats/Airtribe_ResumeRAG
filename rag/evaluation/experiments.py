"""Experiment runners for the notebook: each returns tidy rows (``Rows``, a list of dicts).

The notebook only calls these (and ``rag.evaluation.display``); it holds no logic of its own.
Every runner is deterministic for a given collection and configuration, and reads labels only
through ``rag.evaluation.ground_truth``. The core pipeline stays label-free.

- ``compare_collections``: ranking metrics per collection (chunker x embedder).
- ``recall_curve``: retrieval-order recall@k per mode, k = 1..15.
- ``compare_retrieval_configs``: ranking metrics per ``RetrievalConfig`` variant.
- ``compare_scoring``: scored-order metrics and soft top 10 per ``ScoreWeights`` variant.
- ``calibration_variants``: floor/ceiling percentile pairs, in memory only.
- ``trap_cases``: known hard cases (look-alikes, aliases, coursework, ...) per variant.
- ``chunk_stats``: words per chunk by section.
"""

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rag.calibration import (
    Calibration,
    calibration_from_values,
    load_calibration,
    semantic_aggregates,
)
from rag.embedders.base import Embedder
from rag.evaluation.display import text_table, to_frame
from rag.evaluation.evaluate import (
    MODES,
    ORDERS,
    evaluate_extraction,
    evaluate_ranking,
    load_jobs,
)
from rag.evaluation.ground_truth import (
    GROUND_TRUTH_DIR,
    Manifest,
    Qrels,
    load_manifest,
    load_qrels,
    load_resume_renders,
)
from rag.evaluation.metrics import LENIENT, STRICT, macro_average, recall_at_k
from rag.jd_parser import JobRequirements, parse_job_description
from rag.keyword_index import KeywordIndex, TokenizerMode
from rag.matcher import FilterMode, MatchResult, match
from rag.metadata import CandidateMetadata
from rag.retrieval import RetrievalConfig, RetrievalMode
from rag.scoring import ScoreWeights
from rag.skill_matcher import find_mentions
from rag.store import Sidecar, VectorStore, collection_name, read_sidecar
from rag.timeline import months_to_years

JD_DIR = GROUND_TRUTH_DIR.parent / "job_descriptions"
THRESHOLDS = {"strict": STRICT, "lenient": LENIENT}


class Rows(list):
    """A list of row dicts with notebook conveniences."""

    def to_dataframe(self, columns: Sequence[str] | None = None) -> Any:
        """A pandas DataFrame (pandas imported lazily)."""
        return to_frame(self, columns)

    def text(self, columns: Sequence[str] | None = None) -> str:
        """A plain-text table."""
        return text_table(self, columns)


@dataclass
class ExperimentCollection:
    """An indexed collection plus the embedder that queries it."""

    label: str
    store: VectorStore
    embedder: Embedder
    _indexes: dict[str, KeywordIndex] = field(default_factory=dict, repr=False)
    _sidecar: Sidecar | None = field(default=None, repr=False)

    @property
    def sidecar(self) -> Sidecar:
        """The collection's sidecar (loaded once)."""
        if self._sidecar is None:
            self._sidecar = self.store.load_sidecar()
            if self._sidecar is None:
                raise ValueError(f"{self.store.name} has no sidecar; index it first")
        return self._sidecar

    @property
    def calibration(self) -> Calibration:
        """The stored calibration, else the fallback."""
        return load_calibration(self.sidecar, self.embedder)[0]

    def keyword_index(self, mode: Any) -> KeywordIndex:
        """A keyword index in ``mode`` (built once per mode)."""
        key = str(getattr(mode, "value", mode))
        if key not in self._indexes:
            self._indexes[key] = KeywordIndex.from_store(self.store, mode=key)
        return self._indexes[key]


def open_collection(
    label: str, chunker_id: str, embedder: Embedder, chroma_dir: str | Path = ".chroma"
) -> ExperimentCollection:
    """An existing collection under ``chroma_dir``."""
    store = VectorStore(chroma_dir, chunker_id, embedder.embedder_id, embedder.dim)
    collection = ExperimentCollection(label, store, embedder)
    collection.sidecar  # noqa: B018 - fail early if it was never indexed
    return collection


def _inputs(
    jd_paths: Iterable[str | Path] | None, qrels: Qrels | None
) -> tuple[dict[str, JobRequirements], Qrels]:
    jobs = load_jobs(sorted(JD_DIR.glob("*.txt")) if jd_paths is None else jd_paths)
    return jobs, qrels or load_qrels()


def _soft(
    collection: ExperimentCollection,
    job: JobRequirements,
    config: RetrievalConfig,
    weights: ScoreWeights | None = None,
    calibration: Calibration | None = None,
    filter_mode: FilterMode = FilterMode.SOFT,
) -> MatchResult:
    return match(
        job,
        collection.store,
        collection.keyword_index(config.keyword_mode),
        collection.embedder,
        collection.sidecar,
        config,
        filter_mode,
        k=len(collection.sidecar.candidates),
        weights=weights,
        calibration=calibration or collection.calibration,
    )


def _macro_rows(evaluation: Any, orders: Sequence[str], **labels: Any) -> list[dict]:
    return [
        labels | {"mode": mode, "order": order} | metrics
        for (mode, order), metrics in evaluation.macro.items()
        if order in orders
    ]


# --- Runners ------------------------------------------------------------------------------


def compare_collections(
    collections: Sequence[ExperimentCollection],
    modes: Sequence[str] = MODES,
    orders: Sequence[str] = ORDERS,
    config: RetrievalConfig | None = None,
    jd_paths: Iterable[str | Path] | None = None,
    qrels: Qrels | None = None,
) -> Rows:
    """Macro ranking metrics per collection, mode and order."""
    config = config or RetrievalConfig()
    jobs, qrels = _inputs(jd_paths, qrels)
    rows = Rows()
    for c in collections:
        evaluation = evaluate_ranking(
            jobs,
            qrels,
            c.store,
            c.keyword_index(config.keyword_mode),
            c.embedder,
            config,
            calibration=c.calibration,
            modes=modes,
        )
        rows += _macro_rows(evaluation, orders, collection=c.label)
    return rows


def recall_curve(
    collection: ExperimentCollection,
    modes: Sequence[str] = MODES,
    ks: Iterable[int] = range(1, 16),
    threshold: str = "strict",
    config: RetrievalConfig | None = None,
    jd_paths: Iterable[str | Path] | None = None,
    qrels: Qrels | None = None,
) -> Rows:
    """Retrieval-order recall@k per mode, macro over JDs."""
    if threshold not in THRESHOLDS:
        raise ValueError(f"threshold must be one of {sorted(THRESHOLDS)}")
    config = config or RetrievalConfig()
    jobs, qrels = _inputs(jd_paths, qrels)
    ks = list(ks)
    rows = Rows()
    for mode in modes:
        rankings = {}
        for jd_id, job in jobs.items():
            result = _soft(collection, job, replace(config, mode=mode))
            rankings[jd_id] = (
                [h.candidate_id for h in result.retrieval.hits] if result.retrieval else []
            )
        for k in ks:
            recall = macro_average(
                recall_at_k(rankings[jd], qrels.jds[jd].grades, k, THRESHOLDS[threshold])
                for jd in jobs
            )
            rows.append(
                {
                    "collection": collection.label,
                    "mode": mode,
                    "threshold": threshold,
                    "k": k,
                    "recall": recall,
                }
            )
    return rows


def compare_retrieval_configs(
    collection: ExperimentCollection,
    variants: Mapping[str, RetrievalConfig],
    modes: Sequence[str] = MODES,
    orders: Sequence[str] = ORDERS,
    jd_paths: Iterable[str | Path] | None = None,
    qrels: Qrels | None = None,
    per_jd: bool = False,
) -> Rows:
    """Ranking metrics per retrieval-config variant (each variant's mode is overridden).

    Macro averages by default; ``per_jd=True`` returns one row per JD instead.
    """
    jobs, qrels = _inputs(jd_paths, qrels)
    rows = Rows()
    for name, config in variants.items():
        evaluation = evaluate_ranking(
            jobs,
            qrels,
            collection.store,
            collection.keyword_index(config.keyword_mode),
            collection.embedder,
            config,
            calibration=collection.calibration,
            modes=modes,
        )
        if per_jd:
            rows += [
                {"collection": collection.label, "variant": name} | row
                for row in evaluation.rows
                if row["order"] in orders
            ]
        else:
            rows += _macro_rows(evaluation, orders, collection=collection.label, variant=name)
    return rows


@dataclass(frozen=True)
class ScoringComparison:
    """Scored-order metrics and each variant's soft top 10 per JD."""

    metrics: Rows
    top10: Rows


def compare_scoring(
    collection: ExperimentCollection,
    variants: Mapping[str, ScoreWeights],
    mode: str = "hybrid",
    config: RetrievalConfig | None = None,
    jd_paths: Iterable[str | Path] | None = None,
    qrels: Qrels | None = None,
) -> ScoringComparison:
    """Scored-order ranking metrics per weights variant, plus the soft top 10s with labels."""
    config = replace(config or RetrievalConfig(), mode=mode)
    jobs, qrels = _inputs(jd_paths, qrels)
    metrics, top10 = Rows(), Rows()
    for name, weights in variants.items():
        evaluation = evaluate_ranking(
            jobs,
            qrels,
            collection.store,
            collection.keyword_index(config.keyword_mode),
            collection.embedder,
            config,
            scoring_weights=weights,
            calibration=collection.calibration,
            modes=(mode,),
        )
        metrics += _macro_rows(evaluation, ("scored",), collection=collection.label, variant=name)
        for jd_id, job in jobs.items():
            result = _soft(collection, job, config, weights)
            for rank, m in enumerate(result.matches[:10], start=1):
                top10.append(
                    {
                        "variant": name,
                        "jd_id": jd_id,
                        "rank": rank,
                        "candidate_id": m.candidate_id,
                        "label": qrels.jds[jd_id].labels.get(m.candidate_id),
                        "match_score": m.match_score,
                    }
                )
    return ScoringComparison(metrics, top10)


def calibration_variants(
    collection: ExperimentCollection,
    percentile_pairs: Sequence[tuple[float, float]] = ((50, 95), (50, 99), (50, 100)),
    config: RetrievalConfig | None = None,
    jd_paths: Iterable[str | Path] | None = None,
    qrels: Qrels | None = None,
) -> Rows:
    """Floor/ceiling per percentile pair, semantic saturation, and strict score spread.

    Nothing is written: calibrations are built in memory and passed to ``match``.
    """
    config = config or RetrievalConfig()
    jobs, qrels = _inputs(jd_paths, qrels)
    values = semantic_aggregates(
        jobs,
        collection.store,
        collection.keyword_index(config.keyword_mode),
        collection.embedder,
        config,
    )
    rows = Rows()
    for floor_pct, ceiling_pct in percentile_pairs:
        cal = calibration_from_values(list(values.values()), floor_pct, ceiling_pct, "experiment")
        scores: list[int] = []
        eligible = saturated = 0
        for jd_id, job in jobs.items():
            result = _soft(collection, job, config, calibration=cal, filter_mode=FilterMode.STRICT)
            for m in result.matches:
                scores.append(m.match_score)
                if qrels.jds[jd_id].eligible.get(m.candidate_id):
                    eligible += 1
                    saturated += m.components["semantic"] >= 1.0
        rows.append(
            {
                "collection": collection.label,
                "floor_pct": float(floor_pct),
                "ceiling_pct": float(ceiling_pct),
                "floor": cal.floor,
                "ceiling": cal.ceiling,
                "eligible_pairs": eligible,
                "saturated": saturated,
                "score_min": min(scores) if scores else None,
                "score_max": max(scores) if scores else None,
                "score_std": statistics.pstdev(scores) if scores else None,
            }
        )
    return rows


# --- Trap cases ---------------------------------------------------------------------------


def _unmet(m: Any) -> str:
    return "; ".join(m.unmet) or "all met"


# (trap, JD, candidate, detail from the candidate's ScoredMatch)
TRAPS = (
    (
        "noah_williams: aliases (Postgres, k8s, Python 3)",
        "python_backend",
        "noah_williams",
        lambda m: (
            "met: "
            + ", ".join(
                f"{r.skill} {months_to_years(r.have_months)} yrs"
                for r in m.check.must_have_results
                if r.met
            )
        ),
    ),
    (
        "hannah_berg: ML only in coursework",
        "ml_engineer",
        "hannah_berg",
        lambda m: f"unmet: {_unmet(m)}; sections: {', '.join(m.matched_sections)}",
    ),
    (
        "emily_clarke: JavaScript is not Java",
        "java_backend",
        "emily_clarke",
        lambda m: f"unmet: {_unmet(m)}",
    ),
    (
        "omar_haddad: listed but never used",
        "python_backend",
        "omar_haddad",
        lambda m: (
            f"unmet: {_unmet(m)}; months: "
            + ", ".join(
                f"{s} {m.check.skill_months.get(s, 0)}"
                for s in ("Python", "Django", "FastAPI", "PostgreSQL")
                if s in m.check.skills
            )
        ),
    ),
    (
        "mateo_garcia: merged Python years",
        "python_backend",
        "mateo_garcia",
        lambda m: (
            f"Python {m.check.skill_months.get('Python', 0)} months "
            f"({months_to_years(m.check.skill_months.get('Python', 0))} yrs, overlaps merged); "
            f"unmet: {_unmet(m)}"
        ),
    ),
)
TRAP_NAMES = (
    "peter_hughes: data_engineer keyword retrieval",
    "peter_hughes: 'python' hobbies chunk",
    *(t[0] for t in TRAPS),
)


def trap_cases(
    collection: ExperimentCollection,
    variants: Mapping[str, RetrievalConfig] | None = None,
    jd_paths: Iterable[str | Path] | None = None,
    qrels: Qrels | None = None,
) -> Rows:
    """Rank, score and detail of each known trap, per retrieval-config variant."""
    variants = variants or {"default": RetrievalConfig()}
    jobs, qrels = _inputs(jd_paths, qrels)
    rows = Rows()

    def row(trap, variant, jd_id, cid, order, rank, score, detail):
        rows.append(
            {
                "trap": trap,
                "variant": variant,
                "jd_id": jd_id,
                "candidate_id": cid,
                "label": qrels.jds[jd_id].labels.get(cid) if jd_id in qrels.jds else None,
                "order": order,
                "rank": rank,
                "score": score,
                "detail": detail,
            }
        )

    for name, config in variants.items():
        # 1. Peter Hughes (HVAC) in data_engineer's keyword list: "airflow", "Spark Energy".
        result = _soft(
            collection, jobs["data_engineer"], replace(config, mode=RetrievalMode.KEYWORD)
        )
        hits = list(result.retrieval.hits) if result.retrieval else []
        hit = next((h for h in hits if h.candidate_id == "peter_hughes"), None)
        row(
            TRAP_NAMES[0],
            name,
            "data_engineer",
            "peter_hughes",
            "keyword retrieval",
            hits.index(hit) + 1 if hit else None,
            hit.keyword.raw if hit else None,
            ", ".join(f"{c.section} {c.keyword_raw:.2f}" for c in hit.top_chunks if c.keyword_raw)
            if hit
            else "no keyword match",
        )
        # 2. Peter Hughes's hobbies chunk ("Python scripting for a Raspberry Pi ...").
        index = collection.keyword_index(config.keyword_mode)
        hobby = next(
            (
                d.chunk_id
                for d in index.docs
                if d.candidate_id == "peter_hughes" and d.section == "hobbies"
            ),
            None,
        )
        found = index.search("python", max(1, len(index)))
        ids = [cid for cid, _ in found]
        score = dict(found).get(hobby)
        weight = config.section_weight("hobbies")
        row(
            TRAP_NAMES[1],
            name,
            "-",
            "peter_hughes",
            f"chunk search 'python' ({index.mode.value})",
            ids.index(hobby) + 1 if hobby in ids else None,
            score,
            f"hobbies weight {weight}: weighted {score * weight:.3f}"
            if score is not None
            else "no match (hobbies carry no skill tokens)",
        )
        # 3-7. Soft-mode match_score rank.
        for trap, jd_id, cid, detail in TRAPS:
            result = _soft(collection, jobs[jd_id], config)
            ranked = [m.candidate_id for m in result.matches]
            m = next((x for x in result.matches if x.candidate_id == cid), None)
            row(
                trap,
                name,
                jd_id,
                cid,
                "soft match_score",
                ranked.index(cid) + 1 if m else None,
                m.match_score if m else None,
                detail(m) if m else "not retrieved",
            )
    return rows


# --- Chunks -------------------------------------------------------------------------------


def chunk_stats(collection: ExperimentCollection) -> Rows:
    """Chunks and words per chunk (min / median / mean / max) by section."""
    found = collection.store.get(include=["documents", "metadatas"])
    words: dict[str, list[int]] = {}
    for doc, meta in zip(found["documents"], found["metadatas"], strict=True):
        words.setdefault(meta["section"], []).append(len((doc or "").split()))
    return Rows(
        {
            "collection": collection.label,
            "section": section,
            "chunks": len(counts),
            "words_min": min(counts),
            "words_median": float(statistics.median(counts)),
            "words_mean": round(statistics.fmean(counts), 2),
            "words_max": max(counts),
        }
        for section, counts in sorted(words.items())
    )


# --- Opening the notebook's collections ---------------------------------------------------


@dataclass(frozen=True)
class CollectionSpec:
    """A collection the notebook uses, and the build-script name that creates it."""

    label: str
    chunker_id: str
    model: str  # Hugging Face model id
    build_name: str  # scripts.build_experiment_collections --only NAME


NOTEBOOK_COLLECTIONS = (
    CollectionSpec("section", "section-v1", "BAAI/bge-small-en-v1.5", "section-bge-small"),
    CollectionSpec("fixed", "fixed-w120-o30", "BAAI/bge-small-en-v1.5", "fixed-bge-small"),
    CollectionSpec(
        "section-noctx", "section-v1-noctx", "BAAI/bge-small-en-v1.5", "section-noctx-bge-small"
    ),
    CollectionSpec(
        "section-minilm", "section-v1", "sentence-transformers/all-MiniLM-L6-v2", "section-minilm"
    ),
    CollectionSpec("section-bge-base", "section-v1", "BAAI/bge-base-en-v1.5", "section-bge-base"),
)


def open_collections(
    specs: Sequence[CollectionSpec] = NOTEBOOK_COLLECTIONS,
    chroma_dir: str | Path = ".chroma",
    embedder_factory: Any = None,
) -> tuple[dict[str, ExperimentCollection | None], list[str]]:
    """``({label: collection or None}, messages)``; a missing collection is never created.

    Embedders are shared between specs with the same model (models load lazily).
    """
    from rag.embedders import get_embedder

    factory = embedder_factory or (lambda model: get_embedder("huggingface", model=model))
    embedders: dict[str, Embedder] = {}
    opened: dict[str, ExperimentCollection | None] = {}
    messages = []
    for spec in specs:
        embedder = embedders.setdefault(spec.model, factory(spec.model))
        name = collection_name(spec.chunker_id, embedder.embedder_id)
        if read_sidecar(chroma_dir, name) is None:
            opened[spec.label] = None
            messages.append(
                f"{spec.label}: not built ({name}); run: "
                f"python -m scripts.build_experiment_collections --only {spec.build_name}"
            )
            continue
        opened[spec.label] = open_collection(spec.label, spec.chunker_id, embedder, chroma_dir)
        messages.append(f"{spec.label}: {name}")
    return opened, messages


# --- Dataset ------------------------------------------------------------------------------


def dataset_counts(personas_dir: str | Path | None = None) -> Rows:
    """Resumes per format, layout and date format."""
    renders = load_resume_renders() if personas_dir is None else load_resume_renders(personas_dir)
    rows = Rows()
    for dimension in ("format", "layout", "date_format"):
        counts: dict[str, int] = {}
        for render in renders.values():
            counts[render[dimension]] = counts.get(render[dimension], 0) + 1
        rows += [
            {"dimension": dimension, "value": value, "resumes": n}
            for value, n in sorted(counts.items())
        ]
    return rows


def label_counts(qrels: Qrels | None = None) -> Rows:
    """S / P / D / none candidates per JD."""
    qrels = qrels or load_qrels()
    return Rows(
        {"jd_id": jd_id}
        | {
            label: sum(1 for v in truth.labels.values() if v == label)
            for label in ("S", "P", "D", "none")
        }
        for jd_id, truth in sorted(qrels.jds.items())
    )


def resume_lengths(directory: str = "resumes") -> Rows:
    """Words of extracted text per resume (read through fs_tools, like indexing)."""
    import fs_tools
    from rag.indexer import default_workspace_root

    rows = Rows()
    with default_workspace_root():
        for entry in sorted(fs_tools.list_files(directory), key=lambda e: e["path"]):
            result = fs_tools.read_file(entry["path"])
            if not result["success"]:
                continue
            rows.append(
                {
                    "candidate_id": Path(entry["path"]).stem,
                    "format": entry["extension"].lstrip("."),
                    "words": len(result["data"]["content"].split()),
                }
            )
    return rows


# --- Extraction ---------------------------------------------------------------------------


def extraction_summary(sidecar: Sidecar, manifest: Manifest | None = None) -> Rows:
    """Layer (c) headline numbers as metric / value rows."""
    e = evaluate_extraction(sidecar, manifest or load_manifest())
    return Rows(
        [
            {"metric": "candidates", "value": e.candidates},
            {"metric": "name exact match", "value": e.name_exact},
            {"metric": "skills precision (micro)", "value": e.skills_micro["precision"]},
            {"metric": "skills recall (micro)", "value": e.skills_micro["recall"]},
            {"metric": "skills F1 (micro)", "value": e.skills_micro["f1"]},
            {"metric": "skills F1 (macro)", "value": e.skills_macro["f1"]},
            {"metric": "total years exact match", "value": e.total_years_exact},
            {"metric": "total years MAE", "value": e.total_years_mae},
            {"metric": "skill years MAE", "value": e.skill_years_mae},
            {"metric": "skill-year pairs compared", "value": e.skill_years_pairs},
            {"metric": "highest education exact match", "value": e.education_exact},
            {"metric": "candidates with a mismatch", "value": len(e.mismatches)},
        ]
    )


def _meta(collection: ExperimentCollection, candidate_id: str) -> CandidateMetadata:
    return CandidateMetadata.from_dict(collection.sidecar.candidates[candidate_id].metadata)


def _chunks(collection: ExperimentCollection, candidate_id: str) -> list[tuple[str, dict]]:
    found = collection.store.get(
        where={"candidate_id": candidate_id}, include=["documents", "metadatas"]
    )
    pairs = sorted(
        zip(found["documents"], found["metadatas"], found["ids"], strict=True),
        key=lambda t: (t[1]["section"], t[1].get("role_index", -1), t[2]),
    )
    return [(doc or "", meta | {"chunk_id": cid}) for doc, meta, cid in pairs]


def extraction_traps(collection: ExperimentCollection) -> Rows:
    """What extraction did with the dataset's deliberate edge cases (from the sidecar)."""
    rows = Rows()

    peter = _meta(collection, "peter_hughes")
    rows.append(
        {
            "trap": "look-alike words are not skills",
            "candidate_id": "peter_hughes",
            "extracted": ", ".join(peter.skills),
            "detail": "HVAC engineer: 'Spark Energy Ltd', 'airflow', a Python hobby: no skill",
        }
    )

    skills_text = " ".join(
        doc for doc, meta in _chunks(collection, "noah_williams") if meta["section"] == "skills"
    )
    aliases = [
        f"{m.matched_text} -> {m.canonical}"
        for m in find_mentions(skills_text)
        if m.matched_text.casefold() != m.canonical.casefold()
    ]
    rows.append(
        {
            "trap": "aliases map to canonical skills",
            "candidate_id": "noah_williams",
            "extracted": ", ".join(aliases),
            "detail": f"{len(aliases)} aliases in the skills section",
        }
    )

    mateo = _meta(collection, "mateo_garcia")
    roles = [
        int(loc.split(":")[1])
        for loc in mateo.skill_evidence.get("Python", ())
        if loc.startswith("experience:")
    ]
    summed = sum(mateo.roles[i].months for i in roles if mateo.roles[i].counts_towards_experience)
    merged = mateo.skill_months.get("Python", 0)
    rows.append(
        {
            "trap": "overlapping roles are merged, not summed",
            "candidate_id": "mateo_garcia",
            "extracted": f"Python {merged} months ({months_to_years(merged)} yrs)",
            "detail": f"summed over {len(roles)} roles: {summed} months "
            f"({months_to_years(summed)} yrs)",
        }
    )

    omar = _meta(collection, "omar_haddad")
    listed = [s for s in ("Python", "Django") if s in omar.skills]
    rows.append(
        {
            "trap": "listed but never used: 0 months",
            "candidate_id": "omar_haddad",
            "extracted": ", ".join(f"{s} {omar.skill_months.get(s, 0)} months" for s in listed),
            "detail": "evidence: "
            + "; ".join(f"{s} in {', '.join(omar.skill_evidence.get(s, ()))}" for s in listed),
        }
    )

    ishaan = _meta(collection, "ishaan_gupta")
    rows.append(
        {
            "trap": "internships do not count as experience",
            "candidate_id": "ishaan_gupta",
            "extracted": f"total {ishaan.total_months} months ({ishaan.total_years} yrs)",
            "detail": "; ".join(
                f"{r.title} ({r.months} months, {'internship' if r.is_internship else 'counted'})"
                for r in ishaan.roles
            ),
        }
    )

    nina = _meta(collection, "nina_kowalski")
    from_skills = [s for s, locs in nina.skill_evidence.items() if "skills" in locs]
    rows.append(
        {
            "trap": "skills found in role bullets",
            "candidate_id": "nina_kowalski",
            "extracted": ", ".join(nina.skills),
            "detail": f"{len(nina.skills)} skills; from a skills section: "
            f"{', '.join(from_skills) or 'none'}",
        }
    )
    return rows


def candidate_chunk_tokens(
    collection: ExperimentCollection, candidate_id: str, contains: str | None = None
) -> Rows:
    """A candidate's chunks with the skill tokens the keyword index gives them (skills mode)."""
    index = collection.keyword_index(TokenizerMode.SKILLS)
    rows = Rows()
    for text, meta in _chunks(collection, candidate_id):
        if contains is not None and contains not in text:
            continue
        rows.append(
            {
                "collection": collection.label,
                "section": meta["section"],
                "role_index": meta.get("role_index"),
                "body_start": meta.get("body_start", 0),
                "matching_lines": [ln for ln in text.split("\n") if contains and contains in ln],
                "text": text,
                "skill_tokens": index.tokens(meta["chunk_id"]),
            }
        )
    return rows


# --- Query source -------------------------------------------------------------------------


def query_token_table(
    embedders: Mapping[str, Embedder], jd_paths: Iterable[str | Path] | None = None
) -> Rows:
    """Per JD and model: tokens of the whole JD file vs its ``query_text``, against the limit."""
    paths = sorted(JD_DIR.glob("*.txt")) if jd_paths is None else sorted(map(Path, jd_paths))
    rows = Rows()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        query = parse_job_description(text).query_text
        for name, embedder in embedders.items():
            whole, part = embedder.token_count(text), embedder.token_count(query)
            rows.append(
                {
                    "jd_id": path.stem,
                    "model": name,
                    "max_tokens": embedder.max_tokens,
                    "whole": whole,
                    "query_text": part,
                    "whole_over": whole > embedder.max_tokens,
                    "query_over": part > embedder.max_tokens,
                }
            )
    return rows
