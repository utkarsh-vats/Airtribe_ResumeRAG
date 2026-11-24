"""The three evaluation layers, plus end to end.

- **(a) Ranking** (``evaluate_ranking``): for each JD and retrieval mode, every candidate is
  ranked with *no* must-have filter, in two orders: ``retrieval`` (the fused RRF rank) and
  ``scored`` (the soft-mode ``match_score``). The ranking is measured against the graded
  qrels before the filter, because strict filtering would hide how well the ranker itself
  separates strong, partial and distractor candidates.
- **(b) Filter** (``evaluate_filter``): the system's must-have eligibility, computed from the
  *extracted* metadata, against qrels' ``eligible``.
- **(c) Extraction** (``evaluate_extraction``): the extracted metadata against the manifest
  (names, skills, years, highest education).
- **End to end** (``evaluate_end_to_end``): strict-mode top k per JD, i.e. what the CLI
  returns.

Every result has ``to_dict()`` and ``to_dataframe()``. pandas is imported lazily, only by
``to_dataframe``.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from rag.calibration import Calibration
from rag.embedders.base import Embedder
from rag.evaluation.ground_truth import Manifest, Qrels
from rag.evaluation.metrics import (
    LENIENT,
    STRICT,
    distractor_intrusion_at_k,
    macro_average,
    mrr,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
)
from rag.jd_parser import JobRequirements, parse_job_description
from rag.matcher import FilterMode, match
from rag.metadata import CandidateMetadata
from rag.requirements_check import check_candidate
from rag.retrieval import RetrievalConfig
from rag.scoring import ScoreWeights

MODES = ("semantic", "keyword", "hybrid")
ORDERS = ("retrieval", "scored")
CUTOFFS = (5, 10)


def _dataframe(rows: list[dict]) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a dev dependency
        raise ImportError("to_dataframe() needs pandas (pip install pandas)") from exc
    return pd.DataFrame(rows)


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def load_jobs(
    jd_paths: Iterable[str | Path] | Mapping[str, JobRequirements],
) -> dict[str, JobRequirements]:
    """``{jd_id: JobRequirements}`` for JD files on disk (the id is the file stem).

    An already parsed ``{jd_id: JobRequirements}`` mapping is passed through.
    """
    if isinstance(jd_paths, Mapping):
        return dict(sorted(jd_paths.items()))
    return {
        Path(p).stem: parse_job_description(Path(p).read_text(encoding="utf-8"))
        for p in sorted(jd_paths, key=lambda p: Path(p).stem)
    }


def _metadata(sidecar: Any) -> dict[str, CandidateMetadata]:
    return {
        cid: CandidateMetadata.from_dict(record.metadata)
        for cid, record in sidecar.candidates.items()
    }


# --- (a) Ranking --------------------------------------------------------------------------


def ranking_metrics(
    ranking: Sequence[str], grades: Mapping[str, int], labels: Mapping[str, str]
) -> dict[str, float | None]:
    """Every ranking metric for one ranked list."""
    values: dict[str, float | None] = {}
    for name, threshold in (("strict", STRICT), ("lenient", LENIENT)):
        for k in CUTOFFS:
            values[f"P@{k}_{name}"] = precision_at_k(ranking, grades, k, threshold)
            values[f"R@{k}_{name}"] = recall_at_k(ranking, grades, k, threshold)
        values[f"MRR_{name}"] = mrr(ranking, grades, threshold)
        values[f"R-Prec_{name}"] = r_precision(ranking, grades, threshold)
    for k in CUTOFFS:
        values[f"nDCG@{k}"] = ndcg_at_k(ranking, grades, k)
        values[f"intrusion@{k}"] = distractor_intrusion_at_k(ranking, labels, k)
    return values


@dataclass(frozen=True)
class RankingEvaluation:
    """Per-JD rows and macro averages per (mode, order)."""

    rows: tuple[dict, ...]  # jd_id, mode, order, ranked, then every metric
    macro: dict[tuple[str, str], dict[str, float | None]]

    def to_dict(self) -> dict:
        """JSON form; macro keys become ``"mode/order"``."""
        return {
            "rows": [dict(r) for r in self.rows],
            "macro": {f"{mode}/{order}": dict(v) for (mode, order), v in self.macro.items()},
        }

    def to_dataframe(self) -> Any:
        """The per-JD rows as a DataFrame."""
        return _dataframe([dict(r) for r in self.rows])

    def macro_dataframe(self) -> Any:
        """The macro averages as a DataFrame, one row per (mode, order)."""
        return _dataframe([{"mode": m, "order": o} | dict(v) for (m, o), v in self.macro.items()])


def evaluate_ranking(
    jd_paths: Iterable[str | Path] | Mapping[str, JobRequirements],
    qrels: Qrels,
    store: Any,
    keyword_index: Any,
    embedder: Embedder,
    retrieval_config: RetrievalConfig | None = None,
    scoring_weights: ScoreWeights | None = None,
    calibration: Calibration | None = None,
    modes: Sequence[str] = MODES,
) -> RankingEvaluation:
    """Layer (a): rank every candidate, unfiltered, per JD and mode, in both orders."""
    base = retrieval_config or RetrievalConfig()
    sidecar = store.load_sidecar()
    if sidecar is None:
        raise ValueError(f"no sidecar for {store.name}; index the collection first")
    total = len(sidecar.candidates)
    rows = []
    for jd_id, job in load_jobs(jd_paths).items():
        if jd_id not in qrels.jds:
            raise ValueError(f"{jd_id!r} has no judgements in the qrels")
        truth = qrels.jds[jd_id]
        for mode in modes:
            result = match(
                job,
                store,
                keyword_index,
                embedder,
                sidecar,
                replace(base, mode=mode),
                FilterMode.SOFT,
                k=total,
                weights=scoring_weights,
                calibration=calibration,
            )
            orders = {
                "retrieval": [h.candidate_id for h in result.retrieval.hits]
                if result.retrieval
                else [],
                "scored": [m.candidate_id for m in result.matches],
            }
            for order, ranking in orders.items():
                rows.append(
                    {"jd_id": jd_id, "mode": mode, "order": order, "ranked": len(ranking)}
                    | ranking_metrics(ranking, truth.grades, truth.labels)
                )
    metric_names = [k for k in rows[0] if k not in ("jd_id", "mode", "order", "ranked")]
    macro = {}
    for mode in modes:
        for order in ORDERS:
            group = [r for r in rows if r["mode"] == mode and r["order"] == order]
            macro[(mode, order)] = {m: macro_average(r[m] for r in group) for m in metric_names}
    return RankingEvaluation(tuple(rows), macro)


# --- (b) Filter ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterCounts:
    """A confusion matrix for eligibility (positive = eligible)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float | None:
        """TP / (TP + FP)."""
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        """TP / (TP + FN)."""
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def accuracy(self) -> float | None:
        """(TP + TN) / all."""
        return _ratio(self.tp + self.tn, self.tp + self.fp + self.fn + self.tn)

    def to_dict(self) -> dict:
        """JSON form with the derived rates."""
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True)
class FilterDisagreement:
    """One (JD, candidate) where the system and the qrels disagree on eligibility."""

    jd_id: str
    candidate_id: str
    system_eligible: bool
    qrels_eligible: bool
    system_unmet: tuple[str, ...]
    qrels_unmet: tuple[str, ...]


@dataclass(frozen=True)
class FilterEvaluation:
    """Layer (b): per-JD and overall confusion matrices, and every disagreement."""

    per_jd: dict[str, FilterCounts]
    overall: FilterCounts
    disagreements: tuple[FilterDisagreement, ...]
    missing: tuple[str, ...] = ()  # judged candidates absent from the sidecar

    @property
    def agreements(self) -> int:
        """Pairs where both sides agree."""
        return self.overall.tp + self.overall.tn

    @property
    def pairs(self) -> int:
        """Pairs compared."""
        o = self.overall
        return o.tp + o.fp + o.fn + o.tn

    def to_dict(self) -> dict:
        """JSON form."""
        return {
            "overall": self.overall.to_dict()
            | {"agreements": self.agreements, "pairs": self.pairs},
            "per_jd": {jd: c.to_dict() for jd, c in self.per_jd.items()},
            "disagreements": [asdict(d) for d in self.disagreements],
            "missing": list(self.missing),
        }

    def to_dataframe(self) -> Any:
        """One row per JD plus ``overall``."""
        rows = [{"jd_id": jd} | c.to_dict() for jd, c in self.per_jd.items()]
        return _dataframe([*rows, {"jd_id": "overall"} | self.overall.to_dict()])


def _count(counts: FilterCounts, system: bool, truth: bool) -> FilterCounts:
    return FilterCounts(
        counts.tp + (system and truth),
        counts.fp + (system and not truth),
        counts.fn + (not system and truth),
        counts.tn + (not system and not truth),
    )


def evaluate_filter(
    sidecar: Any, jds: Mapping[str, JobRequirements], qrels: Qrels
) -> FilterEvaluation:
    """Layer (b): eligibility from extracted metadata vs qrels ``eligible``."""
    metas = _metadata(sidecar)
    per_jd: dict[str, FilterCounts] = {}
    overall = FilterCounts()
    disagreements = []
    missing: set[str] = set()
    for jd_id, job in sorted(jds.items()):
        truth = qrels.jds[jd_id]
        counts = FilterCounts()
        for cid in sorted(truth.eligible):
            if cid not in metas:
                missing.add(cid)
                continue
            check = check_candidate(metas[cid], job)
            counts = _count(counts, check.eligible, truth.eligible[cid])
            overall = _count(overall, check.eligible, truth.eligible[cid])
            if check.eligible != truth.eligible[cid]:
                disagreements.append(
                    FilterDisagreement(
                        jd_id,
                        cid,
                        check.eligible,
                        truth.eligible[cid],
                        check.unmet,
                        truth.unmet[cid],
                    )
                )
        per_jd[jd_id] = counts
    return FilterEvaluation(per_jd, overall, tuple(disagreements), tuple(sorted(missing)))


# --- (c) Extraction -----------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionEvaluation:
    """Layer (c): extracted metadata against the manifest."""

    candidates: int
    name_exact: float
    skills_micro: dict[str, float | None]  # precision, recall, f1
    skills_macro: dict[str, float | None]
    total_years_exact: float
    total_years_mae: float
    skill_years_mae: float | None  # over skills present on both sides
    skill_years_pairs: int
    education_exact: float
    mismatches: dict[str, tuple[str, ...]]  # candidate id -> "field: expected ... got ..."
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """JSON form."""
        data = asdict(self)
        data["mismatches"] = {c: list(m) for c, m in self.mismatches.items()}
        data["missing"] = list(self.missing)
        return data

    def to_dataframe(self) -> Any:
        """One row per mismatch: candidate id and description."""
        return _dataframe(
            [{"candidate_id": c, "mismatch": m} for c, ms in self.mismatches.items() for m in ms]
        )


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = (
        None
        if precision is None or recall is None
        else (2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _education(meta: CandidateMetadata) -> dict | None:
    e = meta.highest_education
    if e is None:
        return None
    return {
        "degree": e.degree,
        "field": e.field,
        "institution": e.institution,
        "end_year": e.end_year,
    }


def evaluate_extraction(sidecar: Any, manifest: Manifest) -> ExtractionEvaluation:
    """Layer (c): names, skills, years and highest education against the manifest."""
    metas = _metadata(sidecar)
    mismatches: dict[str, list[str]] = {}
    names = total_exact = education = 0
    total_errors: list[float] = []
    skill_errors: list[float] = []
    tp = fp = fn = 0
    per_candidate: list[dict[str, float | None]] = []
    missing = []
    for cid, want in sorted(manifest.candidates.items()):
        if cid not in metas:
            missing.append(cid)
            continue
        got = metas[cid]
        notes = mismatches.setdefault(cid, [])
        if got.name == want.name:
            names += 1
        else:
            notes.append(f"name: expected {want.name!r}, got {got.name!r}")
        expected, found = set(want.skills), set(got.skills)
        tp, fp, fn = (
            tp + len(expected & found),
            fp + len(found - expected),
            fn + len(expected - found),
        )
        per_candidate.append(
            _prf(len(expected & found), len(found - expected), len(expected - found))
        )
        if found - expected:
            notes.append(f"skills: unexpected {sorted(found - expected)}")
        if expected - found:
            notes.append(f"skills: missing {sorted(expected - found)}")
        total_errors.append(abs(got.total_years - want.total_years))
        if got.total_years == want.total_years:
            total_exact += 1
        else:
            notes.append(f"total_years: expected {want.total_years}, got {got.total_years}")
        for skill in sorted(expected & found):
            error = abs(got.skill_years.get(skill, 0.0) - want.skill_years[skill])
            skill_errors.append(error)
            if error:
                notes.append(
                    f"skill_years[{skill}]: expected {want.skill_years[skill]}, "
                    f"got {got.skill_years.get(skill, 0.0)}"
                )
        if _education(got) == want.highest_education:
            education += 1
        else:
            notes.append(
                f"highest_education: expected {want.highest_education}, got {_education(got)}"
            )
        if not notes:
            del mismatches[cid]
    n = len(manifest.candidates) - len(missing)
    macro = {
        key: macro_average(p[key] for p in per_candidate) for key in ("precision", "recall", "f1")
    }
    return ExtractionEvaluation(
        candidates=n,
        name_exact=names / n if n else 0.0,
        skills_micro=_prf(tp, fp, fn),
        skills_macro=macro,
        total_years_exact=total_exact / n if n else 0.0,
        total_years_mae=sum(total_errors) / n if n else 0.0,
        skill_years_mae=macro_average(skill_errors),
        skill_years_pairs=len(skill_errors),
        education_exact=education / n if n else 0.0,
        mismatches={c: tuple(m) for c, m in mismatches.items()},
        missing=tuple(missing),
    )


# --- End to end ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EndToEndEvaluation:
    """Strict-mode top k per JD against the S labels."""

    k: int
    rows: tuple[dict, ...]  # jd_id, eligible, returned, returned_s, s_total, precision_s, recall_s
    macro: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON form."""
        return {"k": self.k, "rows": [dict(r) for r in self.rows], "macro": dict(self.macro)}

    def to_dataframe(self) -> Any:
        """One row per JD."""
        return _dataframe([dict(r) for r in self.rows])


def evaluate_end_to_end(
    jd_paths: Iterable[str | Path] | Mapping[str, JobRequirements],
    qrels: Qrels,
    store: Any,
    keyword_index: Any,
    embedder: Embedder,
    retrieval_config: RetrievalConfig | None = None,
    scoring_weights: ScoreWeights | None = None,
    calibration: Calibration | None = None,
    k: int = 10,
) -> EndToEndEvaluation:
    """Strict top k per JD: how many are returned, how many of them are S, how many S are."""
    sidecar = store.load_sidecar()
    if sidecar is None:
        raise ValueError(f"no sidecar for {store.name}; index the collection first")
    rows = []
    for jd_id, job in load_jobs(jd_paths).items():
        truth = qrels.jds[jd_id]
        result = match(
            job,
            store,
            keyword_index,
            embedder,
            sidecar,
            retrieval_config,
            FilterMode.STRICT,
            k=k,
            weights=scoring_weights,
            calibration=calibration,
        )
        strong = truth.with_label("S")
        returned = [m.candidate_id for m in result.matches]
        hits = sum(1 for cid in returned if cid in strong)
        rows.append(
            {
                "jd_id": jd_id,
                "eligible": result.eligible_count,
                "returned": len(returned),
                "returned_s": hits,
                "s_total": len(strong),
                "precision_s": _ratio(hits, len(returned)),
                "recall_s": _ratio(hits, len(strong)),
            }
        )
    macro = {
        key: macro_average(r[key] for r in rows) for key in ("precision_s", "recall_s", "returned")
    }
    return EndToEndEvaluation(k, tuple(rows), macro)
