"""One full evaluation run as a JSON-ready report (``job_matcher.py evaluate``).

Order matters for honest timings: query latency runs first, so it measures the model load
on its own; then layers (a)-(c) and end to end; index latency (optional) runs last, in a
temporary Chroma directory.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime
from typing import Any

from rag.calibration import Calibration
from rag.chunking import Chunker
from rag.embedders.base import Embedder
from rag.evaluation.evaluate import (
    evaluate_end_to_end,
    evaluate_extraction,
    evaluate_filter,
    evaluate_ranking,
)
from rag.evaluation.ground_truth import Manifest, Qrels
from rag.evaluation.latency import environment_info, measure_index_latency, measure_query_latency
from rag.jd_parser import parse_job_description
from rag.retrieval import RetrievalConfig
from rag.scoring import ScoreWeights

STRICT_P5_MAX = "strict P@5 is at most 0.6: each JD has 3 S candidates"


def run_evaluation(
    jd_texts: Mapping[str, str],
    qrels: Qrels,
    manifest: Manifest,
    store: Any,
    keyword_index: Any,
    embedder: Embedder,
    calibration: Calibration,
    *,
    collection: str,
    config: RetrievalConfig | None = None,
    modes: Sequence[str] = ("semantic", "keyword", "hybrid"),
    repeat: int = 5,
    warmup: int = 1,
    index_latency: tuple[str, Chunker, date] | None = None,  # (resume dir, chunker, as_of)
    warnings: Sequence[str] = (),
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    """Every evaluation layer plus latency; ``jd_texts`` maps JD ids to their text."""
    config = config or RetrievalConfig()
    sidecar = store.load_sidecar()
    query = measure_query_latency(
        jd_texts,
        store,
        keyword_index,
        embedder,
        config,
        modes,
        repeat=repeat,
        warmup=warmup,
        calibration=calibration,
    )
    jobs = {jd_id: parse_job_description(text) for jd_id, text in sorted(jd_texts.items())}
    ranking = evaluate_ranking(
        jobs, qrels, store, keyword_index, embedder, config, calibration=calibration, modes=modes
    )
    filtering = evaluate_filter(sidecar, jobs, qrels)
    extraction = evaluate_extraction(sidecar, manifest)
    end_to_end = evaluate_end_to_end(
        jobs, qrels, store, keyword_index, embedder, config, calibration=calibration
    )
    index = None
    if index_latency is not None:
        resume_dir, chunker, as_of = index_latency
        index = measure_index_latency(resume_dir, chunker, embedder, as_of).to_dict()
    return {
        "environment": environment_info(embedder, now),
        "config": {
            "collection": collection,
            "chunker_id": store.chunker_id,
            "embedder_id": embedder.embedder_id,
            "index_as_of": sidecar.as_of,
            "retrieval": config.to_dict(),
            "scoring_weights": asdict(ScoreWeights()),
            "modes": list(modes),
            "repeat": repeat,
            "warmup": warmup,
            "jds": sorted(jd_texts),
        },
        "calibration": calibration.to_dict(),
        "warnings": list(warnings),
        "notes": [STRICT_P5_MAX],
        "ranking": ranking.to_dict(),
        "filter": filtering.to_dict(),
        "extraction": extraction.to_dict(),
        "end_to_end": end_to_end.to_dict(),
        "latency": {"query": query.to_dict(), "index": index},
    }
