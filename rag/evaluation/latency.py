"""Latency measurement: per-stage query latency, index latency, and the environment.

**Percentiles** use linear interpolation between closest ranks (numpy's default ``linear``
method): for sorted values ``v`` and ``p`` in 0-100, position ``(n - 1) * p / 100``,
interpolated between its neighbours. ``p50`` is the median, ``p95`` lies between the two
largest of 20 samples, and a single sample is every percentile.

**Query latency** (``measure_query_latency``): for each JD x retrieval mode x filter mode, one
query is ``parse_job_description`` + ``match()`` + ``build_output`` (excerpts and reasoning
included), timed per stage:

    parse, filter, embed_query, semantic_search, keyword_search, aggregate_fuse, backfill,
    scoring, output, total

``warmup`` runs per combination are discarded, then ``repeat`` runs are recorded. Each stage
gets p50 / p95 / mean / max in milliseconds per JD; the macro figure for a (mode, filter)
pair is the mean of the per-JD figures. The model is loaded once, before any query, and its
load time is reported separately (``model_load_ms``), never inside a query. The timer is
injectable, so tests can make every number exact.

**Index latency** (``measure_index_latency``): indexes the resumes from scratch into a
temporary ``CHROMA_DIR`` (never the real one), then again (every file unchanged), and reports
per-stage totals, per-file p50 / p95 per stage, the unchanged re-run and embedding throughput.
The temporary directory is removed afterwards, also when indexing raises.
"""

import importlib.metadata
import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from rag.calibration import Calibration
from rag.calibration import percentile as _percentile
from rag.chunking import Chunker
from rag.embedders.base import Embedder
from rag.indexer import StageTimings, index_directory
from rag.jd_parser import parse_job_description
from rag.matcher import FilterMode, match
from rag.output import build_output
from rag.retrieval import RetrievalConfig
from rag.store import VectorStore

QUERY_STAGES = (
    "parse",
    "filter",
    "embed_query",
    "semantic_search",
    "keyword_search",
    "aggregate_fuse",
    "backfill",
    "scoring",
    "output",
    "total",
)
INDEX_STAGES = ("read", "parse_extract", "chunk", "embed", "upsert", "total")
FILTER_MODES = ("strict", "soft")
REPO_ROOT = Path(__file__).resolve().parents[2]

Timer = Callable[[], float]


def percentile(values: Sequence[float], p: float) -> float:
    """The ``p``-th percentile (0-100), linear interpolation (numpy's default method)."""
    return _percentile(values, p)


def summarise(values_ms: Sequence[float]) -> dict[str, float]:
    """p50, p95, mean and max of millisecond values, rounded to 0.001 ms."""
    return {
        "p50": round(percentile(values_ms, 50), 3),
        "p95": round(percentile(values_ms, 95), 3),
        "mean": round(sum(values_ms) / len(values_ms), 3),
        "max": round(max(values_ms), 3),
    }


def _dataframe(rows: list[dict]) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a dev dependency
        raise ImportError("to_dataframe() needs pandas (pip install pandas)") from exc
    return pd.DataFrame(rows)


# --- Query latency ------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryLatency:
    """Per-stage query latency, per JD and macro over JDs, per (mode, filter mode)."""

    model_load_ms: float
    repeat: int
    warmup: int
    rows: tuple[dict, ...]  # jd_id, mode, filter_mode, stage, p50, p95, mean, max
    macro: dict[tuple[str, str], dict[str, dict[str, float]]]  # (mode, filter) -> stage -> stats

    def to_dict(self) -> dict:
        """JSON form; macro keys become ``"mode/filter"``."""
        return {
            "model_load_ms": self.model_load_ms,
            "repeat": self.repeat,
            "warmup": self.warmup,
            "macro": {f"{m}/{f}": stages for (m, f), stages in self.macro.items()},
            "per_jd": [dict(r) for r in self.rows],
        }

    def to_dataframe(self) -> Any:
        """The per-JD rows."""
        return _dataframe([dict(r) for r in self.rows])


def _one_query(
    text: str,
    store: Any,
    keyword_index: Any,
    embedder: Embedder,
    sidecar: Any,
    config: RetrievalConfig,
    filter_mode: str,
    calibration: Calibration | None,
    timer: Timer,
) -> dict[str, float]:
    start = timer()
    job = parse_job_description(text)
    parsed = timer()
    result = match(
        job,
        store,
        keyword_index,
        embedder,
        sidecar,
        config,
        filter_mode,
        calibration=calibration,
        clock=timer,
    )
    matched = timer()
    build_output(result, job)
    done = timer()
    r = result.timings.retrieval
    return {
        "parse": parsed - start,
        "filter": result.timings.filter,
        "embed_query": r.embed_query,
        "semantic_search": r.semantic_search,
        "keyword_search": r.keyword_search,
        "aggregate_fuse": r.aggregate_fuse,
        "backfill": r.backfill,
        "scoring": result.timings.scoring,
        "output": done - matched,
        "total": done - start,
    }


def measure_query_latency(
    jds: Mapping[str, str],
    store: Any,
    keyword_index: Any,
    embedder: Embedder,
    config: RetrievalConfig | None = None,
    modes: Sequence[str] = ("semantic", "keyword", "hybrid"),
    filter_modes: Sequence[str] = FILTER_MODES,
    repeat: int = 5,
    warmup: int = 1,
    timer: Timer = time.perf_counter,
    calibration: Calibration | None = None,
) -> QueryLatency:
    """Per-stage latency of full queries; ``jds`` maps a JD id to its text."""
    if repeat < 1 or warmup < 0:
        raise ValueError("need repeat >= 1 and warmup >= 0")
    base = config or RetrievalConfig()
    sidecar = store.load_sidecar()
    if sidecar is None:
        raise ValueError(f"no sidecar for {store.name}; index the collection first")
    for name in filter_modes:
        FilterMode(name)
    start = timer()
    embedder.embed_query("warm-up")  # loads the model; never inside a measured query
    model_load_ms = (timer() - start) * 1000

    rows = []
    macro: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for mode in modes:
        mode_config = replace(base, mode=mode)
        for filter_mode in filter_modes:
            per_jd: dict[str, dict[str, dict[str, float]]] = {}
            for jd_id, text in jds.items():
                args = (text, store, keyword_index, embedder, sidecar, mode_config, filter_mode)
                for _ in range(warmup):
                    _one_query(*args, calibration, timer)
                samples = [_one_query(*args, calibration, timer) for _ in range(repeat)]
                per_jd[jd_id] = {
                    stage: summarise([s[stage] * 1000 for s in samples]) for stage in QUERY_STAGES
                }
                for stage in QUERY_STAGES:
                    rows.append(
                        {"jd_id": jd_id, "mode": mode, "filter_mode": filter_mode, "stage": stage}
                        | per_jd[jd_id][stage]
                    )
            macro[(mode, filter_mode)] = {
                stage: {
                    stat: round(sum(per_jd[j][stage][stat] for j in per_jd) / len(per_jd), 3)
                    for stat in ("p50", "p95", "mean", "max")
                }
                for stage in QUERY_STAGES
            }
    return QueryLatency(round(model_load_ms, 3), repeat, warmup, tuple(rows), macro)


# --- Index latency ------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexLatency:
    """Indexing from scratch, then re-indexing with nothing changed."""

    files: int
    chunks: int
    totals_ms: dict[str, float]  # per stage, all files
    per_file: dict[str, dict[str, float]]  # stage -> p50/p95/mean/max over files (ms)
    first_run_ms: float  # wall time of the first run
    rerun_ms: float  # wall time of the unchanged re-run
    rerun_unchanged: int
    embed_chunks_per_second: float | None
    failed: int

    def to_dict(self) -> dict:
        """JSON form."""
        return dict(self.__dict__)


def _stage_ms(timings: StageTimings) -> dict[str, float]:
    values = {stage: getattr(timings, stage) * 1000 for stage in INDEX_STAGES[:-1]}
    values["total"] = timings.total * 1000
    return values


def measure_index_latency(
    resume_dir: str,
    chunker: Chunker,
    embedder: Embedder,
    as_of: date,
    timer: Timer = time.perf_counter,
    temp_root: str | Path | None = None,
) -> IndexLatency:
    """Index ``resume_dir`` (workspace-relative) into a temporary Chroma dir, twice."""
    embedder.embed_query("warm-up")  # model load is not indexing time
    tmp = Path(tempfile.mkdtemp(prefix="resume-rag-index-", dir=temp_root))
    store = None
    try:
        store = VectorStore(tmp / "chroma", chunker.chunker_id, embedder.embedder_id, embedder.dim)
        start = timer()
        first = index_directory(resume_dir, chunker, embedder, store, as_of)
        first_ms = (timer() - start) * 1000
        start = timer()
        second = index_directory(resume_dir, chunker, embedder, store, as_of)
        rerun_ms = (timer() - start) * 1000
    finally:
        if store is not None:
            store.close()
        shutil.rmtree(tmp, ignore_errors=True)
    files = [_stage_ms(t) for t in first.file_timings.values()]
    per_file = {stage: summarise([f[stage] for f in files]) for stage in INDEX_STAGES if files}
    embed_seconds = first.timings.embed
    return IndexLatency(
        files=len(first.file_timings),
        chunks=first.chunks_written,
        totals_ms={k: round(v, 3) for k, v in _stage_ms(first.timings).items()},
        per_file=per_file,
        first_run_ms=round(first_ms, 3),
        rerun_ms=round(rerun_ms, 3),
        rerun_unchanged=len(second.unchanged),
        embed_chunks_per_second=(
            round(first.chunks_written / embed_seconds, 1) if embed_seconds else None
        ),
        failed=len(first.failed),
    )


# --- Environment --------------------------------------------------------------------------


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(run: Callable[..., Any] = subprocess.run, cwd: Path = REPO_ROOT) -> str | None:
    """The short commit hash of ``cwd``, or None without git or outside a checkout."""
    try:
        done = run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    commit = (done.stdout or "").strip()
    return commit if done.returncode == 0 and commit else None


def environment_info(
    embedder: Embedder,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Where and with what the numbers were measured."""
    device = getattr(embedder, "device", None)  # HuggingFace only (imports torch)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch_device": device,
        "embedder_id": embedder.embedder_id,
        "model": embedder.model,
        "chromadb": _version("chromadb"),
        "sentence_transformers": _version("sentence-transformers"),
        "git_commit": git_commit(run),
        "timestamp": now().isoformat(timespec="seconds"),
    }
