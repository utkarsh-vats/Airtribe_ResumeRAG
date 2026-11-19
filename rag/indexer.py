"""Indexing pipeline: resumes in a workspace directory -> chunks and vectors in a VectorStore.

Per file: ``fs_tools.read_file`` -> clean / parse -> ``extract_metadata`` -> chunk ->
``embed_documents`` (the chunks' ``embed_text``) -> upsert. Files are listed and read only
through ``fs_tools``, so the Milestone 1 sandbox applies; ``fs_tools`` reads
``WORKSPACE_ROOT`` on every call and falls back to its own ``./sample_data`` when unset, so
indexing defaults an unset ``WORKSPACE_ROOT`` to ``./data`` for the duration of the run.

Change detection: a file's content hash covers its extracted text and ``PIPELINE_VERSION``.
An unchanged file is not re-chunked or re-embedded (its metadata is still recomputed, so a new
``as_of`` refreshes durations in place). A changed file's old chunks are deleted before the
new ones are stored. Candidates whose file is gone are pruned. A file that fails is recorded
with an error code and the run carries on.
"""

import hashlib
import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import PurePosixPath

import fs_tools
from rag.chunking import Chunker
from rag.embedders.base import Embedder
from rag.metadata import CandidateMetadata, extract_metadata
from rag.sections import parse_resume
from rag.store import CandidateRecord, Sidecar, VectorStore

logger = logging.getLogger(__name__)

# Bump whenever parsing, metadata extraction or chunking output changes for the same input
# text: it is part of every content hash, so a bump re-indexes every file.
PIPELINE_VERSION = "1"
INDEXED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})
DEFAULT_WORKSPACE_ROOT = "./data"


@contextmanager
def default_workspace_root(default: str = DEFAULT_WORKSPACE_ROOT) -> Iterator[None]:
    """Set ``WORKSPACE_ROOT`` to ``default`` while inside, if it is unset or blank."""
    if os.environ.get("WORKSPACE_ROOT"):
        yield
        return
    previous = os.environ.get("WORKSPACE_ROOT")
    os.environ["WORKSPACE_ROOT"] = default
    try:
        yield
    finally:
        if previous is None:
            del os.environ["WORKSPACE_ROOT"]
        else:
            os.environ["WORKSPACE_ROOT"] = previous


def content_hash(text: str) -> str:
    """SHA-256 of the pipeline version and the extracted text."""
    return hashlib.sha256(f"{PIPELINE_VERSION}\n{text}".encode()).hexdigest()


@dataclass
class StageTimings:
    """Seconds spent per stage."""

    read: float = 0.0
    parse_extract: float = 0.0
    chunk: float = 0.0
    embed: float = 0.0
    upsert: float = 0.0

    def add(self, other: StageTimings) -> None:
        """Add another file's timings to these."""
        for name in ("read", "parse_extract", "chunk", "embed", "upsert"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @property
    def total(self) -> float:
        """All stages together."""
        return self.read + self.parse_extract + self.chunk + self.embed + self.upsert


@dataclass(frozen=True)
class FileIssue:
    """A file that was skipped or failed, and why."""

    path: str
    code: str
    message: str


@dataclass
class IndexReport:
    """What one indexing run did. Lists hold workspace-relative resume paths."""

    collection: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)  # candidate ids
    skipped: list[FileIssue] = field(default_factory=list)
    failed: list[FileIssue] = field(default_factory=list)
    metadata_refreshed: list[str] = field(default_factory=list)  # unchanged, new metadata
    chunks_written: int = 0
    timings: StageTimings = field(default_factory=StageTimings)
    file_timings: dict[str, StageTimings] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """A JSON-ready dict (step 13 reads the timings)."""
        return asdict(self)


class _Failed(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _timed(timings: StageTimings, stage: str) -> Callable[[], None]:
    start = time.perf_counter()

    def stop() -> None:
        setattr(timings, stage, getattr(timings, stage) + time.perf_counter() - start)

    return stop


def index_directory(
    directory: str,
    chunker: Chunker,
    embedder: Embedder,
    store: VectorStore,
    as_of: date,
    rebuild: bool = False,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> IndexReport:
    """Index the resumes in a workspace directory (non-recursive) into ``store``."""
    if (store.chunker_id, store.embedder_id) != (chunker.chunker_id, embedder.embedder_id):
        raise ValueError(
            f"store {store.name!r} is for {store.chunker_id} / {store.embedder_id}, not "
            f"{chunker.chunker_id} / {embedder.embedder_id}"
        )
    if rebuild:
        store.reset()
    sidecar = store.load_sidecar()
    if sidecar is None:
        sidecar = Sidecar(
            chunker.chunker_id, embedder.embedder_id, as_of.isoformat(), PIPELINE_VERSION
        )
    sidecar.as_of = as_of.isoformat()
    sidecar.pipeline_version = PIPELINE_VERSION
    report = IndexReport(collection=store.name)

    with default_workspace_root():
        entries = sorted(fs_tools.list_files(directory), key=lambda e: e["path"])
        listed_ids: set[str] = set()
        for entry in entries:
            path = entry["path"]
            if entry["extension"] not in INDEXED_EXTENSIONS:
                report.skipped.append(
                    FileIssue(
                        path,
                        "UNSUPPORTED_FORMAT",
                        f"{entry['extension'] or 'no extension'} is not indexed",
                    )
                )
                continue
            candidate_id = PurePosixPath(path).stem
            if candidate_id in listed_ids:
                report.failed.append(
                    FileIssue(
                        path, "DUPLICATE_CANDIDATE", f"another file already has id {candidate_id!r}"
                    )
                )
                continue
            listed_ids.add(candidate_id)
            timings = StageTimings()
            try:
                _index_file(path, chunker, embedder, store, sidecar, as_of, now, report, timings)
            except _Failed as failure:
                report.failed.append(FileIssue(path, failure.code, failure.message))
            except Exception as exc:  # one bad file never aborts the run
                logger.exception("indexing %s failed", path)
                report.failed.append(FileIssue(path, type(exc).__name__, str(exc)))
            report.file_timings[path] = timings
            report.timings.add(timings)

    for candidate_id in sorted(
        (set(sidecar.candidates) | set(store.list_candidates())) - listed_ids
    ):
        store.delete_candidate(candidate_id)
        sidecar.candidates.pop(candidate_id, None)
        report.removed.append(candidate_id)
    store.save_sidecar(sidecar)
    return report


def _index_file(
    path: str,
    chunker: Chunker,
    embedder: Embedder,
    store: VectorStore,
    sidecar: Sidecar,
    as_of: date,
    now: Callable[[], datetime],
    report: IndexReport,
    timings: StageTimings,
) -> None:
    stop = _timed(timings, "read")
    result = fs_tools.read_file(path)
    stop()
    if not result["success"]:
        raise _Failed(result["error"]["code"], result["error"]["message"])
    text = result["data"]["content"]
    if not text.strip():
        warnings = result["data"]["metadata"].get("warnings") or ["no text extracted"]
        raise _Failed("EMPTY_TEXT", "; ".join(warnings))

    stop = _timed(timings, "parse_extract")
    parsed = parse_resume(text)
    if not parsed.sections:
        stop()
        raise _Failed("NO_SECTIONS", "no section headings or body text found")
    metadata = extract_metadata(path, parsed, as_of)
    stop()

    digest = content_hash(text)
    previous = sidecar.candidates.get(metadata.candidate_id)
    stop = _timed(timings, "chunk")
    chunks = chunker.chunk(path, parsed)
    stop()
    if not chunks:
        raise _Failed("NO_CHUNKS", "the chunker produced no chunks")

    if (
        previous is not None
        and previous.content_hash == digest
        and previous.metadata.get("resume_path") == path
        and store.chunk_ids(metadata.candidate_id) == sorted(c.chunk_id for c in chunks)
    ):
        report.unchanged.append(path)
        if previous.metadata != metadata.to_dict():  # e.g. a new as_of changed durations
            stop = _timed(timings, "upsert")
            store.update_candidate_metadata(chunks, metadata)
            stop()
            sidecar.candidates[metadata.candidate_id] = _record(metadata, digest, now)
            report.metadata_refreshed.append(path)
        return

    stop = _timed(timings, "embed")
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    stop()
    stop = _timed(timings, "upsert")
    store.delete_candidate(metadata.candidate_id)  # also clears leftovers of an interrupted run
    report.chunks_written += store.upsert_chunks(chunks, vectors, metadata)
    stop()
    sidecar.candidates[metadata.candidate_id] = _record(metadata, digest, now)
    (report.updated if previous is not None else report.added).append(path)


def _record(
    metadata: CandidateMetadata, digest: str, now: Callable[[], datetime]
) -> CandidateRecord:
    return CandidateRecord(metadata.to_dict(), digest, now().isoformat(timespec="seconds"))
