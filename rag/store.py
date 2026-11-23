"""The vector store: a repository around a persistent Chroma collection, plus its sidecar.

Nothing else in ``rag/`` imports ``chromadb``. Facts checked against the installed
``chromadb==1.5.9``:

- Collection names: 3-512 characters from ``[a-zA-Z0-9._-]``, starting and ending with a
  letter or digit, no ``..``, and not an IPv4 address.
- Cosine distance is set with ``configuration={"hnsw": {"space": "cosine"}}`` (the 1.x
  collection configuration; the old ``metadata={"hnsw:space": ...}`` form still works but is
  legacy).
- ``embedding_function=None`` must be passed on *every* create / get call. Leaving it out of
  ``get_collection`` gives the collection Chroma's default (ONNX MiniLM) embedding function.
  Vectors always come from our own ``Embedder``.
- Metadata values: scalars, or non-empty lists of one scalar type. Empty lists raise, and a
  ``None`` value drops the key, so both are omitted here.

One collection per (chunker, embedder): ``resumes__<chunker_id>__<embedder_id>``. Its
metadata records the schema version, both ids and the vector dimension; opening it with a
different dimension raises ``DimensionMismatchError``. Structured candidate metadata that
Chroma can't hold lives in a JSON sidecar at ``<CHROMA_DIR>/sidecar/<collection>.json``,
written atomically.
"""

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.chunking import Chunk
from rag.metadata import CandidateMetadata

SCHEMA_VERSION = 1
COLLECTION_PREFIX = "resumes"
MAX_NAME = 512
_HASH_CHARS = 12


class StoreError(RuntimeError):
    """The vector store or its sidecar is unusable as configured."""


class DimensionMismatchError(StoreError):
    """An existing collection holds vectors of a different dimension."""


class SidecarError(StoreError):
    """The sidecar file is missing fields, from another collection, or another schema."""


def _sanitise(part: str) -> str:
    part = re.sub(r"[^A-Za-z0-9._-]+", "-", part)
    return re.sub(r"\.{2,}", ".", part)


def collection_name(chunker_id: str, embedder_id: str) -> str:
    """``resumes__<chunker_id>__<embedder_id>``, made valid for Chroma.

    Characters outside ``[A-Za-z0-9._-]`` become ``-`` and runs of dots collapse. A name over
    the 512-character limit is cut and given a short hash of the full name, so it stays unique.
    """
    name = "__".join(_sanitise(p) for p in (COLLECTION_PREFIX, chunker_id, embedder_id))
    name = name.strip("._-")
    if len(name) > MAX_NAME:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:_HASH_CHARS]
        name = f"{name[: MAX_NAME - _HASH_CHARS - 1].rstrip('._-')}-{digest}"
    return name


def chunk_metadata(chunk: Chunk, candidate: CandidateMetadata) -> dict[str, Any]:
    """Flat Chroma metadata for one chunk; keys whose value is None or empty are left out."""
    highest = candidate.highest_education
    values: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "resume_path": candidate.resume_path,
        "name": candidate.name,
        "section": chunk.section,
        "role_index": chunk.role_index,
        "role_title": chunk.role_title,
        "role_company": chunk.role_company,
        "body_start": chunk.body_start,
        "chunker_id": chunk.chunker_id,
        "total_months": candidate.total_months,
        "highest_degree_level": int(highest.level) if highest and highest.level else None,
        "skills": list(candidate.skills),
    }
    return {k: v for k, v in values.items() if v is not None and v != []}


# --- Sidecar ------------------------------------------------------------------------------


@dataclass
class CandidateRecord:
    """One candidate in the sidecar."""

    metadata: dict  # CandidateMetadata.to_dict()
    content_hash: str
    indexed_at: str  # ISO 8601, UTC

    def to_dict(self) -> dict:
        """JSON form of the record."""
        return {
            "metadata": self.metadata,
            "content_hash": self.content_hash,
            "indexed_at": self.indexed_at,
        }


@dataclass
class Sidecar:
    """Everything about a collection that Chroma metadata can't hold."""

    chunker_id: str
    embedder_id: str
    as_of: str  # the as_of date the metadata was computed with (ISO date)
    pipeline_version: str
    candidates: dict[str, CandidateRecord] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    calibration: dict | None = None  # rag.calibration.Calibration.to_dict(); optional key

    def to_dict(self) -> dict:
        """JSON form of the sidecar, candidates sorted by id."""
        return {
            "schema_version": self.schema_version,
            "chunker_id": self.chunker_id,
            "embedder_id": self.embedder_id,
            "as_of": self.as_of,
            "pipeline_version": self.pipeline_version,
            "candidates": {cid: r.to_dict() for cid, r in sorted(self.candidates.items())},
        } | ({"calibration": self.calibration} if self.calibration is not None else {})

    @classmethod
    def from_dict(cls, data: dict) -> Sidecar:
        """Parse a sidecar dict; raises ``SidecarError`` on another schema version."""
        if data.get("schema_version") != SCHEMA_VERSION:
            raise SidecarError(
                f"sidecar schema version {data.get('schema_version')!r} is not {SCHEMA_VERSION}; "
                "rebuild the index"
            )
        try:
            return cls(
                chunker_id=data["chunker_id"],
                embedder_id=data["embedder_id"],
                as_of=data["as_of"],
                pipeline_version=data["pipeline_version"],
                candidates={
                    cid: CandidateRecord(r["metadata"], r["content_hash"], r["indexed_at"])
                    for cid, r in data["candidates"].items()
                },
                calibration=data.get("calibration"),
            )
        except (KeyError, TypeError) as exc:
            raise SidecarError(f"malformed sidecar: {exc!r}") from exc


def write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON to a temp file in the same folder, then ``os.replace`` it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def sidecar_path(chroma_dir: str | Path, name: str) -> Path:
    """Where the sidecar of collection ``name`` lives."""
    return Path(chroma_dir) / "sidecar" / f"{name}.json"


def read_sidecar(chroma_dir: str | Path, name: str) -> Sidecar | None:
    """The sidecar of collection ``name``, or None if it doesn't exist (nothing is created)."""
    path = sidecar_path(chroma_dir, name)
    if not path.exists():
        return None
    return Sidecar.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --- Store --------------------------------------------------------------------------------


def _client(chroma_dir: Path) -> Any:
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(chroma_dir), settings=Settings(anonymized_telemetry=False)
    )


class VectorStore:
    """One Chroma collection for one (chunker, embedder) pair, and its sidecar."""

    def __init__(
        self,
        chroma_dir: str | Path,
        chunker_id: str,
        embedder_id: str,
        dim: int,
        client: Any = None,
    ):
        """Open the collection, creating it if needed; check it matches this configuration."""
        self.chroma_dir = Path(chroma_dir)
        self.chunker_id = chunker_id
        self.embedder_id = embedder_id
        self.dim = dim
        self.name = collection_name(chunker_id, embedder_id)
        self._client = client or _client(self.chroma_dir)
        self._collection = self._open()

    @property
    def sidecar_path(self) -> Path:
        """``<CHROMA_DIR>/sidecar/<collection>.json``."""
        return sidecar_path(self.chroma_dir, self.name)

    def _open(self) -> Any:
        collection = self._client.get_or_create_collection(
            name=self.name,
            embedding_function=None,  # never let Chroma construct its default embedder
            configuration={"hnsw": {"space": "cosine"}},
            metadata={
                "schema_version": SCHEMA_VERSION,
                "chunker_id": self.chunker_id,
                "embedder_id": self.embedder_id,
                "dim": self.dim,
            },
        )
        meta = collection.metadata or {}
        if meta.get("dim") != self.dim:
            raise DimensionMismatchError(
                f"collection {self.name!r} holds {meta.get('dim')}-dimensional vectors, "
                f"not {self.dim}; rebuild it or use another embedder"
            )
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise StoreError(
                f"collection {self.name!r} has schema version {meta.get('schema_version')!r}, "
                f"not {SCHEMA_VERSION}; rebuild it"
            )
        if (meta.get("chunker_id"), meta.get("embedder_id")) != (self.chunker_id, self.embedder_id):
            raise StoreError(f"collection {self.name!r} belongs to another configuration: {meta}")
        return collection

    # --- Chunks -------------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        candidate: CandidateMetadata,
    ) -> int:
        """Store chunks (``chunk.text`` as the document) with their vectors; returns the count."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return 0
        for vector in vectors:
            if len(vector) != self.dim:
                raise DimensionMismatchError(
                    f"vector of dimension {len(vector)}, expected {self.dim}"
                )
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[list(v) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[chunk_metadata(c, candidate) for c in chunks],
        )
        return len(chunks)

    def chunk_ids(self, candidate_id: str) -> list[str]:
        """Ids of a candidate's chunks, sorted."""
        found = self._collection.get(where={"candidate_id": candidate_id}, include=[])
        return sorted(found["ids"])

    def delete_candidate(self, candidate_id: str) -> int:
        """Remove a candidate's chunks; returns how many there were."""
        ids = self.chunk_ids(candidate_id)
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def update_candidate_metadata(
        self, chunks: Sequence[Chunk], candidate: CandidateMetadata
    ) -> None:
        """Rewrite the metadata of existing chunks (no re-embedding)."""
        if chunks:
            self._collection.update(
                ids=[c.chunk_id for c in chunks],
                metadatas=[chunk_metadata(c, candidate) for c in chunks],
            )

    def count(self) -> int:
        """Number of chunks in the collection."""
        return int(self._collection.count())

    def list_candidates(self) -> list[str]:
        """Candidate ids with at least one chunk, sorted."""
        found = self._collection.get(include=["metadatas"])
        return sorted({m["candidate_id"] for m in found["metadatas"] if m})

    def get(self, **kwargs: Any) -> dict:
        """Thin pass-through to ``collection.get`` (tests and diagnostics)."""
        return self._collection.get(**kwargs)

    def query(self, vector: Sequence[float], n: int, where: dict | None = None) -> dict:
        """Nearest chunks by cosine distance; Chroma's raw result for one query."""
        return self._collection.query(
            query_embeddings=[list(vector)],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def drop(self) -> None:
        """Delete the collection and its sidecar. The store can't be used afterwards."""
        self._client.delete_collection(self.name)
        self.sidecar_path.unlink(missing_ok=True)

    def reset(self) -> None:
        """Drop everything and start an empty collection with the same configuration."""
        self.drop()
        self._collection = self._open()

    # --- Sidecar ------------------------------------------------------------------------

    def load_sidecar(self) -> Sidecar | None:
        """The sidecar, or None if there isn't one yet."""
        sidecar = read_sidecar(self.chroma_dir, self.name)
        if sidecar is None:
            return None
        if (sidecar.chunker_id, sidecar.embedder_id) != (self.chunker_id, self.embedder_id):
            raise SidecarError(f"{self.sidecar_path} belongs to another configuration")
        return sidecar

    def save_sidecar(self, sidecar: Sidecar) -> None:
        """Write the sidecar atomically."""
        write_json_atomic(self.sidecar_path, sidecar.to_dict())


# --- Inspection ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectionStats:
    """A summary of one collection and its sidecar."""

    name: str
    chunker_id: str | None
    embedder_id: str | None
    dim: int | None
    schema_version: int | None
    chunks: int
    candidates: int
    chunks_by_section: dict[str, int]
    as_of: str | None  # from the sidecar
    pipeline_version: str | None  # from the sidecar


def collection_stats(chroma_dir: str | Path, client: Any = None) -> list[CollectionStats]:
    """Every collection under ``chroma_dir``, sorted by name (nothing is created)."""
    if client is None and not Path(chroma_dir).exists():
        return []
    client = client or _client(Path(chroma_dir))
    stats = []
    for listed in sorted(client.list_collections(), key=lambda c: c.name):
        collection = client.get_collection(listed.name, embedding_function=None)
        meta = collection.metadata or {}
        found = collection.get(include=["metadatas"])
        sections: dict[str, int] = {}
        candidates = set()
        for item in found["metadatas"]:
            sections[item.get("section", "?")] = sections.get(item.get("section", "?"), 0) + 1
            candidates.add(item.get("candidate_id"))
        try:
            sidecar = read_sidecar(chroma_dir, listed.name)
        except SidecarError:
            sidecar = None
        stats.append(
            CollectionStats(
                name=listed.name,
                chunker_id=meta.get("chunker_id"),
                embedder_id=meta.get("embedder_id"),
                dim=meta.get("dim"),
                schema_version=meta.get("schema_version"),
                chunks=len(found["ids"]),
                candidates=len(candidates - {None}),
                chunks_by_section=dict(sorted(sections.items())),
                as_of=sidecar.as_of if sidecar else None,
                pipeline_version=sidecar.pipeline_version if sidecar else None,
            )
        )
    return stats
