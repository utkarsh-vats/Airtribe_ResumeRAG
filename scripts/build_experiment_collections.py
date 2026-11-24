"""Build every collection the experiments notebook compares
(``python -m scripts.build_experiment_collections [--only NAME ...] [--as-of 2026-11-01]``).

Sequential and idempotent: indexing skips unchanged files, and a collection is calibrated only
when its sidecar has no calibration (re-indexing a changed file clears it). One model is
loaded at a time and released before the next. It uses the same ``rag`` APIs as the CLIs.

Collections (``--only`` names):

- ``section-bge-small``: section-v1 x BAAI/bge-small-en-v1.5 (the default)
- ``fixed-bge-small``: fixed-w120-o30 x bge-small
- ``section-noctx-bge-small``: section-v1-noctx (no contextual header) x bge-small
- ``section-minilm``: section-v1 x sentence-transformers/all-MiniLM-L6-v2
- ``section-bge-base``: section-v1 x BAAI/bge-base-en-v1.5
- ``section-gemini``: section-v1 x gemini-embedding-001 at 768 dimensions, built only when
  ``GEMINI_API_KEY`` is set (otherwise skipped with a one-line message)
"""

import argparse
import gc
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from rag.calibration import compute_calibration, save_calibration
from rag.chunking import Chunker, FixedSizeChunker, SectionChunker
from rag.cli import (
    DEFAULT_JD_DIRECTORY,
    chroma_dir,
    job_descriptions_in,
    prepare_environment,
    setting,
)
from rag.embedders import from_env, get_embedder
from rag.embedders.base import Embedder
from rag.indexer import index_directory
from rag.jd_parser import parse_job_description
from rag.keyword_index import KeywordIndex
from rag.retrieval import RetrievalConfig
from rag.store import VectorStore
from scripts.persona_schema import ROOT

DEFAULT_AS_OF = "2026-11-01"
BGE_SMALL = "BAAI/bge-small-en-v1.5"


def _hf(model: str) -> Callable[[], Embedder]:
    return lambda: get_embedder("huggingface", model=model)


def _gemini() -> Embedder:
    env = dict(
        os.environ,
        EMBEDDER="gemini",
        GEMINI_EMBEDDING_MODEL="gemini-embedding-001",
        GEMINI_EMBEDDING_DIM="768",
    )
    return from_env(env)


@dataclass(frozen=True)
class Spec:
    """One collection to build."""

    name: str
    chunker: Callable[[], Chunker]
    embedder: Callable[[], Embedder]
    needs_gemini: bool = False


SPECS = (
    Spec("section-bge-small", SectionChunker, _hf(BGE_SMALL)),
    Spec("fixed-bge-small", FixedSizeChunker, _hf(BGE_SMALL)),
    Spec(
        "section-noctx-bge-small", lambda: SectionChunker(contextual_header=False), _hf(BGE_SMALL)
    ),
    Spec("section-minilm", SectionChunker, _hf("sentence-transformers/all-MiniLM-L6-v2")),
    Spec("section-bge-base", SectionChunker, _hf("BAAI/bge-base-en-v1.5")),
    Spec("section-gemini", SectionChunker, _gemini, needs_gemini=True),
)
NAMES = tuple(s.name for s in SPECS)


def build(spec: Spec, embedder: Embedder, as_of: date) -> int:
    """Index and (if needed) calibrate one collection; returns the number of failed files."""
    chunker = spec.chunker()
    started = time.perf_counter()
    embedder.embed_query("warm-up")  # model load, reported apart from indexing
    load_s = time.perf_counter() - started
    store = VectorStore(chroma_dir(), chunker.chunker_id, embedder.embedder_id, embedder.dim)
    started = time.perf_counter()
    report = index_directory("resumes", chunker, embedder, store, as_of)
    index_s = time.perf_counter() - started
    print(
        f"{spec.name}: {store.name}\n"
        f"  model load {load_s:.1f} s; index {index_s:.1f} s ({len(report.added)} added, "
        f"{len(report.updated)} updated, {len(report.unchanged)} unchanged, "
        f"{len(report.failed)} failed, {report.chunks_written} chunks written)"
    )
    if store.load_sidecar().calibration is None:
        config = RetrievalConfig()
        jobs = {
            origin: parse_job_description(text)
            for origin, text in job_descriptions_in(DEFAULT_JD_DIRECTORY).items()
        }
        keyword_index = KeywordIndex.from_store(store, mode=config.keyword_mode)
        started = time.perf_counter()
        calibration = compute_calibration(jobs, store, keyword_index, embedder, config)
        save_calibration(store, calibration)
        print(
            f"  calibrate {time.perf_counter() - started:.1f} s (floor {calibration.floor:.4f}, "
            f"ceiling {calibration.ceiling:.4f}, {calibration.pairs} pairs)"
        )
    else:
        print("  calibration kept (already calibrated)")
    store.close()
    return len(report.failed)


def main(
    argv: list[str] | None = None,
    *,
    embedder_override: Callable[[Spec], Embedder] | None = None,
    load_env: bool = True,
) -> int:
    """Build the selected collections in order; exit 1 if any file failed to index."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", action="append", choices=NAMES, help="build only this one")
    parser.add_argument("--as-of", default=DEFAULT_AS_OF, help="reference date (2026-11-01)")
    args = parser.parse_args(argv)
    prepare_environment(ROOT, load_env)
    as_of = date.fromisoformat(args.as_of)
    failed = 0
    for spec in SPECS:
        if args.only and spec.name not in args.only:
            continue
        if spec.needs_gemini and embedder_override is None and not setting("GEMINI_API_KEY"):
            print(f"{spec.name}: skipped (GEMINI_API_KEY is not set)")
            continue
        embedder = embedder_override(spec) if embedder_override else spec.embedder()
        failed += build(spec, embedder, as_of)
        del embedder
        gc.collect()  # release the model before loading the next one
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
