"""Retrieval report against the qrels (``python -m scripts.retrieval_report``).

Tooling, not part of the pipeline: it reads ``data/ground_truth/qrels.json``. For each JD and
each mode (semantic, keyword, hybrid) on one indexed collection it prints the top 10 with
their labels (S / P / D, ``.`` for none) and one line per JD: S candidates in the top 10,
D candidates in the top 10, and the rank of every S candidate (``-`` if the mode never ranks
it). The pool is widened to every indexed candidate so all ranks are known.

The collection must already exist (``python resume_rag.py index ...``). Defaults: section
chunker and the huggingface (bge-small) embedder.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from rag.chunking import available_chunkers, get_chunker
from rag.embedders import available_embedders, from_env, get_embedder
from rag.jd_parser import parse_job_description
from rag.keyword_index import KeywordIndex
from rag.retrieval import RetrievalConfig, RetrievalMode, retrieve
from rag.store import VectorStore, collection_name, read_sidecar
from scripts.persona_schema import ROOT

JD_DIR = ROOT / "data" / "job_descriptions"
QRELS = ROOT / "data" / "ground_truth" / "qrels.json"
TOP = 10
MODES = (RetrievalMode.SEMANTIC, RetrievalMode.KEYWORD, RetrievalMode.HYBRID)


def make_embedder(name: str, model: str | None):
    if name == "gemini":
        load_dotenv(ROOT / ".env")
        env = dict(os.environ, EMBEDDER="gemini")
        if model:
            env["GEMINI_EMBEDDING_MODEL"] = model
        return from_env(env)
    return get_embedder(name, **({"model": model} if model else {}))


def labels() -> dict[str, dict[str, str]]:
    qrels = json.loads(QRELS.read_text(encoding="utf-8"))
    return {
        jd["jd_id"]: {j["candidate_id"]: j["label"] for j in jd["judgements"]}
        for jd in qrels["jds"]
    }


def mark(label: str) -> str:
    return label if label in ("S", "P", "D") else "."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunker", default="section", choices=available_chunkers())
    parser.add_argument("--embedder", default="huggingface", choices=available_embedders())
    parser.add_argument("--model", help="embedding model id (huggingface or gemini)")
    parser.add_argument("--query-source", default="query_text", choices=["query_text", "full_text"])
    parser.add_argument("--chroma-dir", default=os.environ.get("CHROMA_DIR") or "./.chroma")
    parser.add_argument("--jd", action="append", help="only these JD ids (repeatable)")
    args = parser.parse_args(argv)

    chunker = get_chunker(args.chunker)
    embedder = make_embedder(args.embedder, args.model)
    name = collection_name(chunker.chunker_id, embedder.embedder_id)
    sidecar = read_sidecar(Path(args.chroma_dir), name)
    if sidecar is None:
        print(f"no index for {name} in {args.chroma_dir}; run resume_rag.py index", file=sys.stderr)
        return 2
    store = VectorStore(args.chroma_dir, chunker.chunker_id, embedder.embedder_id, embedder.dim)
    keyword_index = KeywordIndex.from_store(store)
    qrels = labels()
    pool = len(sidecar.candidates)
    print(f"collection: {name}  ({store.count()} chunks, {pool} candidates)")
    started = time.perf_counter()
    embedder.embed_query("warm-up")  # load the model now, so per-mode timings exclude it
    print(
        f"query source: {args.query_source}   keyword index: {keyword_index.mode.value} "
        f"(built in {keyword_index.build_seconds * 1000:.0f} ms)   "
        f"model load: {time.perf_counter() - started:.1f} s"
    )

    summary = []
    for path in sorted(JD_DIR.glob("*.txt")):
        jd_id = path.stem
        if args.jd and jd_id not in args.jd:
            continue
        job = parse_job_description(path.read_text(encoding="utf-8"))
        jd_labels = qrels[jd_id]
        strong = sorted(c for c, label in jd_labels.items() if label == "S")
        print(f"\n=== {jd_id}: {job.title}")
        print(f"    keyword skills: {', '.join(job.critical_skills) or '-'}")
        for mode in MODES:
            config = RetrievalConfig(mode=mode, query_source=args.query_source, pool_size=pool)
            result = retrieve(job, store, keyword_index, embedder, config)
            print(f"  -- {mode.value}  ({result.timings.total * 1000:.0f} ms)")
            print(
                f"     {'#':>2} lbl {'candidate':<17} {'fused':>7}  {'sem':>6} {'r':>3}  "
                f"{'bm25':>6} {'r':>3}  sections"
            )
            for rank, hit in enumerate(result.hits[:TOP], start=1):
                sem, kw = hit.semantic, hit.keyword
                print(
                    f"     {rank:>2}  {mark(jd_labels.get(hit.candidate_id, ''))}  "
                    f"{hit.candidate_id:<17} {hit.fused_score:7.4f}  "
                    f"{sem.raw:6.3f} {sem.rank or '-':>3}  {kw.raw:6.2f} {kw.rank or '-':>3}  "
                    f"{','.join(hit.matched_sections)}"
                )
            order = [h.candidate_id for h in result.hits]
            top = order[:TOP]
            s_found = sum(1 for c in strong if c in top)
            d_found = sum(1 for c in top if jd_labels.get(c) == "D")
            s_ranks = ", ".join(f"{c} {order.index(c) + 1 if c in order else '-'}" for c in strong)
            line = (
                f"{jd_id:<15} {mode.value:<8} S in top {TOP}: {s_found}/{len(strong)}  "
                f"D in top {TOP}: {d_found}  S ranks: {s_ranks}"
            )
            print(f"     {line}")
            summary.append(line)
            for note in result.notes:
                print(f"     note: {note}")

    print("\nSummary")
    for line in summary:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
