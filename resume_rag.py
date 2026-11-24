"""Part A: index resumes into a Chroma vector store, and inspect what was indexed.

Usage::

    python resume_rag.py index [--dir resumes] [--chunker section|fixed] [--chunker-opt k=v ...]
                               [--embedder fake|huggingface|gemini] [--model ID] [--rebuild]
                               [--as-of YYYY-MM-DD] [--json]
    python resume_rag.py stats
    python resume_rag.py candidates [--chunker ...] [--embedder ...] [--model ID]
    python resume_rag.py show CANDIDATE_ID [--chunker ...] [--embedder ...] [--model ID] [--json]
    python resume_rag.py calibrate [--jd-dir job_descriptions] [--chunker ...] [--embedder ...]
                                   [--model ID] [--floor-pct 50] [--ceiling-pct 99] [--json]

Settings come from CLI flags, then the environment (``.env`` is loaded), then defaults:
``WORKSPACE_ROOT`` (./data), ``CHROMA_DIR`` (./.chroma), ``EMBEDDER`` and its model settings,
and ``AS_OF_DATE`` (blank = today). This is the only place that reads today's date; it is
passed into the pipeline from here.

Exit codes: 0 success; 1 indexing finished but some files failed; 2 usage or configuration
error (one line on stderr, no traceback).
"""

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

import fs_tools
from fs_tools import FsToolError
from rag.calibration import (
    DEFAULT_CEILING_PCT,
    DEFAULT_FLOOR_PCT,
    check_percentiles,
    compute_calibration,
    save_calibration,
)
from rag.chunking import available_chunkers, get_chunker
from rag.console import (
    make_console,
    render_candidate,
    render_candidates,
    render_index_report,
    render_stats,
)
from rag.embedders import available_embedders, from_env, get_embedder
from rag.embedders.base import Embedder
from rag.indexer import DEFAULT_WORKSPACE_ROOT, index_directory
from rag.jd_parser import (
    JD_EXTENSIONS,
    JobDescriptionError,
    load_job_description,
    parse_job_description,
)
from rag.keyword_index import KeywordIndex
from rag.metadata import CandidateMetadata
from rag.retrieval import RetrievalConfig
from rag.store import StoreError, VectorStore, collection_name, collection_stats, read_sidecar

ROOT = Path(__file__).resolve().parent
DEFAULT_CHROMA_DIR = "./.chroma"
DEFAULT_DIRECTORY = "resumes"
DEFAULT_JD_DIRECTORY = "job_descriptions"
CHUNKER_DEFAULT = "section"


class UsageError(Exception):
    """A usage or configuration problem: reported in one line, exit code 2."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    """The argument parser; errors raise ``UsageError`` instead of exiting."""
    parser = _Parser(prog="resume_rag.py", description="Index resumes into Chroma.")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    def configuration(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--chunker", choices=available_chunkers(), default=CHUNKER_DEFAULT)
        sub.add_argument(
            "--chunker-opt",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="chunker parameter, e.g. max_words=200 or size=200 (repeatable)",
        )
        sub.add_argument("--embedder", choices=available_embedders(), help="default: $EMBEDDER")
        sub.add_argument("--model", help="embedding model id (huggingface or gemini)")

    index = commands.add_parser("index", help="index resumes from WORKSPACE_ROOT")
    index.add_argument("--dir", default=DEFAULT_DIRECTORY, help="workspace-relative directory")
    configuration(index)
    index.add_argument("--rebuild", action="store_true", help="drop and recreate the collection")
    index.add_argument("--as-of", help="reference date YYYY-MM-DD (default: $AS_OF_DATE or today)")
    index.add_argument("--json", action="store_true", help="print the IndexReport as JSON")

    commands.add_parser("stats", help="list every collection under CHROMA_DIR")

    candidates = commands.add_parser("candidates", help="list indexed candidates")
    configuration(candidates)

    show = commands.add_parser("show", help="show one candidate's metadata")
    show.add_argument("candidate_id")
    configuration(show)
    show.add_argument("--json", action="store_true", help="print the metadata as JSON")

    calibrate = commands.add_parser(
        "calibrate", help="compute the semantic floor/ceiling from a folder of JDs (no labels)"
    )
    calibrate.add_argument(
        "--jd-dir", default=DEFAULT_JD_DIRECTORY, help="workspace-relative folder of JDs"
    )
    configuration(calibrate)
    calibrate.add_argument(
        "--floor-pct", type=float, default=DEFAULT_FLOOR_PCT, help="floor percentile (50)"
    )
    calibrate.add_argument(
        "--ceiling-pct", type=float, default=DEFAULT_CEILING_PCT, help="ceiling percentile (99)"
    )
    calibrate.add_argument("--json", action="store_true", help="print the calibration as JSON")
    return parser


# --- Configuration ------------------------------------------------------------------------


def _setting(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


def resolve_as_of(flag: str | None, today: Callable[[], date]) -> date:
    """``--as-of``, else ``AS_OF_DATE``, else ``today()``."""
    text = (flag or "").strip() or _setting("AS_OF_DATE")
    if not text:
        return today()
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise UsageError(f"invalid as-of date {text!r}; expected YYYY-MM-DD") from None


def _option_value(text: str) -> int | bool | str:
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        return text


def make_chunker(name: str, options: Sequence[str]):
    """A chunker from ``--chunker`` and its ``--chunker-opt KEY=VALUE`` options."""
    params = {}
    for option in options:
        key, sep, value = option.partition("=")
        if not sep or not key.strip():
            raise UsageError(f"--chunker-opt expects KEY=VALUE, got {option!r}")
        params[key.strip().replace("-", "_")] = _option_value(value.strip())
    try:
        return get_chunker(name, **params)
    except TypeError as exc:
        raise UsageError(f"bad option for the {name} chunker: {exc}") from None
    except ValueError as exc:
        raise UsageError(f"{name} chunker: {exc}") from None


def make_embedder(name: str | None, model: str | None) -> Embedder:
    """``--embedder`` / ``--model`` over ``EMBEDDER`` and the backend's env settings."""
    env = dict(os.environ)
    if name:
        env["EMBEDDER"] = name
    chosen = (env.get("EMBEDDER") or "").strip() or "huggingface"
    if chosen not in available_embedders():
        raise UsageError(f"unknown embedder {chosen!r}; available: {available_embedders()}")
    if model:
        env["HF_EMBEDDING_MODEL" if chosen == "huggingface" else "GEMINI_EMBEDDING_MODEL"] = model
    if chosen == "fake":
        return get_embedder("fake")
    try:
        return from_env(env)
    except ValueError as exc:  # includes GeminiConfigError
        raise UsageError(str(exc)) from None


def chroma_dir() -> Path:
    """``CHROMA_DIR`` or ``./.chroma``."""
    return Path(_setting("CHROMA_DIR", DEFAULT_CHROMA_DIR))


def _load_records(args) -> tuple[str, dict[str, dict]]:
    chunker = make_chunker(args.chunker, args.chunker_opt)
    embedder = make_embedder(args.embedder, args.model)
    name = collection_name(chunker.chunker_id, embedder.embedder_id)
    sidecar = read_sidecar(chroma_dir(), name)
    if sidecar is None:
        raise UsageError(f"no index for collection {name!r} in {chroma_dir()}; run index first")
    return name, {cid: record.metadata for cid, record in sidecar.candidates.items()}


# --- Commands -----------------------------------------------------------------------------


def cmd_index(args, console, today: Callable[[], date]) -> int:
    """Index ``--dir``; exit code 1 if any file failed."""
    as_of = resolve_as_of(args.as_of, today)
    chunker = make_chunker(args.chunker, args.chunker_opt)
    embedder = make_embedder(args.embedder, args.model)
    store = VectorStore(chroma_dir(), chunker.chunker_id, embedder.embedder_id, embedder.dim)
    report = index_directory(args.dir, chunker, embedder, store, as_of, rebuild=args.rebuild)
    if args.json:
        payload = {"as_of": as_of.isoformat(), **report.to_dict()}
        console.file.write(json.dumps(payload, indent=2) + "\n")
    else:
        console.print(f"[bold]as_of:[/bold] {as_of.isoformat()}")
        render_index_report(report, console)
    return 1 if report.failed else 0


def cmd_stats(args, console) -> int:
    """Every collection under ``CHROMA_DIR``."""
    render_stats(collection_stats(chroma_dir()), console)
    return 0


def cmd_candidates(args, console) -> int:
    """The candidates in one collection's sidecar."""
    name, records = _load_records(args)
    render_candidates(name, records, console)
    return 0


def cmd_calibrate(args, console) -> int:
    """Calibrate the collection's semantic scores against every JD in ``--jd-dir``."""
    try:
        check_percentiles(args.floor_pct, args.ceiling_pct)
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    chunker = make_chunker(args.chunker, args.chunker_opt)
    embedder = make_embedder(args.embedder, args.model)
    name = collection_name(chunker.chunker_id, embedder.embedder_id)
    if read_sidecar(chroma_dir(), name) is None:
        raise UsageError(f"no index for collection {name!r} in {chroma_dir()}; run index first")
    entries = sorted(fs_tools.list_files(args.jd_dir), key=lambda e: e["path"])
    jobs = {}
    for entry in entries:
        if entry["extension"] in JD_EXTENSIONS:
            try:
                text, origin = load_job_description(entry["path"])
            except JobDescriptionError as exc:
                raise UsageError(f"{entry['path']}: {exc}") from None
            jobs[origin] = parse_job_description(text)
    if not jobs:
        raise UsageError(f"no .txt, .pdf or .docx job descriptions in {args.jd_dir!r}")
    store = VectorStore(chroma_dir(), chunker.chunker_id, embedder.embedder_id, embedder.dim)
    config = RetrievalConfig()
    keyword_index = KeywordIndex.from_store(store, mode=config.keyword_mode)
    calibration = compute_calibration(
        jobs,
        store,
        keyword_index,
        embedder,
        config,
        floor_pct=args.floor_pct,
        ceiling_pct=args.ceiling_pct,
    )
    save_calibration(store, calibration)
    if args.json:
        console.file.write(json.dumps(calibration.to_dict(), indent=2) + "\n")
    else:
        console.print(f"[bold]Collection:[/bold] {name}")
        console.print(
            f"Semantic calibration from {calibration.pairs} JD x candidate pairs "
            f"({len(jobs)} JDs): floor (p{args.floor_pct:g}) {calibration.floor:.4f}, "
            f"ceiling (p{args.ceiling_pct:g}) {calibration.ceiling:.4f}"
        )
        console.print("JDs: " + ", ".join(calibration.jd_files))
    return 0


def cmd_show(args, console) -> int:
    """One candidate's metadata from the sidecar."""
    _, records = _load_records(args)
    record = records.get(args.candidate_id)
    if record is None:
        raise UsageError(f"unknown candidate {args.candidate_id!r}")
    if args.json:
        console.file.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    else:
        render_candidate(CandidateMetadata.from_dict(record), console)
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    today: Callable[[], date] = date.today,
    load_env: bool = True,
) -> int:
    """Run the CLI; returns the exit code. ``today`` is injectable for tests."""
    if load_env:
        load_dotenv(ROOT / ".env", override=False)  # real environment variables win
    if not _setting("WORKSPACE_ROOT"):
        os.environ["WORKSPACE_ROOT"] = DEFAULT_WORKSPACE_ROOT
    console = make_console()
    try:
        args = build_parser().parse_args(argv)
        if args.command == "index":
            return cmd_index(args, console, today)
        commands = {
            "stats": cmd_stats,
            "candidates": cmd_candidates,
            "show": cmd_show,
            "calibrate": cmd_calibrate,
        }
        return commands[args.command](args, console)
    except (UsageError, StoreError, FsToolError) as exc:
        print(f"resume_rag.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
