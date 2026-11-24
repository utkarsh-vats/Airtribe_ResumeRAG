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
and ``AS_OF_DATE`` (blank = today). The CLIs are the only place that reads today's date; it
is passed into the pipeline from here.

Exit codes: 0 success; 1 indexing finished but some files failed; 2 usage or configuration
error (one line on stderr, no traceback), including running on a Python older than 3.14.
"""

import sys

# Before any other import: the project needs Python 3.14 (this file must still parse on
# older versions so the message can be shown).
if sys.version_info < (3, 14):
    found = ".".join(str(part) for part in sys.version_info[:3])
    sys.stderr.write(
        "resume_rag.py: error: Python 3.14 or newer is required, but this is Python "
        + found
        + " ("
        + str(sys.executable)
        + "). Activate .venv first: .venv\\Scripts\\activate on Windows, "
        "source .venv/bin/activate elsewhere.\n"
    )
    raise SystemExit(2)

import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from fs_tools import FsToolError
from rag.calibration import (
    DEFAULT_CEILING_PCT,
    DEFAULT_FLOOR_PCT,
    check_percentiles,
    compute_calibration,
    save_calibration,
)
from rag.cli import (
    DEFAULT_JD_DIRECTORY,
    ArgumentParser,
    UsageError,
    add_configuration,
    chroma_dir,
    job_descriptions_in,
    make_chunker,
    make_embedder,
    prepare_environment,
    require_index,
    resolve_as_of,
)
from rag.console import (
    make_console,
    render_candidate,
    render_candidates,
    render_index_report,
    render_stats,
)
from rag.indexer import index_directory
from rag.jd_parser import parse_job_description
from rag.keyword_index import KeywordIndex
from rag.metadata import CandidateMetadata
from rag.retrieval import RetrievalConfig
from rag.store import StoreError, VectorStore, collection_stats

ROOT = Path(__file__).resolve().parent
DEFAULT_DIRECTORY = "resumes"


def build_parser() -> ArgumentParser:
    """The argument parser; errors raise ``UsageError`` instead of exiting."""
    parser = ArgumentParser(prog="resume_rag.py", description="Index resumes into Chroma.")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=ArgumentParser)

    index = commands.add_parser("index", help="index resumes from WORKSPACE_ROOT")
    index.add_argument("--dir", default=DEFAULT_DIRECTORY, help="workspace-relative directory")
    add_configuration(index)
    index.add_argument("--rebuild", action="store_true", help="drop and recreate the collection")
    index.add_argument("--as-of", help="reference date YYYY-MM-DD (default: $AS_OF_DATE or today)")
    index.add_argument("--json", action="store_true", help="print the IndexReport as JSON")

    commands.add_parser("stats", help="list every collection under CHROMA_DIR")

    candidates = commands.add_parser("candidates", help="list indexed candidates")
    add_configuration(candidates)

    show = commands.add_parser("show", help="show one candidate's metadata")
    show.add_argument("candidate_id")
    add_configuration(show)
    show.add_argument("--json", action="store_true", help="print the metadata as JSON")

    calibrate = commands.add_parser(
        "calibrate", help="compute the semantic floor/ceiling from a folder of JDs (no labels)"
    )
    calibrate.add_argument(
        "--jd-dir", default=DEFAULT_JD_DIRECTORY, help="workspace-relative folder of JDs"
    )
    add_configuration(calibrate)
    calibrate.add_argument(
        "--floor-pct", type=float, default=DEFAULT_FLOOR_PCT, help="floor percentile (50)"
    )
    calibrate.add_argument(
        "--ceiling-pct", type=float, default=DEFAULT_CEILING_PCT, help="ceiling percentile (99)"
    )
    calibrate.add_argument("--json", action="store_true", help="print the calibration as JSON")
    return parser


def _load_records(args) -> tuple[str, dict[str, dict]]:
    chunker = make_chunker(args.chunker, args.chunker_opt)
    embedder = make_embedder(args.embedder, args.model)
    name, sidecar = require_index(chunker, embedder)
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
    name, _ = require_index(chunker, embedder)
    jobs = {
        origin: parse_job_description(text)
        for origin, text in job_descriptions_in(args.jd_dir).items()
    }
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
    prepare_environment(ROOT, load_env)
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
