"""Part B: match job descriptions against the indexed resumes.

Usage::

    python job_matcher.py match (--jd PATH | --jd-text TEXT) [-k 10] [--filter strict|soft]
                                [--mode hybrid|semantic|keyword] [--chunker ...] [--embedder ...]
                                [--model ID] [--as-of YYYY-MM-DD] [--format full|brief]
                                [--out PATH] [--json]
    python job_matcher.py batch [--jd-dir job_descriptions] [--out-dir results] [same options]
    python job_matcher.py parse (--jd PATH | --jd-text TEXT)
    python job_matcher.py evaluate [--chunker ...] [--embedder ...] [--model ID]
                                   [--modes semantic,keyword,hybrid] [--repeat 5]
                                   [--measure-indexing] [--out PATH] [--json]

``--jd`` and ``--jd-dir`` are relative to ``WORKSPACE_ROOT`` (./data), read through
``fs_tools`` (.txt, .pdf or .docx); ``--out`` and ``--out-dir`` are ordinary paths. Settings
follow ``resume_rag.py``: flags, then the environment (``.env`` is loaded), then defaults;
``AS_OF_DATE`` blank means today, and only differs from the index's ``as_of`` with a warning.

Exit codes: 0 success; 2 usage or configuration error (no index for the chunker/embedder, an
unreadable JD, a bad option, missing Gemini settings, Python older than 3.14). A missing
semantic calibration is a warning, not an error.
"""

import sys

# Before any other import: the project needs Python 3.14 (this file must still parse on
# older versions so the message can be shown).
if sys.version_info < (3, 14):
    found = ".".join(str(part) for part in sys.version_info[:3])
    sys.stderr.write(
        "job_matcher.py: error: Python 3.14 or newer is required, but this is Python "
        + found
        + " ("
        + str(sys.executable)
        + "). Activate .venv first: .venv\\Scripts\\activate on Windows, "
        "source .venv/bin/activate elsewhere.\n"
    )
    raise SystemExit(2)

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

from fs_tools import FsToolError
from rag.calibration import load_calibration
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
from rag.console import make_console, render_batch_summary, render_evaluation, render_match
from rag.embedders.base import Embedder
from rag.evaluation.ground_truth import GroundTruthError, load_manifest, load_qrels
from rag.evaluation.report import run_evaluation
from rag.jd_parser import JobDescriptionError, parse_job_description, read_job_description
from rag.keyword_index import KeywordIndex
from rag.matcher import DEFAULT_K, FilterMode, match
from rag.output import OutputFormat, build_output
from rag.retrieval import RetrievalConfig, RetrievalMode
from rag.store import Sidecar, StoreError, VectorStore, write_json_atomic

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = "results"
PROG = "job_matcher.py"


def _jd_source(sub: ArgumentParser) -> None:
    group = sub.add_mutually_exclusive_group(required=True)
    group.add_argument("--jd", help="JD file, relative to WORKSPACE_ROOT (.txt, .pdf, .docx)")
    group.add_argument("--jd-text", help="the JD itself, as text")


def _match_options(sub: ArgumentParser) -> None:
    sub.add_argument("-k", type=int, default=DEFAULT_K, help=f"matches to return ({DEFAULT_K})")
    sub.add_argument(
        "--filter",
        choices=[m.value for m in FilterMode],
        default=FilterMode.STRICT.value,
        help="strict: must-haves filter before ranking; soft: rank everyone",
    )
    sub.add_argument(
        "--mode",
        choices=[m.value for m in RetrievalMode],
        default=RetrievalMode.HYBRID.value,
        help="retrieval lists to fuse",
    )
    add_configuration(sub)
    sub.add_argument("--as-of", help="reference date YYYY-MM-DD (default: $AS_OF_DATE or today)")
    sub.add_argument(
        "--format",
        choices=[f.value for f in OutputFormat],
        default=OutputFormat.FULL.value,
        help="JSON shape: full (default) or exactly the brief's",
    )


def build_parser() -> ArgumentParser:
    """The argument parser; errors raise ``UsageError`` instead of exiting."""
    parser = ArgumentParser(prog=PROG, description="Match job descriptions to indexed resumes.")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=ArgumentParser)

    one = commands.add_parser("match", help="match one JD")
    _jd_source(one)
    _match_options(one)
    one.add_argument("--out", help="write the JSON output here (directories are created)")
    one.add_argument("--json", action="store_true", help="print the JSON output")

    batch = commands.add_parser("batch", help="match every JD in a folder; one JSON per JD")
    batch.add_argument(
        "--jd-dir", default=DEFAULT_JD_DIRECTORY, help="workspace-relative folder of JDs"
    )
    batch.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="folder for the JSON files")
    _match_options(batch)

    evaluate = commands.add_parser(
        "evaluate", help="evaluate against the ground truth (ranking, filter, latency, ...)"
    )
    add_configuration(evaluate)
    evaluate.add_argument(
        "--modes",
        default=",".join(m.value for m in RetrievalMode),
        help="comma-separated retrieval modes (default: semantic,keyword,hybrid)",
    )
    evaluate.add_argument("--repeat", type=int, default=5, help="timed runs per query (5)")
    evaluate.add_argument(
        "--measure-indexing",
        action="store_true",
        help="also time indexing from scratch, in a temporary Chroma directory",
    )
    evaluate.add_argument("--out", help="write the full report JSON here")
    evaluate.add_argument("--json", action="store_true", help="print the report JSON")

    parse = commands.add_parser("parse", help="print a JD's parsed requirements")
    _jd_source(parse)
    return parser


# --- Shared -------------------------------------------------------------------------------


@dataclass
class Index:
    """Everything matching needs from one indexed collection."""

    name: str
    sidecar: Sidecar
    store: VectorStore
    keyword_index: KeywordIndex
    embedder: Embedder
    config: RetrievalConfig
    load_seconds: float  # model load (warm-up), kept out of the per-JD timings


def open_index(args) -> Index:
    """The collection for ``--chunker`` / ``--embedder``; ``UsageError`` if not indexed."""
    if args.k < 1:
        raise UsageError(f"-k must be at least 1, not {args.k}")
    chunker = make_chunker(args.chunker, args.chunker_opt)
    embedder = make_embedder(args.embedder, args.model)
    name, sidecar = require_index(chunker, embedder)
    store = VectorStore(chroma_dir(), chunker.chunker_id, embedder.embedder_id, embedder.dim)
    config = RetrievalConfig(mode=args.mode)
    keyword_index = KeywordIndex.from_store(store, mode=config.keyword_mode)
    start = time.perf_counter()
    embedder.embed_query("warm-up")  # load the model now so per-JD latency excludes it
    load_seconds = time.perf_counter() - start
    return Index(name, sidecar, store, keyword_index, embedder, config, load_seconds)


def read_jd(args) -> tuple[str, str]:
    """``(text, source)`` from ``--jd`` or ``--jd-text``."""
    try:
        if args.jd is not None:
            return read_job_description(args.jd)
        if not args.jd_text.strip():
            raise JobDescriptionError("the job description is empty")
        return args.jd_text, "<text>"
    except JobDescriptionError as exc:
        raise UsageError(f"cannot read the job description: {exc}") from None


def run_match(index: Index, text: str, source: str, args, as_of: date) -> tuple[dict, dict]:
    """``(output in --format, full output for display)`` for one JD."""
    job = parse_job_description(text)
    result = match(
        job,
        index.store,
        index.keyword_index,
        index.embedder,
        index.sidecar,
        index.config,
        args.filter,
        args.k,
        as_of=as_of,
    )
    full = build_output(result, job, OutputFormat.FULL, source=source, collection=index.name)
    if args.format == OutputFormat.FULL.value:
        return full, full
    return build_output(result, job, args.format, source=source, collection=index.name), full


def _dumps(data: Any, ascii_only: bool = False) -> str:
    return json.dumps(data, indent=2, ensure_ascii=ascii_only) + "\n"


def _stdout_is_utf8(console) -> bool:
    encoding = (getattr(console.file, "encoding", None) or "utf-8").lower().replace("-", "")
    return encoding == "utf8"


def _warn(warnings: Sequence[str], seen: set[str] | None = None) -> None:
    for warning in warnings:
        if seen is None or warning not in seen:
            print(f"{PROG}: warning: {warning}", file=sys.stderr)
            if seen is not None:
                seen.add(warning)


# --- Commands -----------------------------------------------------------------------------


def cmd_match(args, console, today: Callable[[], date]) -> int:
    """Match one JD: a rich table, ``--json`` output and/or an ``--out`` file."""
    as_of = resolve_as_of(args.as_of, today)
    text, source = read_jd(args)
    index = open_index(args)
    output, full = run_match(index, text, source, args, as_of)
    _warn(full["warnings"])
    if args.out:
        write_json_atomic(Path(args.out), output)
    if args.json:
        console.file.write(_dumps(output, ascii_only=not _stdout_is_utf8(console)))
    else:
        render_match(full, console)
        if args.out:
            console.print(f"Wrote {args.out}")
    return 0


def cmd_batch(args, console, today: Callable[[], date]) -> int:
    """Match every JD in ``--jd-dir``; write ``<out-dir>/<jd>.json`` and print a summary."""
    as_of = resolve_as_of(args.as_of, today)
    jobs = job_descriptions_in(args.jd_dir)
    index = open_index(args)
    out_dir = Path(args.out_dir)
    rows, seen = [], set()
    for origin, text in jobs.items():
        output, full = run_match(index, text, origin, args, as_of)
        _warn(full["warnings"], seen)
        path = out_dir / f"{PurePosixPath(origin).stem}.json"
        write_json_atomic(path, output)
        top = full["top_matches"][0] if full["top_matches"] else None
        rows.append(
            {
                "jd": PurePosixPath(origin).stem,
                "eligible": full["filter"]["eligible_count"],
                "returned": len(full["top_matches"]),
                "top_candidate": top["candidate_name"] if top else None,
                "top_score": top["match_score"] if top else None,
                "latency_ms": full["timings_ms"]["total"],
                "output": path.as_posix(),
            }
        )
    render_batch_summary(rows, console)
    console.print(f"Model load (not in the latencies): {index.load_seconds:.1f} s")
    return 0


def _modes(text: str) -> list[str]:
    allowed = [m.value for m in RetrievalMode]
    modes = [m.strip() for m in text.split(",") if m.strip()]
    unknown = [m for m in modes if m not in allowed]
    if not modes or unknown:
        raise UsageError(f"--modes takes a comma-separated subset of {allowed}, got {text!r}")
    return list(dict.fromkeys(modes))


def cmd_evaluate(args, console) -> int:
    """Run every evaluation layer and the latency measurements against the ground truth."""
    modes = _modes(args.modes)
    if args.repeat < 1:
        raise UsageError(f"--repeat must be at least 1, not {args.repeat}")
    try:
        qrels, manifest = load_qrels(), load_manifest()
    except GroundTruthError as exc:
        raise UsageError(f"ground truth: {exc}") from None
    chunker = make_chunker(args.chunker, args.chunker_opt)
    embedder = make_embedder(args.embedder, args.model)
    name, sidecar = require_index(chunker, embedder)
    texts = {}
    for jd_id, truth in sorted(qrels.jds.items()):
        try:
            texts[jd_id], _ = read_job_description(truth.jd_path)
        except JobDescriptionError as exc:
            raise UsageError(f"{truth.jd_path}: {exc}") from None
    store = VectorStore(chroma_dir(), chunker.chunker_id, embedder.embedder_id, embedder.dim)
    config = RetrievalConfig()
    keyword_index = KeywordIndex.from_store(store, mode=config.keyword_mode)
    calibration, warnings = load_calibration(sidecar, embedder)
    _warn(warnings)
    index_latency = None
    if args.measure_indexing:
        index_latency = ("resumes", chunker, date.fromisoformat(sidecar.as_of))
    report = run_evaluation(
        texts,
        qrels,
        manifest,
        store,
        keyword_index,
        embedder,
        calibration,
        collection=name,
        config=config,
        modes=modes,
        repeat=args.repeat,
        index_latency=index_latency,
        warnings=warnings,
    )
    if args.out:
        write_json_atomic(Path(args.out), report)
    if args.json:
        console.file.write(_dumps(report, ascii_only=not _stdout_is_utf8(console)))
    else:
        render_evaluation(report, console)
        if args.out:
            console.print(f"Wrote {args.out}")
    return 0


def cmd_parse(args, console) -> int:
    """Print the parsed requirements as JSON."""
    text, source = read_jd(args)
    data = {"source": source} | parse_job_description(text).to_dict()
    console.file.write(_dumps(data, ascii_only=not _stdout_is_utf8(console)))
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
        if args.command == "match":
            return cmd_match(args, console, today)
        if args.command == "evaluate":
            return cmd_evaluate(args, console)
        if args.command == "batch":
            return cmd_batch(args, console, today)
        return cmd_parse(args, console)
    except (UsageError, StoreError, FsToolError) as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
