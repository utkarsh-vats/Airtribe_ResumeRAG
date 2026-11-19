"""Terminal rendering with rich, shared by the CLIs (``resume_rag.py``, ``job_matcher.py``).

Only presentation lives here: the CLIs gather the data and call these functions. When output
isn't a terminal (pipes, tests) the console uses a fixed wide width so tables don't wrap or
truncate, and rich falls back to ASCII borders on non-UTF-8 encodings by itself.
"""

import sys
from collections.abc import Iterable
from typing import IO

from rich.console import Console
from rich.table import Table

from rag.indexer import IndexReport
from rag.metadata import CandidateMetadata
from rag.store import CollectionStats
from rag.timeline import months_to_years

PIPED_WIDTH = 200


def make_console(file: IO[str] | None = None) -> Console:
    """A console on ``file`` (default: the current ``sys.stdout``)."""
    file = file or sys.stdout
    isatty = getattr(file, "isatty", lambda: False)()
    return Console(file=file, width=None if isatty else PIPED_WIDTH, highlight=False)


def _degree(metadata: dict) -> str:
    highest = metadata.get("highest_education")
    return f"{highest['degree']} {highest['field']}".strip() if highest else "-"


def render_index_report(report: IndexReport, console: Console) -> None:
    """Collection, counts, failures, refreshed metadata and per-stage timings."""
    console.print(f"[bold]Collection:[/bold] {report.collection}")
    counts = Table(title="Files", show_header=True)
    counts.add_column("added", justify="right")
    counts.add_column("updated", justify="right")
    counts.add_column("unchanged", justify="right")
    counts.add_column("removed", justify="right")
    counts.add_column("skipped", justify="right")
    counts.add_column("failed", justify="right")
    counts.add_column("chunks written", justify="right")
    counts.add_row(
        *(
            str(len(x))
            for x in (
                report.added,
                report.updated,
                report.unchanged,
                report.removed,
                report.skipped,
                report.failed,
            )
        ),
        str(report.chunks_written),
    )
    console.print(counts)
    if report.failed or report.skipped:
        issues = Table(title="Failed and skipped files")
        issues.add_column("path")
        issues.add_column("status")
        issues.add_column("code")
        issues.add_column("message")
        for issue in report.failed:
            issues.add_row(issue.path, "failed", issue.code, issue.message)
        for issue in report.skipped:
            issues.add_row(issue.path, "skipped", issue.code, issue.message)
        console.print(issues)
    if report.removed:
        console.print("Removed: " + ", ".join(report.removed))
    if report.metadata_refreshed:
        console.print(
            f"Metadata refreshed (unchanged text, new as_of): {len(report.metadata_refreshed)} "
            f"file(s): " + ", ".join(report.metadata_refreshed)
        )
    timings = Table(title="Timings (seconds)")
    for stage in ("read", "parse_extract", "chunk", "embed", "upsert", "total"):
        timings.add_column(stage, justify="right")
    t = report.timings
    timings.add_row(
        *(f"{v:.3f}" for v in (t.read, t.parse_extract, t.chunk, t.embed, t.upsert, t.total))
    )
    console.print(timings)


def render_stats(stats: Iterable[CollectionStats], console: Console) -> None:
    """One row per collection, plus chunks by section."""
    stats = list(stats)
    if not stats:
        console.print("No collections yet. Run: python resume_rag.py index")
        return
    table = Table(title="Collections")
    for column in (
        "collection",
        "chunker",
        "embedder",
        "dim",
        "chunks",
        "candidates",
        "chunks by section",
        "as_of",
        "pipeline",
    ):
        table.add_column(column)
    for s in stats:
        table.add_row(
            s.name,
            s.chunker_id or "?",
            s.embedder_id or "?",
            str(s.dim),
            str(s.chunks),
            str(s.candidates),
            ", ".join(f"{k} {v}" for k, v in s.chunks_by_section.items()),
            s.as_of or "-",
            s.pipeline_version or "-",
        )
    console.print(table)


def render_candidates(collection: str, records: dict[str, dict], console: Console) -> None:
    """id, name, total years, highest degree and skill count, from sidecar metadata dicts."""
    table = Table(title=f"Candidates in {collection} ({len(records)})")
    table.add_column("id")
    table.add_column("name")
    table.add_column("total years", justify="right")
    table.add_column("highest degree")
    table.add_column("skills", justify="right")
    for candidate_id, meta in sorted(records.items()):
        table.add_row(
            candidate_id,
            meta["name"],
            f"{meta['total_years']:.1f}",
            _degree(meta),
            str(len(meta["skills"])),
        )
    console.print(table)


def render_candidate(metadata: CandidateMetadata, console: Console) -> None:
    """Everything the pipeline knows about one candidate."""
    console.print(f"[bold]{metadata.name}[/bold] ({metadata.candidate_id}, {metadata.resume_path})")
    console.print(
        f"Total experience: {metadata.total_years:.1f} years ({metadata.total_months} months)"
    )
    if metadata.education:
        education = Table(title="Education")
        for column in ("degree", "level", "field", "institution", "end year"):
            education.add_column(column)
        for e in metadata.education:
            marker = " (highest)" if e == metadata.highest_education else ""
            education.add_row(
                e.degree + marker,
                e.level.name.title() if e.level else "-",
                e.field,
                e.institution,
                str(e.end_year or "-"),
            )
        console.print(education)
    roles = Table(title="Roles")
    for column in ("#", "title", "company", "dates", "months", "internship"):
        roles.add_column(column)
    for index, role in enumerate(metadata.roles):
        dates = f"{role.dates.start} to {role.dates.end}" if role.dates else role.date_text
        if role.dates and role.dates.ongoing:
            dates += " (ongoing)"
        roles.add_row(
            str(index),
            role.title,
            role.company,
            dates,
            str(role.months),
            "yes" if role.is_internship else "",
        )
    console.print(roles)
    skills = Table(title=f"Skills ({len(metadata.skills)})")
    skills.add_column("skill")
    skills.add_column("years", justify="right")
    skills.add_column("months", justify="right")
    skills.add_column("evidence")
    for skill in metadata.skills:
        months = metadata.skill_months.get(skill, 0)
        skills.add_row(
            skill,
            f"{months_to_years(months):.1f}",
            str(months),
            ", ".join(metadata.skill_evidence.get(skill, ())),
        )
    console.print(skills)
    if metadata.warnings:
        console.print("[bold]Warnings:[/bold]")
        for warning in metadata.warnings:
            console.print(f"  - {warning}")
    else:
        console.print("Warnings: none")
