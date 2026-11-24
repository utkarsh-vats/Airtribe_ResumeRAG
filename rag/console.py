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


# --- job_matcher.py -----------------------------------------------------------------------


def render_match(output: dict, console: Console) -> None:
    """A full-format match output: header, ranked table, then each match's excerpts."""
    job, filt = output["job"], output["filter"]
    console.print(f"[bold]{job['title']}[/bold] ({job['source']})")
    must = "; ".join(
        " or ".join(r["any_of"]) + (f" {r['min_years']}+ yrs" if r["min_years"] is not None else "")
        for r in job["must_have"]
    )
    console.print(f"Must-haves: {must or 'none'}")
    if job["min_total_years"] is not None:
        console.print(f"Total experience: {job['min_total_years']}+ yrs")
    console.print(f"Nice-to-haves: {', '.join(job['nice_to_have']) or 'none'}")
    console.print(
        f"Filter: {filt['mode']} - {filt['eligible_count']} of {filt['candidates_total']} "
        f"candidates meet every must-have; {filt['filtered_count']} filtered, "
        f"{filt['scored_count']} scored"
    )
    for item in job["not_assessed"]:
        console.print(f"Not assessed: {item}")
    for note in output["notes"]:
        console.print(f"Note: {note}")
    table = Table(title=f"Top {len(output['top_matches'])} matches")
    table.add_column("#", justify="right")
    table.add_column("name")
    table.add_column("score", justify="right")
    table.add_column("matched skills")
    table.add_column("reasoning")
    for rank, match in enumerate(output["top_matches"], start=1):
        table.add_row(
            str(rank),
            match["candidate_name"],
            str(match["match_score"]),
            ", ".join(match["matched_skills"]),
            match["reasoning"],
        )
    console.print(table)
    for rank, match in enumerate(output["top_matches"], start=1):
        console.print(f"[bold]{rank}. {match['candidate_name']}[/bold] ({match['resume_path']})")
        for excerpt in match["relevant_excerpts"]:
            console.print(f"   - {excerpt}")
    console.print(f"Total time: {output['timings_ms']['total']:.0f} ms")


def render_batch_summary(rows: list[dict], console: Console) -> None:
    """One row per JD: eligible, returned, top candidate, top score, latency."""
    table = Table(title="Batch summary")
    for column, justify in (
        ("JD", "left"),
        ("eligible", "right"),
        ("returned", "right"),
        ("top candidate", "left"),
        ("top score", "right"),
        ("latency ms", "right"),
        ("output", "left"),
    ):
        table.add_column(column, justify=justify)
    for row in rows:
        table.add_row(
            row["jd"],
            str(row["eligible"]),
            str(row["returned"]),
            row["top_candidate"] or "-",
            "-" if row["top_score"] is None else str(row["top_score"]),
            f"{row['latency_ms']:.0f}",
            row["output"],
        )
    console.print(table)


# --- job_matcher.py evaluate --------------------------------------------------------------

RANKING_COLUMNS = (
    "P@5_strict",
    "P@5_lenient",
    "R@10_strict",
    "R@10_lenient",
    "MRR_strict",
    "MRR_lenient",
    "nDCG@10",
    "intrusion@10",
)


def _num(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_evaluation(report: dict, console: Console) -> None:
    """Ranking, filter, extraction, end-to-end and latency tables for an evaluation report."""
    config, env = report["config"], report["environment"]
    console.print(
        f"[bold]Evaluation[/bold]: {config['collection']} "
        f"({env['model']}, {env['torch_device'] or 'n/a'}; git {env['git_commit'] or 'n/a'})"
    )
    cal = report["calibration"]
    console.print(
        f"Calibration: {cal['source']} floor {cal['floor']:.4f} ceiling {cal['ceiling']:.4f}"
    )

    ranking = Table(title="Ranking before the filter (macro over JDs)")
    ranking.add_column("mode")
    ranking.add_column("order")
    for column in RANKING_COLUMNS:
        ranking.add_column(column, justify="right")
    for key, values in report["ranking"]["macro"].items():
        mode, order = key.split("/")
        ranking.add_row(mode, order, *(_num(values[c]) for c in RANKING_COLUMNS))
    console.print(ranking)
    for note in report["notes"]:
        console.print(f"Note: {note}")

    f = report["filter"]["overall"]
    console.print(
        f"[bold]Filter[/bold]: {f['agreements']}/{f['pairs']} pairs agree "
        f"(TP {f['tp']}, FP {f['fp']}, FN {f['fn']}, TN {f['tn']}); precision "
        f"{_num(f['precision'])}, recall {_num(f['recall'])}, accuracy {_num(f['accuracy'])}"
    )
    for d in report["filter"]["disagreements"]:
        console.print(
            f"  {d['jd_id']} / {d['candidate_id']}: system "
            f"{'eligible' if d['system_eligible'] else 'not eligible'} "
            f"({'; '.join(d['system_unmet']) or 'nothing unmet'}), qrels "
            f"{'eligible' if d['qrels_eligible'] else 'not eligible'} "
            f"({'; '.join(d['qrels_unmet']) or 'nothing unmet'})"
        )
    e = report["extraction"]
    console.print(
        f"[bold]Extraction[/bold] ({e['candidates']} candidates): name {_num(e['name_exact'])}, "
        f"skills F1 micro {_num(e['skills_micro']['f1'])} / macro "
        f"{_num(e['skills_macro']['f1'])}, total years exact {_num(e['total_years_exact'])} "
        f"(MAE {_num(e['total_years_mae'])}), skill years MAE {_num(e['skill_years_mae'])} over "
        f"{e['skill_years_pairs']} pairs, education {_num(e['education_exact'])}"
    )
    for cid, notes in e["mismatches"].items():
        console.print(f"  {cid}: {'; '.join(notes)}")

    end = Table(title=f"End to end (strict, top {report['end_to_end']['k']})")
    for column in (
        "JD",
        "eligible",
        "returned",
        "S returned",
        "S total",
        "precision S",
        "recall S",
    ):
        end.add_column(column, justify="left" if column == "JD" else "right")
    for row in report["end_to_end"]["rows"]:
        end.add_row(
            row["jd_id"],
            str(row["eligible"]),
            str(row["returned"]),
            str(row["returned_s"]),
            str(row["s_total"]),
            _num(row["precision_s"]),
            _num(row["recall_s"]),
        )
    console.print(end)

    query = report["latency"]["query"]
    by_mode: dict[str, list[str]] = {}
    for key in query["macro"]:  # "mode/filter"
        by_mode.setdefault(key.split("/")[0], []).append(key)
    for mode, keys in by_mode.items():
        latency = Table(
            title=f"Query latency, {mode}, ms (macro over JDs; {query['repeat']} runs after "
            f"{query['warmup']} warm-up)"
        )
        latency.add_column("stage")
        for key in keys:
            latency.add_column(f"{key.split('/')[1]} p50", justify="right")
            latency.add_column(f"{key.split('/')[1]} p95", justify="right")
        for stage in query["macro"][keys[0]]:
            latency.add_row(
                stage,
                *(
                    _num(query["macro"][key][stage][stat], 1)
                    for key in keys
                    for stat in ("p50", "p95")
                ),
            )
        console.print(latency)
    console.print(f"Model load (not in any query): {query['model_load_ms'] / 1000:.1f} s")

    index = report["latency"]["index"]
    if index:
        table = Table(
            title=f"Index latency: {index['files']} files, {index['chunks']} chunks, from scratch"
        )
        for column in ("stage", "total ms", "per-file p50", "per-file p95"):
            table.add_column(column, justify="left" if column == "stage" else "right")
        for stage, total in index["totals_ms"].items():
            per_file = index["per_file"].get(stage, {})
            table.add_row(
                stage, _num(total, 1), _num(per_file.get("p50"), 1), _num(per_file.get("p95"), 1)
            )
        console.print(table)
        console.print(
            f"First run {index['first_run_ms'] / 1000:.2f} s wall; unchanged re-run "
            f"{index['rerun_ms'] / 1000:.2f} s ({index['rerun_unchanged']} unchanged); embedding "
            f"{_num(index['embed_chunks_per_second'], 1)} chunks/s"
        )
