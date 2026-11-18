"""Parser for the Slots table in ``data/ground_truth/persona_plan.md`` (dataset tooling).

The plan holds the hand-assigned relevance label of every candidate for every JD theme
(strong / partial / distractor / none). ``scripts/build_qrels.py`` turns those labels into
``qrels.json``; ``tests/test_personas.py`` checks them against the manifest.
"""

from dataclasses import dataclass
from pathlib import Path

from scripts.jd_themes import load_jd_themes
from scripts.persona_schema import GROUND_TRUTH_DIR

PLAN_PATH = GROUND_TRUTH_DIR / "persona_plan.md"
RELEVANCE = ("strong", "partial", "distractor", "none")
_FIXED_COLUMNS = [
    "id",
    "archetype",
    "seniority",
    "years",
    "format",
    "layout",
    "date_format",
    "edge_cases",
]


class PlanError(ValueError):
    """The Slots table does not match the expected structure."""


@dataclass(frozen=True)
class Slot:
    id: str
    archetype: str
    seniority: str
    years: float
    format: str
    layout: str
    date_format: str
    edge_cases: frozenset[str]
    relevance: dict[str, str]  # JD theme id -> one of RELEVANCE


def load_plan(path: Path = PLAN_PATH) -> dict[str, Slot]:
    """Parse the Slots table; its JD columns must be the theme ids, in jd_themes.json order."""
    section = path.read_text(encoding="utf-8").split("## Slots", 1)[1]
    rows: list[list[str]] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        rows.append([cell.strip().strip("`") for cell in line.strip().strip("|").split("|")])
    header, _separator, *body = rows
    jds = tuple(theme.id for theme in load_jd_themes())
    if header[:8] != _FIXED_COLUMNS or tuple(header[8:]) != jds:
        raise PlanError(f"unexpected Slots header: {header}")
    slots: dict[str, Slot] = {}
    for row in body:
        if len(row) != len(header):
            raise PlanError(f"row has {len(row)} cells, expected {len(header)}: {row}")
        tags = frozenset() if row[7] == "-" else frozenset(t.strip() for t in row[7].split(","))
        relevance = dict(zip(jds, row[8:], strict=True))
        if not set(relevance.values()) <= set(RELEVANCE):
            raise PlanError(f"{row[0]}: unknown relevance label in {relevance}")
        if row[0] in slots:
            raise PlanError(f"duplicate plan id {row[0]}")
        slots[row[0]] = Slot(row[0], row[1], row[2], float(row[3]), *row[4:7], tags, relevance)
    return slots
