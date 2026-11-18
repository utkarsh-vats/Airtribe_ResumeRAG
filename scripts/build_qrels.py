"""Build ``data/ground_truth/qrels.json`` (``python -m scripts.build_qrels [--check]``).

One judgement per (JD theme, candidate) pair:

- ``label`` and ``grade`` come from the hand-assigned relevance in ``persona_plan.md``
  (strong -> S / 2, partial -> P / 1, distractor -> D / 0, none -> none / 0);
- ``eligible`` and ``unmet`` come from ``manifest.json`` + ``jd_themes.json`` with the same
  must-have rule as the label-consistency tests (``scripts.jd_themes``).

Nothing here reads ``rag/``: the system is never graded against its own output.
"""

import argparse
import json
import sys
from pathlib import Path

from scripts.build_manifest import MANIFEST_PATH
from scripts.jd_themes import load_jd_themes, unmet_details, unmet_must_haves
from scripts.persona_plan import load_plan
from scripts.persona_schema import GROUND_TRUTH_DIR

QRELS_PATH = GROUND_TRUTH_DIR / "qrels.json"
LABELS = {"strong": "S", "partial": "P", "distractor": "D", "none": "none"}
GRADES = {"S": 2, "P": 1, "D": 0, "none": 0}


def build_qrels(manifest: dict) -> dict:
    plan = load_plan()
    candidates = sorted(manifest["candidates"], key=lambda c: c["candidate_id"])
    if {c["candidate_id"] for c in candidates} != set(plan):
        raise ValueError("manifest.json and persona_plan.md list different candidates")
    jds = []
    for theme in load_jd_themes():
        judgements = []
        for candidate in candidates:
            cid = candidate["candidate_id"]
            label = LABELS[plan[cid].relevance[theme.id]]
            judgements.append(
                {
                    "candidate_id": cid,
                    "resume_path": candidate["resume_path"],
                    "label": label,
                    "grade": GRADES[label],
                    "eligible": not unmet_must_haves(theme, candidate),
                    "unmet": unmet_details(theme, candidate),
                }
            )
        jds.append(
            {
                "jd_id": theme.id,
                "title": theme.title,
                "jd_path": f"job_descriptions/{theme.id}.txt",
                "judgements": judgements,
            }
        )
    return {"as_of": manifest["as_of"], "grades": GRADES, "jds": jds}


def render_qrels(qrels: dict) -> str:
    return json.dumps(qrels, indent=2, ensure_ascii=False) + "\n"


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if qrels.json is stale")
    args = parser.parse_args(argv)
    text = render_qrels(build_qrels(load_manifest()))
    if args.check:
        current = QRELS_PATH.read_text(encoding="utf-8") if QRELS_PATH.exists() else None
        if current != text:
            print(f"{QRELS_PATH} is stale; run: python -m scripts.build_qrels", file=sys.stderr)
            return 1
        print(f"{QRELS_PATH} is up to date")
        return 0
    QRELS_PATH.write_bytes(text.encode("utf-8"))
    print(f"Wrote {QRELS_PATH} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
