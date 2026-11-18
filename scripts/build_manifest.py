"""Build ``data/ground_truth/manifest.json`` (the extraction answer key) from the personas.

Run from the repo root::

    python -m scripts.build_manifest           # rewrite the manifest
    python -m scripts.build_manifest --check   # exit 1 if the committed manifest is stale

The maths below is the ground-truth definition (see ``data/ground_truth/README.md``). It is
deliberately independent of the runtime extractor in ``rag/`` so that the two can be compared.
"""

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from datetime import date

from rag.skills import expand_implied
from scripts.persona_schema import (
    AS_OF,
    GROUND_TRUTH_DIR,
    PRESENT,
    EmploymentType,
    Persona,
    Role,
    load_personas,
)

MANIFEST_PATH = GROUND_TRUTH_DIR / "manifest.json"


def month_index(value: str, as_of: date = AS_OF) -> int:
    """Months since year 0 for ``"YYYY-MM"``; ``"present"`` is the ``as_of`` month."""
    if value == PRESENT:
        return as_of.year * 12 + as_of.month - 1
    year, month = value.split("-")
    return int(year) * 12 + int(month) - 1


def role_interval(role: Role, as_of: date = AS_OF) -> tuple[int, int]:
    """Inclusive ``(start, end)`` month indexes of a role."""
    return month_index(role.start, as_of), month_index(role.end, as_of)


def merged_months(intervals: Iterable[tuple[int, int]]) -> int:
    """Number of distinct months covered by inclusive intervals (overlaps counted once)."""
    total = 0
    current: tuple[int, int] | None = None
    for start, end in sorted(intervals):
        if current is None or start > current[1] + 1:
            if current is not None:
                total += current[1] - current[0] + 1
            current = (start, end)
        else:
            current = (current[0], max(current[1], end))
    if current is not None:
        total += current[1] - current[0] + 1
    return total


def months_to_years(months: int) -> float:
    """Years rounded to one decimal (k/12 never lands on a .x5 tie, so rounding is exact)."""
    return round(months / 12, 1)


def counted_roles(persona: Persona) -> list[Role]:
    """Roles that count towards experience years: everything except internships."""
    return [r for r in persona.experience if r.employment_type != EmploymentType.INTERNSHIP]


def total_years(persona: Persona, as_of: date = AS_OF) -> float:
    """Merged months of non-internship roles, in years."""
    return months_to_years(merged_months(role_interval(r, as_of) for r in counted_roles(persona)))


def candidate_skills(persona: Persona) -> frozenset[str]:
    """Skills section + role skills + project skills, with implications expanded.

    Hobbies and coursework are never skills.
    """
    names = [*persona.skills_section]
    for role in persona.experience:
        names.extend(role.skills_used)
    for project in persona.projects:
        names.extend(project.skills_used)
    return expand_implied(names)


def skill_years(persona: Persona, as_of: date = AS_OF) -> dict[str, float]:
    """Per candidate skill: merged months of non-internship roles that use it (implied too)."""
    roles = [
        (role_interval(r, as_of), expand_implied(r.skills_used)) for r in counted_roles(persona)
    ]
    return {
        skill: months_to_years(merged_months(span for span, used in roles if skill in used))
        for skill in sorted(candidate_skills(persona))
    }


def highest_education(persona: Persona) -> dict | None:
    """Highest degree level; ties go to the most recent end year, then to the first listed."""
    if not persona.education:
        return None
    best = max(
        enumerate(persona.education), key=lambda item: (item[1].level, item[1].end_year, -item[0])
    )[1]
    return {
        "degree": best.degree,
        "field": best.field,
        "institution": best.institution,
        "end_year": best.end_year,
    }


def manifest_entry(persona: Persona, as_of: date = AS_OF) -> dict:
    """Ground-truth record for one candidate."""
    return {
        "candidate_id": persona.id,
        "name": persona.name,
        "resume_path": f"resumes/{persona.id}.{persona.render.format}",
        "format": str(persona.render.format),
        "skills": sorted(candidate_skills(persona)),
        "total_years": total_years(persona, as_of),
        "skill_years": skill_years(persona, as_of),
        "highest_education": highest_education(persona),
        "edge_cases": sorted(persona.edge_cases),
    }


def build_manifest(personas: Iterable[Persona], as_of: date = AS_OF) -> dict:
    """Manifest for all personas, ordered by candidate id."""
    entries = [manifest_entry(p, as_of) for p in sorted(personas, key=lambda p: p.id)]
    return {"as_of": as_of.isoformat(), "candidates": entries}


def render_manifest(manifest: dict) -> str:
    """Deterministic JSON text: sorted keys, two-space indent, trailing newline."""
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the manifest is stale")
    args = parser.parse_args(argv)

    text = render_manifest(build_manifest(load_personas()))
    if args.check:
        current = MANIFEST_PATH.read_text(encoding="utf-8") if MANIFEST_PATH.exists() else ""
        if current != text:
            print(f"{MANIFEST_PATH} is stale; run: python -m scripts.build_manifest")
            return 1
        print(f"{MANIFEST_PATH} is up to date")
        return 0
    MANIFEST_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {MANIFEST_PATH} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
