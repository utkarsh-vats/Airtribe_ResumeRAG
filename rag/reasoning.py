"""Templated, deterministic reasoning for a ``ScoredMatch`` (no LLM).

Every clause traces back to a field of the match: the must-have results with the evidenced
years of the skill that satisfied each, the nice-to-haves present, total experience, and the
sections (and roles) of the top retrieved chunks. For example::

    Meets all 3 must-haves: Python 7.1 yrs (5+ required), FastAPI 4.3 yrs (Django or
    FastAPI), PostgreSQL 7.1 yrs. Has 2 of 4 nice-to-haves (Docker, AWS). 7.1 years' total
    experience. Strongest evidence in Experience (Senior Backend Engineer at Harbourline
    Freight) and Skills.

In soft mode a candidate with unmet requirements also gets ``Misses: ...``.
"""

from collections.abc import Sequence

from rag.jd_parser import JobRequirements
from rag.requirements_check import RequirementResult
from rag.scoring import ScoredMatch
from rag.timeline import months_to_years

MAX_ROLES = 2


def join_words(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def describe_met(result: RequirementResult) -> str:
    """``Python 7.1 yrs (5+ required)``, ``FastAPI 4.3 yrs (Django or FastAPI)``."""
    req = result.requirement
    qualifiers = []
    if result.have_months == 0:
        qualifiers.append("no counted experience")  # e.g. only listed, or internships only
    if req is not None and len(req.any_of) > 1:
        qualifiers.append(" or ".join(req.any_of))
    if req is not None and req.min_years is not None:
        qualifiers.append(f"{req.min_years}+ required")
    text = result.skill or ""
    if result.have_months:
        text += f" {months_to_years(result.have_months)} yrs"
    return text + (f" ({', '.join(qualifiers)})" if qualifiers else "")


def _must_haves(match: ScoredMatch) -> str | None:
    results = match.check.must_have_results
    if not results:
        return None
    met = [r for r in results if r.met]
    n = len(results)
    items = ", ".join(describe_met(r) for r in met)
    if len(met) == n:
        lead = "Meets the must-have" if n == 1 else f"Meets all {n} must-haves"
        return f"{lead}: {items}."
    if not met:
        return f"Meets none of the {_plural(n, 'must-have')}."
    return f"Meets {len(met)} of {n} must-haves: {items}."


def _nice_to_haves(match: ScoredMatch, job: JobRequirements) -> str | None:
    if not job.nice_to_have:
        return None
    present = set(match.check.nice_present)
    ordered = [s for s in job.nice_to_have_in_jd_order() if s in present]
    n = len(job.nice_to_have)
    if not ordered:
        return f"Has none of the {_plural(n, 'nice-to-have')}."
    return f"Has {len(ordered)} of {_plural(n, 'nice-to-have')} ({', '.join(ordered)})."


def _experience(match: ScoredMatch, job: JobRequirements) -> str:
    years = months_to_years(match.check.total_months)
    required = f" ({job.min_total_years}+ required)" if job.min_total_years is not None else ""
    return f"{years} years' total experience{required}."


def _evidence(match: ScoredMatch) -> str | None:
    sections: list[str] = []
    roles: list[str] = []
    for chunk in match.top_chunks:
        if chunk.section not in sections:
            sections.append(chunk.section)
        labels = match.check.role_labels
        if chunk.role_index is not None and chunk.role_index < len(labels):
            label = labels[chunk.role_index]
            if label not in roles and len(roles) < MAX_ROLES:
                roles.append(label)
    if not sections:
        return None
    names = []
    for section in sections:
        name = section.replace("_", " ").title()
        if section == "experience" and roles:
            name += f" ({', '.join(roles)})"
        names.append(name)
    return f"Strongest evidence in {join_words(names)}."


def reasoning(match: ScoredMatch, job: JobRequirements, show_misses: bool = False) -> str:
    """The reasoning string; ``show_misses`` (soft mode) adds ``Misses: ...``."""
    parts = [_must_haves(match)]
    if show_misses and match.unmet:
        parts.append(f"Misses: {'; '.join(match.unmet)}.")
    parts += [_nice_to_haves(match, job), _experience(match, job), _evidence(match)]
    return " ".join(p for p in parts if p)
