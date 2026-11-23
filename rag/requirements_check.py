"""Must-have checks: does a candidate's extracted metadata meet a JD's hard requirements?

The rule is the one ``data/ground_truth/qrels.json`` uses for ``eligible``:

- a must-have is met when one skill in its ``any_of`` is among the candidate's skills (with
  implications, so Django implies Python) and, if ``min_years`` is set, that same skill has
  at least ``min_years * 12`` evidenced months;
- ``min_total_years`` is met when ``total_months >= min_total_years * 12``;
- the candidate is eligible when every requirement is met.

Each requirement also gets partial credit for scoring: 1.0 when a skill with no year minimum
is present, ``min(1, have / need)`` months when there is one, 0.0 when no listed skill is
present. ``unmet`` strings match qrels' style: ``"4+ total yrs (has 3.9)"``,
``"Python 5+ yrs (has 4.3)"``, ``"Django or FastAPI (missing)"``.
"""

from dataclasses import dataclass

from rag.jd_parser import JobRequirements, Requirement
from rag.metadata import CandidateMetadata
from rag.timeline import months_to_years


@dataclass(frozen=True)
class RequirementResult:
    """One must-have (or the total-years minimum) checked against one candidate."""

    label: str  # "Python 5+ yrs", "Django or FastAPI", "4+ total yrs"
    requirement: Requirement | None  # None for the total-years minimum
    met: bool
    skill: str | None  # the skill that satisfies it, else the best present one; None if absent
    have_months: int  # that skill's evidenced months (total months for total years)
    need_months: int | None  # min years x 12, or None without a year minimum
    credit: float  # 0-1

    @property
    def is_total(self) -> bool:
        """Whether this is the total-years minimum."""
        return self.requirement is None

    @property
    def has_years(self) -> bool:
        """Whether a year minimum applies."""
        return self.need_months is not None

    @property
    def unmet_text(self) -> str | None:
        """The qrels-style description when unmet, else None."""
        if self.met:
            return None
        if self.skill is None:
            return f"{self.label} (missing)"
        return f"{self.label} (has {months_to_years(self.have_months)})"


@dataclass(frozen=True)
class RequirementCheck:
    """A candidate against a JD's hard requirements."""

    candidate_id: str
    name: str
    resume_path: str
    total_months: int
    results: tuple[RequirementResult, ...]  # the total-years minimum first, then must-haves
    nice_present: tuple[str, ...]  # in ``job.nice_to_have`` order
    nice_missing: tuple[str, ...]
    skills: tuple[str, ...]  # the candidate's skills, implications included
    skill_months: dict[str, int]
    role_labels: tuple[str, ...]  # "Title at Company" by role index, for reasoning

    @property
    def eligible(self) -> bool:
        """All requirements met."""
        return all(r.met for r in self.results)

    @property
    def unmet(self) -> tuple[str, ...]:
        """Human-readable unmet requirements, in qrels order."""
        return tuple(r.unmet_text for r in self.results if not r.met)

    @property
    def must_have_results(self) -> tuple[RequirementResult, ...]:
        """The must-have skills, without the total-years minimum."""
        return tuple(r for r in self.results if not r.is_total)

    @property
    def total_result(self) -> RequirementResult | None:
        """The total-years minimum, if the JD has one."""
        return next((r for r in self.results if r.is_total), None)


def _skill_requirement(req: Requirement, meta: CandidateMetadata) -> RequirementResult:
    skills = " or ".join(req.any_of)
    present = [s for s in req.any_of if s in meta.skills]
    months = {s: meta.skill_months.get(s, 0) for s in present}
    if req.min_years is None:
        label, need = skills, None
        met = bool(present)
        skill = present[0] if present else None
        credit = 1.0 if present else 0.0
    else:
        label, need = f"{skills} {req.min_years}+ yrs", req.min_years * 12
        meeting = [s for s in present if months[s] >= need]
        met = bool(meeting)
        # The first skill that meets it, else the present skill with the most months.
        skill = meeting[0] if meeting else max(present, key=lambda s: months[s], default=None)
        credit = 0.0 if skill is None else (1.0 if need == 0 else min(1.0, months[skill] / need))
    return RequirementResult(
        label=label,
        requirement=req,
        met=met,
        skill=skill,
        have_months=months.get(skill, 0) if skill else 0,
        need_months=need,
        credit=credit,
    )


def _total_requirement(years: int, meta: CandidateMetadata) -> RequirementResult:
    need = years * 12
    return RequirementResult(
        label=f"{years}+ total yrs",
        requirement=None,
        met=meta.total_months >= need,
        skill="total",
        have_months=meta.total_months,
        need_months=need,
        credit=1.0 if need == 0 else min(1.0, meta.total_months / need),
    )


def check_candidate(meta: CandidateMetadata, job: JobRequirements) -> RequirementCheck:
    """Check ``meta`` against ``job``'s must-haves, total-years minimum and nice-to-haves."""
    results = []
    if job.min_total_years is not None:
        results.append(_total_requirement(job.min_total_years, meta))
    results += [_skill_requirement(req, meta) for req in job.must_have]
    return RequirementCheck(
        candidate_id=meta.candidate_id,
        name=meta.name,
        resume_path=meta.resume_path,
        total_months=meta.total_months,
        results=tuple(results),
        nice_present=tuple(s for s in job.nice_to_have if s in meta.skills),
        nice_missing=tuple(s for s in job.nice_to_have if s not in meta.skills),
        skills=tuple(meta.skills),
        skill_months=dict(meta.skill_months),
        role_labels=tuple(f"{r.title} at {r.company}" for r in meta.roles),
    )
