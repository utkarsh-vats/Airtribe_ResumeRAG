"""Synthetic candidates, hits and JDs for the scoring, reasoning and matcher tests."""

from rag.jd_parser import parse_job_description
from rag.metadata import CandidateMetadata
from rag.retrieval import CandidateHit, ListScore, TopChunk
from rag.timeline import RoleRecord, months_to_years

PY_JD = parse_job_description(
    "Senior Python Backend Engineer\nAcme - Remote\n\n"
    "Requirements\n- 5+ years of Python\n- Django or FastAPI\n- PostgreSQL\n\n"
    "Nice to have\n- Docker\n- Redis\n- Celery\n- AWS\n"
)


def meta(
    candidate_id: str = "noah",
    skills: dict[str, int] | None = None,
    total_months: int = 85,
    roles: tuple[tuple[str, str], ...] = (("Senior Backend Engineer", "Harbourline Freight"),),
) -> CandidateMetadata:
    skills = skills or {}
    return CandidateMetadata(
        candidate_id=candidate_id,
        resume_path=f"resumes/{candidate_id}.txt",
        name=candidate_id.title(),
        roles=tuple(RoleRecord(t, c, "2020 - Present", None, False) for t, c in roles),
        total_months=total_months,
        total_years=months_to_years(total_months),
        education=(),
        highest_education=None,
        skills=tuple(sorted(skills)),
        skill_months=dict(skills),
    )


def top(section: str, role_index: int | None = None) -> TopChunk:
    return TopChunk(
        f"{section}-{role_index}", section, role_index, ("semantic",), 0.7, 0.7, None, None, "t"
    )


def hit(candidate_id: str = "noah", aggregate: float = 0.7, chunks=()) -> CandidateHit:
    return CandidateHit(
        candidate_id=candidate_id,
        fused_score=0.03,
        semantic=ListScore(aggregate, aggregate, aggregate, 1),
        keyword=ListScore(0.0, 0.0, 0.0, None, True),
        top_chunks=tuple(chunks),
        matched_sections=tuple(sorted({c.section for c in chunks})),
    )
