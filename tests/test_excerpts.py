"""rag.excerpts: sentence choice, header and section rules, fallback, dedupe, truncation."""

import pytest
from match_helpers import PY_JD, hit, meta

from rag.calibration import Calibration
from rag.excerpts import ELLIPSIS, body_sentences, select_excerpts, truncate
from rag.requirements_check import check_candidate
from rag.retrieval import TopChunk
from rag.scoring import score_candidate

CAL = Calibration(0.6, 0.8, "collection")


def chunk(text, section="experience", body_start=0, semantic=0.5, chunk_id=None):
    return TopChunk(
        chunk_id or f"{section}-{hash(text) % 1000}",
        section,
        0 if section == "experience" else None,
        ("semantic",),
        semantic,
        semantic,
        None,
        None,
        text,
        body_start,
    )


def excerpts(*chunks, **kwargs):
    check = check_candidate(meta(skills={"Python": 85}), PY_JD)
    match = score_candidate(hit(chunks=chunks), check, PY_JD, CAL)
    return select_excerpts(match, PY_JD, **kwargs)


def test_sentence_splitting_edge_cases():
    text = "Used e.g. Python daily. Cut latency by 3.5x\nacross 12 services. Shipped i.e. code!"
    assert body_sentences(chunk(text)) == [
        "Used e.g. Python daily.",
        "Cut latency by 3.5x across 12 services.",
        "Shipped i.e. code!",
    ]


def test_skill_weighted_choice_and_tie_order():
    text = "Built Docker images. Ran Redis caches. Wrote Python and Django services. Hiked."
    assert excerpts(chunk(text)) == [
        "Wrote Python and Django services.",  # two must-haves: 4
        "Built Docker images.",  # nice-to-have: 1, earlier sentence
        "Ran Redis caches.",
    ]


def test_must_haves_outrank_nice_to_haves_across_chunks():
    first = chunk("Ran Docker and Redis and Celery.", chunk_id="a")  # 3 nice = 3
    second = chunk("Tuned PostgreSQL and Python.", "projects", chunk_id="b")  # 2 must = 4
    assert excerpts(first, second, max_excerpts=2) == [
        "Tuned PostgreSQL and Python.",
        "Ran Docker and Redis and Celery.",
    ]


def test_role_header_lines_are_never_used():
    text = "Senior Python Developer\nPython Works | 2020 - Present\nShipped dashboards."
    assert excerpts(chunk(text, body_start=2)) == ["Shipped dashboards."]


def test_hobbies_and_coursework_are_never_used():
    hobby = chunk("Python scripting for a weather station.", "hobbies", semantic=0.9)
    course = chunk("Python and Django fundamentals.", "coursework", semantic=0.9)
    work = chunk("Shipped dashboards for ops.", semantic=0.4)
    assert excerpts(hobby, course, work) == ["Shipped dashboards for ops."]
    assert excerpts(hobby, course) == []


def test_summary_only_when_preferred_sections_have_nothing():
    summary = chunk("Backend engineer working in Python.", "summary")
    plain = chunk("Shipped dashboards for ops.")
    assert excerpts(summary, plain) == ["Backend engineer working in Python."]
    work = chunk("Wrote Python services.")
    assert excerpts(summary, work) == ["Wrote Python services."]


def test_fallback_to_the_best_semantic_chunk():
    low = chunk("Led a small team. Planned sprints.", semantic=0.3, chunk_id="a")
    high = chunk("Owned the billing service. Wrote docs.", "projects", semantic=0.8, chunk_id="b")
    assert excerpts(low, high) == ["Owned the billing service."]


def test_dedupe():
    one = chunk("Wrote Python services.", chunk_id="a")
    two = chunk("Wrote  Python services.", "projects", chunk_id="b")
    assert excerpts(one, two) == ["Wrote Python services."]


def test_truncation_at_a_word_boundary():
    long = "Wrote Python services " + " ".join(f"word{i}" for i in range(60)) + "."
    (out,) = excerpts(chunk(long))
    assert len(out) <= 220 and out.endswith(ELLIPSIS)
    assert out[:-1] == long[: len(out) - 1] and long[len(out) - 1] == " "
    assert truncate("short text", 50) == "short text"
    assert truncate("alpha beta gamma", 12) == "alpha beta" + ELLIPSIS
    assert truncate("supercalifragilistic", 8) == "superca" + ELLIPSIS


def test_limits_and_determinism():
    text = " ".join(f"Wrote Python service {i}." for i in range(6))
    assert len(excerpts(chunk(text))) == 3
    assert excerpts(chunk(text), max_excerpts=1) == ["Wrote Python service 0."]
    assert excerpts(chunk(text)) == excerpts(chunk(text))
    with pytest.raises(ValueError):
        excerpts(chunk(text), max_excerpts=0)


# --- Line joining and skills (step 12c) ---------------------------------------------------


def test_title_lines_start_a_new_unit_and_are_never_excerpts():
    text = (
        "django-audit-lite\n"
        "Maintainer of an open-source Python package for Django model change history.\n"
        "Async Python in Production\n"
        "Talk given at the Austin Python Meetup on running asyncio services."
    )
    assert body_sentences(chunk(text, "projects")) == [
        "Maintainer of an open-source Python package for Django model change history.",
        "Talk given at the Austin Python Meetup on running asyncio services.",
    ]


def test_wrapped_lines_still_join_even_before_a_capital():
    text = "Built HIPAA-compliant REST APIs for clinics with\nDjango REST Framework.\nShipped it."
    assert body_sentences(chunk(text)) == [
        "Built HIPAA-compliant REST APIs for clinics with Django REST Framework.",
        "Shipped it.",
    ]
    # a short unpunctuated tail of a longer unit ends that unit, and is kept
    tail = "Cut report time from nine minutes to forty-five\nseconds overall\nLed the team."
    assert body_sentences(chunk(tail)) == [
        "Cut report time from nine minutes to forty-five seconds overall",
        "Led the team.",
    ]


def test_a_last_short_line_is_kept_and_five_words_is_a_title():
    assert body_sentences(chunk("Wrote Python services.\nOn call weekly")) == [
        "Wrote Python services.",
        "On call weekly",
    ]
    # exactly five words, unpunctuated, before a capital: a title line by the rule
    assert body_sentences(chunk("REST APIs built with care\nDjango REST Framework.")) == [
        "Django REST Framework."
    ]


def test_skills_chunks_never_supply_excerpts():
    skills = chunk("Python, Django, FastAPI, PostgreSQL, Docker", "skills", semantic=0.9)
    work = chunk("Shipped dashboards for ops.", semantic=0.2)
    assert excerpts(skills, work) == ["Shipped dashboards for ops."]
    assert excerpts(skills) == []


# --- Corpus: every excerpt is verbatim persona text ---------------------------------------


def _collapse(text: str) -> str:
    return " ".join(text.split())


def test_corpus_excerpts_are_verbatim_bullets_projects_or_summaries(fake_corpus):
    from datetime import date

    from rag.jd_parser import parse_job_description
    from rag.matcher import match
    from scripts.persona_schema import ROOT, load_personas

    personas = {p.id: p for p in load_personas()}
    found = fake_corpus.store.get(include=["documents", "metadatas"])
    skills_text = {
        meta["candidate_id"]: _collapse(doc)
        for doc, meta in zip(found["documents"], found["metadatas"], strict=True)
        if meta["section"] == "skills"
    }
    checked = 0
    for path in sorted((ROOT / "data" / "job_descriptions").glob("*.txt")):
        job = parse_job_description(path.read_text("utf-8"))
        result = match(
            job,
            fake_corpus.store,
            fake_corpus.keyword_index,
            fake_corpus.embedder,
            fake_corpus.sidecar,
            filter_mode="soft",
            k=10,
            as_of=date(2026, 11, 1),
        )
        for m in result.matches:
            persona = personas[m.candidate_id]
            sources = [_collapse(b) for role in persona.experience for b in role.bullets]
            sources += [_collapse(p.description) for p in persona.projects]
            sources.append(_collapse(persona.summary))
            for excerpt in m.relevant_excerpts:
                text = _collapse(excerpt.removesuffix(ELLIPSIS))
                hits = [s for s in sources if text in s]
                assert len(hits) == 1, (path.stem, m.candidate_id, excerpt, hits)
                for project in persona.projects:
                    glued = _collapse(f"{project.name} {project.description}")[: len(text)]
                    assert not text.startswith(_collapse(project.name) + " "), (excerpt, glued)
                if m.candidate_id in skills_text:
                    assert text not in skills_text[m.candidate_id] or any(
                        text in s for s in sources
                    )
                    assert not skills_text[m.candidate_id].startswith(text)
                checked += 1
    assert checked > 100
