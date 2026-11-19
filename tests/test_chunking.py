"""rag.chunking: the section-aware chunker, the fixed-size baseline and the registry."""

import hashlib
import itertools
import json
from dataclasses import replace

import pytest

import fs_tools
from rag.chunking import (
    DEFAULT_MAX_WORDS,
    Chunk,
    FixedSizeChunker,
    SectionChunker,
    available_chunkers,
    chunk_id,
    get_chunker,
    word_count,
)
from rag.sections import SectionKind, parse_resume, split_roles
from scripts.build_manifest import MANIFEST_PATH
from scripts.persona_schema import ROOT, load_personas

PERSONAS = load_personas()
any_persona = pytest.mark.parametrize("persona", PERSONAS, ids=[p.id for p in PERSONAS])

DOC = """Ann Lee
ann@example.com | +1 555-0100
SUMMARY
Builds reliable things.
EXPERIENCE
An intro line before any role.
Senior Engineer
Acme | 2020 - Present
Led the team.
Shipped the product.
Engineer | Beta Corp | 2018 - 2019
Wrote code.
SKILLS
Python, SQL
PROJECTS
"""
PATH = "resumes/ann_lee.txt"


def collapse(text: str) -> str:
    return " ".join(text.split())


# --- SectionChunker -----------------------------------------------------------------------


def test_section_chunks_fields_and_order():
    parsed = parse_resume(DOC)
    chunks = SectionChunker().chunk(PATH, parsed)
    summary = [
        (c.section, c.role_index, c.role_title, c.role_company, c.text, c.start_line, c.end_line)
        for c in chunks
    ]
    assert summary == [
        ("summary", None, None, None, "Builds reliable things.", 3, 4),
        ("experience", None, None, None, "An intro line before any role.", 5, 6),
        (
            "experience",
            0,
            "Senior Engineer",
            "Acme",
            "Senior Engineer\nAcme | 2020 - Present\nLed the team.\nShipped the product.",
            6,
            10,
        ),
        (
            "experience",
            1,
            "Engineer",
            "Beta Corp",
            "Engineer | Beta Corp | 2018 - 2019\nWrote code.",
            10,
            12,
        ),
        ("skills", None, None, None, "Python, SQL", 13, 14),
    ]  # the empty PROJECTS section gives no chunk; the header block is skipped
    for chunk in chunks:
        assert chunk.candidate_id == "ann_lee"
        assert chunk.resume_path == PATH
        assert chunk.chunker_id == "section-v1"
        assert chunk.part == 0
        assert "\n".join(parsed.lines[chunk.start_line : chunk.end_line]) == chunk.text
        assert chunk.chunk_id == chunk_id(
            PATH, chunk.chunker_id, chunk.section, chunk.role_index, chunk.part, chunk.text
        )


def test_contextual_header_formats():
    chunks = SectionChunker().chunk(PATH, parse_resume(DOC))
    headers = [c.embed_text.split("\n", 1)[0] for c in chunks]
    assert headers == [
        "Section: Summary",
        "Section: Experience",
        "Section: Experience | Role: Senior Engineer at Acme",
        "Section: Experience | Role: Engineer at Beta Corp",
        "Section: Skills",
    ]
    assert all(c.embed_text == f"{h}\n{c.text}" for c, h in zip(chunks, headers, strict=True))
    assert "Ann Lee" not in " ".join(c.embed_text for c in chunks)


def test_noctx_mode_embeds_the_text_only():
    chunker = SectionChunker(contextual_header=False)
    assert chunker.chunker_id == "section-v1-noctx"
    chunks = chunker.chunk(PATH, parse_resume(DOC))
    assert chunks and all(c.embed_text == c.text for c in chunks)
    with_ctx = SectionChunker().chunk(PATH, parse_resume(DOC))
    assert [c.text for c in chunks] == [c.text for c in with_ctx]
    assert {c.chunk_id for c in chunks}.isdisjoint(c.chunk_id for c in with_ctx)


def test_experience_without_roles_is_one_preamble_chunk():
    chunks = SectionChunker().chunk(
        PATH, parse_resume("Ann\nEXPERIENCE\nDid many things.\nAnd more.\n")
    )
    assert [(c.section, c.role_index, c.text) for c in chunks] == [
        ("experience", None, "Did many things.\nAnd more."),
    ]


def test_oversize_chunks_split_at_line_boundaries():
    lines = [f"line {i} has five words" for i in range(6)]  # 5 words each
    doc = "Ann\nSUMMARY\n" + "\n".join(lines) + "\n"
    chunks = SectionChunker(max_words=12).chunk(PATH, parse_resume(doc))
    assert [c.part for c in chunks] == [0, 1, 2]
    assert [c.text.split("\n") for c in chunks] == [lines[0:2], lines[2:4], lines[4:6]]
    assert all(word_count(c.text) <= 12 for c in chunks)
    assert all(c.embed_text.startswith("Section: Summary\n") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == 3
    assert [(c.start_line, c.end_line) for c in chunks] == [(2, 4), (4, 6), (6, 8)]


def test_oversize_single_line_split_by_words():
    long_line = " ".join(f"w{i}" for i in range(25))
    doc = f"Ann\nSUMMARY\nshort line\n{long_line}\n"
    chunks = SectionChunker(max_words=10).chunk(PATH, parse_resume(doc))
    assert [c.text for c in chunks] == [
        "short line",
        " ".join(f"w{i}" for i in range(10)),
        " ".join(f"w{i}" for i in range(10, 20)),
        " ".join(f"w{i}" for i in range(20, 25)),
    ]
    assert all(word_count(c.text) <= 10 for c in chunks)


def test_section_chunker_validates_max_words():
    with pytest.raises(ValueError):
        SectionChunker(max_words=0)


# --- FixedSizeChunker ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "size", "overlap", "expected"),
    [
        (10, 4, 1, [(0, 4), (3, 7), (6, 10)]),
        (11, 4, 1, [(0, 4), (3, 7), (6, 10), (9, 11)]),  # last window not padded
        (3, 4, 1, [(0, 3)]),  # shorter than one window
        (4, 4, 0, [(0, 4)]),
        (0, 4, 1, []),
    ],
)
def test_fixed_windows(n, size, overlap, expected):
    assert FixedSizeChunker(size, overlap).windows(n) == expected


@pytest.mark.parametrize(("n", "size", "overlap"), [(250, 120, 30), (37, 10, 3), (9, 5, 4)])
def test_fixed_windows_cover_every_word_with_the_overlap(n, size, overlap):
    spans = FixedSizeChunker(size, overlap).windows(n)
    assert spans[0][0] == 0 and spans[-1][1] == n
    assert all(end - start <= size for start, end in spans)
    for (_s1, e1), (s2, _e2) in itertools.pairwise(spans):
        assert e1 - s2 == overlap


def test_fixed_chunks_text_header_included_and_no_context():
    parsed = parse_resume(DOC)
    chunker = FixedSizeChunker(size=8, overlap=2)
    chunks = chunker.chunk(PATH, parsed)
    words = " ".join(parsed.lines).split()
    for chunk, (start, end) in zip(chunks, chunker.windows(len(words)), strict=True):
        assert chunk.text.split() == words[start:end]
        assert chunk.embed_text == chunk.text
        assert (chunk.role_index, chunk.role_title, chunk.role_company) == (None, None, None)
        assert chunk.chunker_id == "fixed-w8-o2"
    assert chunks[0].text.startswith("Ann Lee\nann@example.com")  # header included
    assert chunks[0].section == "header"
    assert [c.part for c in chunks] == list(range(len(chunks)))


def test_fixed_short_document_is_one_chunk():
    chunks = FixedSizeChunker().chunk(PATH, parse_resume(DOC))
    assert len(chunks) == 1
    assert chunks[0].text.split() == " ".join(parse_resume(DOC).lines).split()


def test_fixed_majority_section_label():
    doc = "Ann\nSKILLS\none two three four five six\nSUMMARY\nseven eight\n"
    (chunk,) = FixedSizeChunker(size=50, overlap=0).chunk(PATH, parse_resume(doc))
    assert chunk.section == "skills"  # 7 of 11 words (heading included)
    tie = "Ann\nSKILLS\nb\nSUMMARY\nc\n"  # header 1, skills 2, summary 2: first wins on a tie
    (chunk,) = FixedSizeChunker(size=50, overlap=0).chunk(PATH, parse_resume(tie))
    assert chunk.section == "skills"


def test_fixed_chunker_validates_params():
    for size, overlap in ((0, 0), (10, 10), (10, -1)):
        with pytest.raises(ValueError):
            FixedSizeChunker(size, overlap)


# --- ids and registry ---------------------------------------------------------------------


def test_chunk_id_deterministic_and_sensitive_to_every_field():
    base = ("resumes/a.pdf", "section-v1", "experience", 0, 0, "Led the team.")
    first = chunk_id(*base)
    assert first == chunk_id(*base) and len(first) == 16 and int(first, 16) >= 0
    changes = [
        (0, "resumes/b.pdf"),
        (1, "section-v1-noctx"),
        (2, "skills"),
        (3, 1),
        (4, 1),
        (5, "Led the teams."),
    ]
    variants = set()
    for index, value in changes:
        args = list(base)
        args[index] = value
        variants.add(chunk_id(*args))
    assert len(variants) == len(changes) and first not in variants
    assert chunk_id(*base[:3], None, 0, base[5]) != first  # role_index None vs 0


def test_registry_round_trip():
    assert available_chunkers() == ["fixed", "section"]
    assert get_chunker("section").chunker_id == "section-v1"
    assert get_chunker("section", contextual_header=False).chunker_id == "section-v1-noctx"
    assert get_chunker("section", max_words=200).chunker_id == "section-v1-w200"
    assert get_chunker("fixed").chunker_id == "fixed-w120-o30"
    assert get_chunker("fixed", size=200, overlap=50).chunker_id == "fixed-w200-o50"
    assert isinstance(get_chunker("fixed"), FixedSizeChunker)
    with pytest.raises(ValueError, match="available"):
        get_chunker("semantic")


def test_chunk_is_frozen():
    chunk = SectionChunker().chunk(PATH, parse_resume(DOC))[0]
    assert isinstance(chunk, Chunk)
    with pytest.raises(AttributeError):
        chunk.text = "x"  # type: ignore[misc]
    assert replace(chunk, text="x").text == "x"


# --- Corpus -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> dict:
    """``{candidate_id: (resume_path, ParsedResume)}`` for all 35 resumes via fs_tools."""
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = {c["candidate_id"]: c for c in json.load(handle)["candidates"]}
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WORKSPACE_ROOT", str(ROOT / "data"))
        for persona in PERSONAS:
            path = manifest[persona.id]["resume_path"]
            result = fs_tools.read_file(path)
            assert result["success"], result["error"]
            out[persona.id] = (path, parse_resume(result["data"]["content"]))
    return out


@pytest.fixture(scope="module")
def section_chunks(corpus) -> dict[str, list[Chunk]]:
    chunker = SectionChunker()
    return {pid: chunker.chunk(path, parsed) for pid, (path, parsed) in corpus.items()}


@any_persona
def test_section_chunk_count(persona, corpus, section_chunks):
    _, parsed = corpus[persona.id]
    expected = 0
    for section in parsed.sections:
        if section.kind is SectionKind.EXPERIENCE:
            roles = split_roles(section)
            expected += len(roles) + (
                1 if not roles or roles[0].start_line > section.start_line + 1 else 0
            )
        elif section.lines:
            expected += 1
    assert len(section_chunks[persona.id]) == expected


@any_persona
def test_role_chunks_match_persona(persona, section_chunks):
    roles = [c for c in section_chunks[persona.id] if c.role_index is not None]
    assert [c.role_index for c in roles] == list(range(len(persona.experience)))
    for chunk, role in zip(roles, persona.experience, strict=True):
        assert (chunk.role_title, chunk.role_company) == (role.title, role.company)
        assert chunk.embed_text.startswith(
            f"Section: Experience | Role: {role.title} at {role.company}\n"
        )
        assert chunk.candidate_id == persona.id


@any_persona
def test_every_bullet_in_exactly_one_chunk(persona, section_chunks):
    texts = [collapse(c.text) for c in section_chunks[persona.id]]
    for role in persona.experience:
        for bullet in role.bullets:
            assert sum(collapse(bullet) in t for t in texts) == 1, bullet


@any_persona
def test_no_contact_details_and_size_guard(persona, section_chunks):
    for chunk in section_chunks[persona.id]:
        assert persona.email not in chunk.embed_text
        assert persona.phone not in chunk.embed_text
        assert word_count(chunk.text) <= DEFAULT_MAX_WORDS
        assert chunk.part == 0  # nothing in the corpus needs splitting


def test_chunk_ids_unique_and_stable(corpus, section_chunks):
    ids = [c.chunk_id for chunks in section_chunks.values() for c in chunks]
    assert len(ids) == len(set(ids))
    again = SectionChunker()
    rerun = [
        c.chunk_id for pid, (path, parsed) in corpus.items() for c in again.chunk(path, parsed)
    ]
    assert rerun == ids


@any_persona
def test_fixed_chunks_cover_the_text(persona, corpus):
    path, parsed = corpus[persona.id]
    chunker = FixedSizeChunker()
    chunks = chunker.chunk(path, parsed)
    words = " ".join(parsed.lines).split()
    rebuilt = list(chunks[0].text.split())
    for chunk in chunks[1:]:
        rebuilt += chunk.text.split()[chunker.overlap :]
    assert rebuilt == words
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.section in {k.value for k in SectionKind} for c in chunks)


def test_fixed_chunk_ids_unique_across_corpus(corpus):
    chunker = FixedSizeChunker()
    ids = [c.chunk_id for path, parsed in corpus.values() for c in chunker.chunk(path, parsed)]
    assert len(ids) == len(set(ids))


# --- body_start ---------------------------------------------------------------------------


def test_body_start_counts_role_header_lines():
    chunks = SectionChunker().chunk(PATH, parse_resume(DOC))
    starts = {(c.section, c.role_index): c.body_start for c in chunks}
    assert starts == {
        ("summary", None): 0,
        ("experience", None): 0,  # the preamble
        ("experience", 0): 2,  # stacked header: title, then company | dates
        ("experience", 1): 1,  # inline header
        ("skills", None): 0,
    }
    role = next(c for c in chunks if c.role_index == 0)
    assert role.text.split("\n")[role.body_start :] == ["Led the team.", "Shipped the product."]


def test_body_start_is_zero_after_the_first_part():
    bullets = [f"bullet {i} has five words" for i in range(4)]
    doc = "Ann\nEXPERIENCE\nEngineer | Acme | 2020 - 2021\n" + "\n".join(bullets) + "\n"
    chunks = SectionChunker(max_words=12).chunk(PATH, parse_resume(doc))
    assert [(c.part, c.body_start) for c in chunks] == [(0, 1), (1, 0), (2, 0)]


def test_fixed_chunks_have_body_start_zero(corpus):
    path, parsed = corpus["john_doe"]
    assert {c.body_start for c in FixedSizeChunker().chunk(path, parsed)} == {0}


@any_persona
def test_role_chunk_body_is_the_role_body(persona, corpus, section_chunks):
    _, parsed = corpus[persona.id]
    roles = [r for s in parsed.of_kind(SectionKind.EXPERIENCE) for r in split_roles(s)]
    for chunk in section_chunks[persona.id]:
        if chunk.role_index is not None:
            body = tuple(chunk.text.split("\n")[chunk.body_start :])
            assert body == roles[chunk.role_index].lines


# Sorted chunk ids of every resume under both default chunkers, from the chunker before
# body_start existed: adding the field must not change any id.
CORPUS_CHUNK_IDS_SHA256 = "2575c7092cab435525e013f2f8416f968a2cc2f95b8f86e78c27d317dc843252"


def test_body_start_does_not_change_chunk_ids(corpus):
    ids = sorted(
        c.chunk_id
        for path, parsed in corpus.values()
        for chunker in (SectionChunker(), FixedSizeChunker())
        for c in chunker.chunk(path, parsed)
    )
    assert len(ids) == 314
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest() == CORPUS_CHUNK_IDS_SHA256
    chunk = SectionChunker().chunk(PATH, parse_resume(DOC))[2]
    assert replace(chunk, body_start=0).chunk_id == chunk_id(
        PATH, chunk.chunker_id, chunk.section, chunk.role_index, chunk.part, chunk.text
    )
