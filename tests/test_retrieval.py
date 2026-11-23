"""rag.retrieval: section weighting, aggregation, RRF fusion, backfill and the corpus smoke."""

from datetime import date

import pytest

from rag.chunking import SectionChunker
from rag.embedders.fake import FakeEmbedder
from rag.indexer import index_directory
from rag.jd_parser import NO_REQUIREMENTS_WARNING, parse_job_description
from rag.keyword_index import KeywordDoc, KeywordIndex
from rag.retrieval import (
    DEFAULT_SECTION_WEIGHTS,
    KEYWORD,
    SEMANTIC,
    ChunkInfo,
    RetrievalConfig,
    RetrievalMode,
    aggregate,
    rank_candidates,
    retrieve,
    rrf_fuse,
    weight_chunks,
)
from rag.store import VectorStore
from scripts.persona_schema import ROOT

K = 60
JD = parse_job_description(
    "Backend Engineer\nAcme\n\nThe role\nBuild APIs.\n\n"
    "Requirements\n- Python\n- Django\n\nNice to have\n- Docker\n\nBenefits\n- Holidays\n"
)


def info(chunk_id, candidate, section="experience", role_index=0):
    return ChunkInfo(chunk_id, candidate, section, role_index, f"text of {chunk_id}")


class StubStore:
    """Answers queries from fixed similarities; records every ``where``."""

    embedder_id = "fake-d64"

    def __init__(self, scored):
        self.scored = scored  # [(ChunkInfo, similarity)]
        self.wheres = []

    def count(self):
        return len(self.scored)

    def query(self, vector, n, where=None):
        self.wheres.append(where)
        items = [
            (i, s)
            for i, s in self.scored
            if where is None or i.candidate_id == where["candidate_id"]
        ]
        items = sorted(items, key=lambda x: (-x[1], x[0].chunk_id))[:n]
        return {
            "ids": [[i.chunk_id for i, _ in items]],
            "documents": [[i.text for i, _ in items]],
            "metadatas": [
                [
                    {"candidate_id": i.candidate_id, "section": i.section}
                    | ({"role_index": i.role_index} if i.role_index is not None else {})
                    for i, _ in items
                ]
            ],
            "distances": [[1.0 - s for _, s in items]],
        }


class StubKeyword:
    """Answers search_skills from fixed BM25 scores; records skills and filters."""

    def __init__(self, scored):
        self.scored = scored
        self.calls = []
        self.docs = {
            i.chunk_id: KeywordDoc(i.chunk_id, i.text, i.candidate_id, i.section, 0, i.role_index)
            for i, _ in scored
        }

    def search_skills(self, skills, n, where=None):
        self.calls.append((list(skills), where))
        allowed = set(where["candidate_ids"]) if where else None
        items = [
            (i.chunk_id, s) for i, s in self.scored if allowed is None or i.candidate_id in allowed
        ]
        return sorted(items, key=lambda x: (-x[1], x[0]))[:n]

    def doc(self, chunk_id):
        return self.docs[chunk_id]


class SpyEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return super().embed_query(text)


def run(sem, kw, job=JD, **config):
    store, keyword = StubStore(sem), StubKeyword(kw)
    result = retrieve(job, store, keyword, SpyEmbedder(), RetrievalConfig(**config))
    return result, store, keyword


def by_id(result):
    return {h.candidate_id: h for h in result.hits}


# --- Building blocks ----------------------------------------------------------------------


def test_rrf_arithmetic():
    fused = rrf_fuse(
        {SEMANTIC: ["a", "b", "c"], KEYWORD: ["b", "d"]}, {SEMANTIC: 1.0, KEYWORD: 1.0}, 60
    )
    assert fused == pytest.approx({"a": 1 / 61, "b": 1 / 62 + 1 / 61, "c": 1 / 63, "d": 1 / 62})


def test_rrf_list_weights():
    fused = rrf_fuse({SEMANTIC: ["a", "b"], KEYWORD: ["b", "a"]}, {SEMANTIC: 2.0, KEYWORD: 0.5}, 10)
    assert fused == pytest.approx({"a": 2 / 11 + 0.5 / 12, "b": 2 / 12 + 0.5 / 11})


def test_aggregation_with_and_without_a_second_chunk():
    config = RetrievalConfig()
    chunks = weight_chunks(
        [(info("a1", "a"), 0.8), (info("a2", "a", "skills"), 0.5), (info("b1", "b"), 0.6)],
        config,
    )
    scores = aggregate(chunks, 0.3)
    assert scores["a"].score == pytest.approx(0.8 + 0.3 * 0.35)
    assert scores["b"].score == pytest.approx(0.6)
    assert aggregate(chunks, 0.0)["a"].score == pytest.approx(0.8)
    assert [c.chunk.chunk_id for c in scores["a"].chunks] == ["a1", "a2"]


def test_negative_similarity_counts_as_zero_and_is_not_ranked():
    chunks = weight_chunks([(info("a1", "a"), -0.2), (info("b1", "b"), 0.1)], RetrievalConfig())
    assert chunks[0].weighted == 0.0 and chunks[0].raw == -0.2
    assert rank_candidates(aggregate(chunks, 0.3)) == ["b"]


def test_unweighted_sections_use_the_other_weight():
    config = RetrievalConfig(section_weights={"other": 0.25})
    assert config.section_weight("header") == 0.25
    assert config.section_weight("hobbies") == DEFAULT_SECTION_WEIGHTS["hobbies"]


# --- Config -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"section_weights": {"hobbies": -0.1}}, "section_weights"),
        ({"section_weights": {"interests": 0.5}}, "unknown section_weights"),
        ({"list_weights": {"semantic": -1}}, "list_weights"),
        ({"list_weights": {"dense": 1}}, "unknown list_weights"),
        ({"second_chunk_weight": -0.3}, "second_chunk_weight"),
        ({"rrf_k": 0}, "rrf_k"),
        ({"rrf_k": -5}, "rrf_k"),
        ({"mode": "sparse"}, "mode must be one of"),
        ({"query_source": "title"}, "query_source"),
        ({"keyword_skills": "all"}, "keyword_skills"),
        ({"pool_size": 0}, "pool_size"),
        ({"n_semantic": 2.5}, "n_semantic"),
        ({"mode": "keyword", "list_weights": {"keyword": 0}}, "weight 0"),
    ],
)
def test_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RetrievalConfig(**kwargs)


def test_config_merges_weights_and_serialises():
    config = RetrievalConfig(section_weights={"hobbies": 0.0}, mode="keyword")
    assert config.section_weights["experience"] == 1.0 and config.section_weights["hobbies"] == 0
    assert config.mode is RetrievalMode.KEYWORD
    data = config.to_dict()
    assert data["mode"] == "keyword" and data["section_weights"]["hobbies"] == 0.0
    assert data["list_weights"] == {"semantic": 1.0, "keyword": 1.0}


# --- Retrieval with stubs -----------------------------------------------------------------


def test_section_weights_demote_hobbies():
    result, _, _ = run(
        [(info("a1", "alice", "hobbies", None), 0.9), (info("b1", "bob"), 0.9)],
        [],
        mode="semantic",
    )
    assert [h.candidate_id for h in result.hits] == ["bob", "alice"]
    alice = by_id(result)["alice"]
    assert alice.semantic.raw == pytest.approx(0.9)
    assert alice.semantic.weighted == pytest.approx(0.09)


def test_hybrid_fusion_matches_hand_rrf():
    sem = [(info("a1", "a"), 0.9), (info("b1", "b"), 0.8), (info("c1", "c"), 0.7)]
    kw = [(info("b1", "b"), 5.0), (info("c1", "c"), 4.0), (info("d1", "d", "skills"), 9.0)]
    result, _, _ = run(sem, kw)
    hits = by_id(result)
    # keyword weighted: d 9*0.7=6.3, b 5.0, c 4.0 -> d, b, c
    assert hits["a"].fused_score == pytest.approx(1 / (K + 1))
    assert hits["b"].fused_score == pytest.approx(1 / (K + 2) + 1 / (K + 2))
    assert hits["c"].fused_score == pytest.approx(1 / (K + 3) + 1 / (K + 3))
    assert hits["d"].fused_score == pytest.approx(1 / (K + 1))
    assert [h.candidate_id for h in result.hits] == ["b", "c", "a", "d"]  # a/d tie: by id
    assert (hits["b"].semantic.rank, hits["b"].keyword.rank) == (2, 2)


def test_candidate_found_by_one_list_only():
    result, _, _ = run([(info("a1", "a"), 0.9)], [(info("b1", "b"), 3.0)])
    hits = by_id(result)
    assert hits["a"].keyword.rank is None and hits["a"].semantic.rank == 1
    assert hits["b"].semantic.rank is None and hits["b"].keyword.rank == 1
    assert hits["a"].fused_score == hits["b"].fused_score == pytest.approx(1 / (K + 1))


def test_backfill_semantic_for_keyword_only_candidates():
    sem = [(info("a1", "a"), 0.9), (info("b1", "b"), 0.4), (info("b2", "b", "skills"), 0.5)]
    kw = [(info("b1", "b"), 3.0)]
    store, keyword = StubStore(sem), StubKeyword(kw)
    result = retrieve(
        JD, store, keyword, SpyEmbedder(), RetrievalConfig(n_semantic=1)
    )  # the semantic list only sees a1
    bob = by_id(result)["b"]
    assert bob.semantic.rank is None and bob.semantic.backfilled
    assert bob.semantic.raw == pytest.approx(0.5)  # best raw over both chunks
    assert bob.semantic.weighted == pytest.approx(0.4)  # experience 0.4 beats skills 0.35
    assert bob.semantic.aggregate == pytest.approx(0.4 + 0.3 * 0.35)
    assert {"candidate_id": "b"} in store.wheres
    assert not by_id(result)["a"].semantic.backfilled


def test_backfill_keyword_zero_for_semantic_only_candidates():
    result, _, keyword = run([(info("a1", "a"), 0.9)], [(info("b1", "b"), 3.0)])
    alice = by_id(result)["a"]
    assert (alice.keyword.raw, alice.keyword.weighted, alice.keyword.aggregate) == (0.0, 0.0, 0.0)
    assert alice.keyword.rank is None and alice.keyword.backfilled
    assert (["Python", "Django"], {"candidate_ids": ["a"]}) in keyword.calls


def test_keyword_backfill_beyond_n_keyword_is_noted():
    kw = [(info("b1", "b"), 3.0), (info("a1", "a"), 1.0)]
    result, _, _ = run([(info("a1", "a"), 0.9)], kw, n_keyword=1)
    alice = by_id(result)["a"]
    assert alice.keyword.raw == 1.0 and alice.keyword.backfilled and alice.keyword.rank is None
    assert any("beyond n_keyword" in note for note in result.notes)


def test_semantic_mode_uses_only_the_semantic_list():
    sem = [(info("a1", "a"), 0.5), (info("b1", "b"), 0.9)]
    kw = [(info("a1", "a"), 9.0)]
    result, _, keyword = run(sem, kw, mode="semantic")
    assert [h.candidate_id for h in result.hits] == ["b", "a"]
    assert all(where is not None for _, where in keyword.calls)  # backfill only
    assert by_id(result)["a"].keyword.raw == 9.0 and by_id(result)["a"].keyword.backfilled
    assert result.ranked == {SEMANTIC: 2, KEYWORD: 0}


def test_keyword_mode_uses_only_the_keyword_list():
    sem = [(info("a1", "a"), 0.9), (info("b1", "b"), 0.1)]
    kw = [(info("b1", "b"), 4.0)]
    result, store, _ = run(sem, kw, mode="keyword")
    assert [h.candidate_id for h in result.hits] == ["b"]  # a has no BM25 score
    assert store.wheres == [{"candidate_id": "b"}]  # only the backfill queried the store
    bob = by_id(result)["b"]
    assert bob.semantic.backfilled and bob.semantic.raw == pytest.approx(0.1)
    assert result.timings.semantic_search == 0.0


def test_query_source_switches_what_is_embedded():
    for source, expected in (("query_text", JD.query_text), ("full_text", JD.full_text)):
        embedder, store = SpyEmbedder(), StubStore([(info("a1", "a"), 0.5)])
        result = retrieve(
            JD, store, StubKeyword([]), embedder, RetrievalConfig(query_source=source)
        )
        assert embedder.queries == [expected] and result.query_text == expected
    assert "Holidays" in JD.full_text and "Holidays" not in JD.query_text


def test_keyword_skills_option():
    _, _, keyword = run([], [(info("a1", "a"), 1.0)])
    assert keyword.calls[0] == (["Python", "Django"], None)
    result, _, keyword = run([], [(info("a1", "a"), 1.0)], keyword_skills="critical+nice")
    assert keyword.calls[0] == (["Python", "Django", "Docker"], None)
    assert result.keyword_skills == ("Python", "Django", "Docker")


def test_nice_to_have_fallback_with_note():
    job = parse_job_description("Data Person\n\nThe role\nApache Spark and dbt pipelines.\n")
    assert NO_REQUIREMENTS_WARNING in job.warnings and job.critical_skills == ()
    result, _, keyword = run([], [(info("a1", "a"), 1.0)], job=job)
    assert keyword.calls[0][0] == ["dbt", "Spark"]
    assert any("nice-to-have" in note for note in result.notes)


def test_no_keyword_skills_at_all():
    job = parse_job_description("Generalist\n\nRequirements\n- Kindness\n")
    result, _, keyword = run([(info("a1", "a"), 0.5)], [(info("a1", "a"), 1.0)], job=job)
    assert keyword.calls == [] and result.keyword_skills == ()
    assert by_id(result)["a"].keyword.raw == 0.0
    assert any("keyword list is empty" in note for note in result.notes)


def test_pool_size_truncation_and_tie_breaks():
    sem = [(info(f"{c}1", c), 0.5) for c in "edcba"]
    result, _, _ = run(sem, [], mode="semantic", pool_size=3)
    assert [h.candidate_id for h in result.hits] == ["a", "b", "c"]  # equal scores: by id
    chunks = [(info(c, "a"), 0.5) for c in ("z", "y")]
    hit = run(chunks, [], mode="semantic")[0].hits[0]
    assert [c.chunk_id for c in hit.top_chunks] == ["y", "z"]


def test_top_chunks_and_matched_sections_follow_contributing_chunks():
    sem = [
        (info("a_exp", "a"), 0.8),
        (info("a_sk", "a", "skills", None), 0.7),
        (info("a_hob", "a", "hobbies", None), 0.6),
    ]
    kw = [(info("a_sk", "a", "skills", None), 4.0), (info("a_proj", "a", "projects", None), 2.0)]
    hit = run(sem, kw)[0].hits[0]
    # semantic contributes a_exp (0.8) and a_sk (0.49); keyword a_sk (2.8) and a_proj (1.6)
    assert hit.matched_sections == ("experience", "projects", "skills")
    assert [c.chunk_id for c in hit.top_chunks] == ["a_sk", "a_exp", "a_proj"]
    skills_chunk = hit.top_chunks[0]
    assert skills_chunk.found_by == (SEMANTIC, KEYWORD)
    assert skills_chunk.semantic_raw == pytest.approx(0.7)
    assert skills_chunk.semantic_weighted == pytest.approx(0.49)
    assert skills_chunk.keyword_raw == 4.0 and skills_chunk.keyword_weighted == pytest.approx(2.8)
    assert hit.top_chunks[1].found_by == (SEMANTIC,) and hit.top_chunks[1].keyword_raw is None
    assert skills_chunk.text == "text of a_sk" and skills_chunk.role_index is None
    assert "a_hob" not in {c.chunk_id for c in hit.top_chunks}  # third semantic chunk


def test_timings_present():
    result, _, _ = run([(info("a1", "a"), 0.5)], [(info("a1", "a"), 1.0)])
    t = result.timings
    for stage in ("embed_query", "semantic_search", "keyword_search", "aggregate_fuse", "backfill"):
        assert getattr(t, stage) >= 0
    assert t.total >= t.embed_query
    assert set(result.to_dict()["timings"]) == {
        "embed_query",
        "semantic_search",
        "keyword_search",
        "aggregate_fuse",
        "backfill",
        "total",
    }


def test_embedder_must_match_the_store():
    store = StubStore([])
    store.embedder_id = "hf-bge-small-en-v1.5"
    with pytest.raises(ValueError, match="vectors"):
        retrieve(JD, store, StubKeyword([]), FakeEmbedder())


# --- Corpus smoke (FakeEmbedder, real Chroma) ---------------------------------------------


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    chunker, embedder = SectionChunker(), FakeEmbedder()
    store = VectorStore(
        tmp_path_factory.mktemp("retrieval") / "chroma",
        chunker.chunker_id,
        embedder.embedder_id,
        embedder.dim,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WORKSPACE_ROOT", str(ROOT / "data"))
        index_directory("resumes", chunker, embedder, store, date(2026, 11, 1))
    return store, KeywordIndex.from_store(store), embedder


JDS = sorted((ROOT / "data" / "job_descriptions").glob("*.txt"))


@pytest.mark.parametrize("mode", list(RetrievalMode))
@pytest.mark.parametrize("path", JDS, ids=[p.stem for p in JDS])
def test_corpus_smoke(corpus, path, mode):
    store, keyword_index, embedder = corpus
    job = parse_job_description(path.read_text(encoding="utf-8"))
    result = retrieve(job, store, keyword_index, embedder, RetrievalConfig(mode=mode))
    assert 0 < len(result.hits) <= 20
    ids = [h.candidate_id for h in result.hits]
    assert len(ids) == len(set(ids))
    assert [h.fused_score for h in result.hits] == sorted(
        (h.fused_score for h in result.hits), reverse=True
    )
    for hit in result.hits:
        assert hit.semantic.raw is not None and hit.keyword.raw is not None
        assert hit.top_chunks and len(hit.top_chunks) <= 3
        assert set(hit.matched_sections) >= {c.section for c in hit.top_chunks}
    if mode is RetrievalMode.KEYWORD:
        assert all(h.keyword.rank is not None for h in result.hits)
        assert all(h.semantic.backfilled for h in result.hits)
