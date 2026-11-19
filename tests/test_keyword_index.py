"""rag.keyword_index: tokenizer modes, BM25 search and the corpus cases the notebook shows."""

import json
from datetime import date

import pytest

from rag.chunking import SectionChunker
from rag.embedders.fake import FakeEmbedder
from rag.indexer import index_directory
from rag.keyword_index import (
    DEFAULT_MODE,
    STOPWORDS,
    KeywordDoc,
    KeywordIndex,
    KeywordIndexError,
    TokenizerMode,
    document_tokens,
    query_tokens,
    skill_token,
    word_tokens,
)
from rag.store import VectorStore
from scripts.persona_schema import ROOT

WORDS, SKILLS, BOTH = TokenizerMode.WORDS, TokenizerMode.SKILLS, TokenizerMode.WORDS_SKILLS


def doc(chunk_id, text, candidate="c1", section="experience", body_start=0) -> KeywordDoc:
    return KeywordDoc(chunk_id, text, candidate, section, body_start)


# --- Tokenizers ---------------------------------------------------------------------------


def test_word_tokens_keep_symbol_names_whole():
    text = "C++, C#, .NET and Node.js; ASP.NET with React. Rust/Go for CI/CD."
    assert word_tokens(text) == [
        "c++",
        "c#",
        ".net",
        "node.js",
        "asp.net",
        "react",
        "rust",
        "go",
        "ci",
        "cd",
    ]


def test_word_tokens_lowercase_no_stemming_and_trailing_punctuation():
    assert word_tokens("Deployed Kubernetes clusters. Node.js.") == [
        "deployed",
        "kubernetes",
        "clusters",
        "node.js",
    ]
    assert word_tokens("5+ years") == ["5", "years"]  # "+" only sticks to a letter


def test_stopwords_dropped_but_skill_letters_kept():
    assert word_tokens("The team and I wrote it in C and R with Go") == [
        "team",
        "wrote",
        "c",
        "r",
        "go",
    ]
    assert {"c", "r", "go"}.isdisjoint(STOPWORDS)


def test_mode_parsing_and_default():
    assert DEFAULT_MODE is BOTH
    assert TokenizerMode("words+skills") is BOTH
    with pytest.raises(ValueError):
        TokenizerMode("stems")


def test_document_tokens_per_mode():
    text = "Ran k8s clusters"
    assert document_tokens(text, "experience", 0, WORDS) == ["ran", "k8s", "clusters"]
    assert document_tokens(text, "experience", 0, SKILLS) == ["skill:kubernetes"]
    assert document_tokens(text, "experience", 0, BOTH) == [
        "ran",
        "k8s",
        "clusters",
        "skill:kubernetes",
    ]


def test_aliases_become_the_same_skill_token():
    assert document_tokens("k8s", "skills", 0, SKILLS) == document_tokens(
        "Kubernetes", "skills", 0, SKILLS
    )
    assert document_tokens("Postgres, PostgreSQL", "skills", 0, SKILLS) == [
        "skill:postgresql",
        "skill:postgresql",
    ]
    assert skill_token("REST APIs") == "skill:rest_apis"
    assert skill_token("C++") == "skill:c++"


def test_ambiguous_case_rules_carry_into_skill_tokens():
    assert document_tokens("Spark jobs", "skills", 0, SKILLS) == ["skill:spark"]
    assert document_tokens("a spark of joy", "skills", 0, SKILLS) == []
    assert document_tokens("apache spark", "skills", 0, SKILLS) == ["skill:spark"]  # safe alias
    assert query_tokens("airflow spark", SKILLS) == []
    assert query_tokens("Airflow Spark", SKILLS) == ["skill:airflow", "skill:spark"]
    assert document_tokens("JavaScript", "skills", 0, SKILLS) == ["skill:javascript"]  # no Java


def test_no_implied_skill_tokens():
    assert document_tokens("Django", "skills", 0, SKILLS) == ["skill:django"]


def test_body_start_excludes_header_lines_from_skill_tokens():
    text = "Data Engineer\nSpark Energy Ltd | 2020 - Present\nBuilt Spark pipelines in Python."
    assert document_tokens(text, "experience", 2, SKILLS) == ["skill:spark", "skill:python"]
    assert document_tokens(text, "experience", 0, SKILLS).count("skill:spark") == 2
    assert "spark" in document_tokens(text, "experience", 2, WORDS)[:5]  # words see the header


@pytest.mark.parametrize("section", ["hobbies", "summary", "education", "certifications"])
def test_non_source_sections_get_words_but_no_skill_tokens(section):
    text = "Python scripting for a Raspberry Pi"
    assert document_tokens(text, section, 0, SKILLS) == []
    assert document_tokens(text, section, 0, BOTH) == word_tokens(text)
    assert "python" in document_tokens(text, section, 0, WORDS)


def test_query_text_is_always_a_skill_source():
    assert query_tokens("someone with k8s", SKILLS) == ["skill:kubernetes"]
    assert query_tokens("someone with k8s", WORDS) == ["someone", "k8s"]


# --- Search -------------------------------------------------------------------------------


@pytest.fixture
def small() -> list[KeywordDoc]:
    return [
        doc("a1", "Ran Kubernetes clusters on AWS", "alice"),
        doc("a2", "Hiking and k8s meetups", "alice", "hobbies"),
        doc("b1", "Operated k8s in production for payments", "bob"),
        doc("b2", "Kubernetes, Terraform, AWS", "bob", "skills"),
        doc("c1", "Baking sourdough bread at weekends", "carol", "hobbies"),
    ]


def ids(results):
    return [chunk_id for chunk_id, _ in results]


def test_search_skills_matches_aliases_only_in_skills_modes(small):
    assert set(ids(KeywordIndex(small, mode=SKILLS).search_skills(["Kubernetes"], 10))) == {
        "a1",
        "b1",
        "b2",
    }  # a2 says k8s but is a hobbies chunk
    assert set(ids(KeywordIndex(small, mode=WORDS).search_skills(["Kubernetes"], 10))) == {
        "a1",
        "b2",
    }


def test_search_skills_accepts_aliases_and_rejects_unknown(small):
    index = KeywordIndex(small, mode=SKILLS)
    assert index.search_skills(["k8s"], 10) == index.search_skills(["Kubernetes", "K8S"], 10)
    with pytest.raises(ValueError, match="unknown skill 'Cobol'"):
        index.search_skills(["Cobol"], 10)


def test_zero_scores_are_dropped(small):
    index = KeywordIndex(small, mode=WORDS)
    assert ids(index.search("sourdough", 10)) == ["c1"]
    assert index.search("quantum", 10) == []
    assert index.search("", 10) == []
    assert index.search("the and of", 10) == []  # only stopwords
    assert all(score > 0 for _, score in index.search("kubernetes aws", 10))


def test_results_sorted_by_score_then_chunk_id():
    # Filler documents keep "python" under half the corpus: BM25Okapi floors the IDF of
    # terms in more than half the documents, which would make every score here non-positive.
    docs = [doc(cid, "python developer") for cid in ("d", "b", "c", "a")] + [
        doc("e", "python python developer"),
        *(doc(f"z{i}", f"filler text {i}") for i in range(6)),
    ]
    results = KeywordIndex(docs, mode=WORDS).search("python", 10)
    assert ids(results) == ["e", "a", "b", "c", "d"]
    assert len({score for _, score in results[1:]}) == 1  # an exact tie, broken by id


def test_n_limits_results(small):
    index = KeywordIndex(small, mode=BOTH)
    assert len(index.search("kubernetes k8s aws", 2)) == 2
    assert index.search("kubernetes", 0) == []


def test_where_filters_candidates_and_sections(small):
    index = KeywordIndex(small, mode=WORDS)
    assert set(ids(index.search("k8s kubernetes", 10, where={"candidate_ids": ["bob"]}))) == {
        "b1",
        "b2",
    }
    assert ids(index.search("k8s kubernetes", 10, where={"sections": ["hobbies"]})) == ["a2"]
    assert (
        ids(index.search("k8s", 10, where={"candidate_ids": {"alice"}, "sections": ["experience"]}))
        == []
    )
    assert index.search("k8s", 10, where={"candidate_ids": []}) == []
    with pytest.raises(ValueError, match="unknown filter keys"):
        index.search("k8s", 10, where={"section": ["skills"]})


def test_k1_and_b_are_passed_through(small):
    default = KeywordIndex(small, mode=WORDS)
    tuned = KeywordIndex(small, mode=WORDS, k1=0.9, b=0.2)
    assert (default.bm25.k1, default.bm25.b) == (1.5, 0.75)
    assert (tuned.k1, tuned.b, tuned.bm25.k1, tuned.bm25.b) == (0.9, 0.2, 0.9, 0.2)
    assert default.search("kubernetes aws", 5) != tuned.search("kubernetes aws", 5)


def test_empty_and_tokenless_indexes():
    assert KeywordIndex([]).search("python", 5) == []
    tokenless = KeywordIndex([doc("h", "Python", section="hobbies")], mode=SKILLS)
    assert tokenless.bm25 is None and tokenless.search_skills(["Python"], 5) == []


def test_duplicate_chunk_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        KeywordIndex([doc("x", "a"), doc("x", "b")])


def test_timings_recorded(small):
    index = KeywordIndex(small)
    assert index.build_seconds >= 0 and index.last_query_seconds is None
    index.search("kubernetes", 3)
    index.search_skills(["AWS"], 3)
    assert len(index.query_seconds) == 2 and index.last_query_seconds >= 0


def test_tokens_accessor(small):
    index = KeywordIndex(small, mode=SKILLS)
    assert index.tokens("b2") == ["skill:kubernetes", "skill:terraform", "skill:aws"]
    with pytest.raises(KeyError):
        index.tokens("zzz")


class _OldStore:
    def get(self, include):
        return {"ids": ["x"], "documents": ["text"], "metadatas": [{"candidate_id": "c"}]}


def test_from_store_requires_body_start():
    with pytest.raises(KeywordIndexError, match="re-index"):
        KeywordIndex.from_store(_OldStore())


# --- Corpus: FakeEmbedder store of all 35, section chunker --------------------------------

DEVOPS_MUST_HAVES = ["Kubernetes", "Terraform", "AWS"]  # data/job_descriptions/devops_sre.txt


@pytest.fixture(scope="module")
def corpus_store(tmp_path_factory) -> VectorStore:
    chunker, embedder = SectionChunker(), FakeEmbedder()
    store = VectorStore(
        tmp_path_factory.mktemp("kw") / "chroma",
        chunker.chunker_id,
        embedder.embedder_id,
        embedder.dim,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WORKSPACE_ROOT", str(ROOT / "data"))
        report = index_directory("resumes", chunker, embedder, store, date(2026, 11, 1))
    assert len(report.added) == 35 and not report.failed
    return store


@pytest.fixture(scope="module")
def corpus(corpus_store) -> dict:
    found = corpus_store.get(include=["metadatas"])
    meta = dict(zip(found["ids"], found["metadatas"], strict=True))
    return {
        "meta": meta,
        **{m: KeywordIndex.from_store(corpus_store, mode=m) for m in TokenizerMode},
    }


def candidates(corpus, results) -> list[str]:
    return [corpus["meta"][chunk_id]["candidate_id"] for chunk_id, _ in results]


def test_corpus_index_covers_every_chunk(corpus, corpus_store):
    assert len(corpus[BOTH]) == corpus_store.count() == 228


def test_k8s_found_by_skill_tokens_not_words(corpus):
    assert "noah_williams" in candidates(corpus, corpus[SKILLS].search_skills(["Kubernetes"], 50))
    assert "noah_williams" not in candidates(corpus, corpus[WORDS].search("kubernetes", 228))


def test_airflow_spark_words_vs_skills(corpus):
    words = candidates(corpus, corpus[WORDS].search("airflow spark", 10))
    assert "peter_hughes" in words  # "airflow" in HVAC bullets, "Spark Energy Ltd"
    skills = candidates(corpus, corpus[SKILLS].search("airflow spark", 228))
    assert "peter_hughes" not in skills
    assert skills == []  # both names are ambiguous: lowercase is not a mention


def test_peter_role_chunk_has_no_spark_token(corpus):
    role = next(
        cid
        for cid, m in corpus["meta"].items()
        if m["candidate_id"] == "peter_hughes" and m.get("role_company") == "Spark Energy Ltd"
    )
    assert corpus["meta"][role]["body_start"] == 2
    assert "skill:spark" not in corpus[SKILLS].tokens(role)
    assert "spark" in corpus[WORDS].tokens(role)


def test_devops_must_haves_rank_s_candidates_first(corpus):
    with (ROOT / "data" / "ground_truth" / "qrels.json").open(encoding="utf-8") as handle:
        jd = next(j for j in json.load(handle)["jds"] if j["jd_id"] == "devops_sre")
    labels = {j["candidate_id"]: j["label"] for j in jd["judgements"]}
    strong = {c for c, label in labels.items() if label == "S"}
    weak = {c for c, label in labels.items() if label == "D"}
    assert strong == {"marcus_chen", "olga_petrova", "david_okonkwo"}

    ranked = candidates(corpus, corpus[SKILLS].search_skills(DEVOPS_MUST_HAVES, 228))
    s_ranks = [i for i, c in enumerate(ranked) if c in strong]
    d_ranks = [i for i, c in enumerate(ranked) if c in weak]
    assert s_ranks == list(range(len(s_ranks))) and len(s_ranks) >= 9  # S chunks lead
    assert all(d > max(s_ranks) for d in d_ranks)


def test_emily_never_matches_java(corpus):
    emily = {cid for cid, m in corpus["meta"].items() if m["candidate_id"] == "emily_clarke"}
    for mode in (SKILLS, BOTH):
        assert not emily & set(ids(corpus[mode].search_skills(["Java"], 228)))
        assert all("skill:java" not in corpus[mode].tokens(cid) for cid in emily)
