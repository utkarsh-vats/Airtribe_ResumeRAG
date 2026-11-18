# Design: Airtribe Resume RAG (Milestone 2)

> **Provenance.** `fs_tools.py` and its tests (`tests/conftest.py`, `tests/helpers.py`,
> `tests/test_sandbox.py`, `tests/test_read_file.py`, `tests/test_list_files.py`,
> `tests/test_write_file.py`, `tests/test_search_in_file.py`) are copied **unmodified** from
> Milestone 1 ([Airtribe_LLMFileAssistant](https://github.com/utkarsh-vats/Airtribe_LLMFileAssistant),
> commit `19ffe3a`). `fs_tools.py` imports only the standard library, `pypdf` and `python-docx`.
> It does not depend on Milestone 1's `assistant/` package. Fix bugs upstream first, then
> re-copy, so the two repos don't drift.

A retrieval-augmented system that indexes resumes into a vector database and ranks candidates
against a job description (JD) with hybrid search and a 0-100 match score.

- **Part A** – `resume_rag.py`: load → section-detect → chunk → extract metadata → embed →
  Chroma.
- **Part B** – `job_matcher.py`: parse JD → hybrid retrieval → score → JSON.
- Library code lives in the `rag/` package. It is named `rag` rather than `resume_rag` so it
  can't be confused with the root `resume_rag.py` script when importing.

## 1. Pipeline overview

```
Part A – indexing
┌───────────────┐   ┌────────────────┐   ┌──────────┐   ┌───────────────────┐   ┌─────────┐   ┌────────┐
│ fs_tools      │──▶│ section detect │──▶│  chunk   │──▶│ extract metadata  │──▶│  embed  │──▶│ Chroma │
│ list/read_file│   │ (header map)   │   │ (section │   │ name, skills,     │   │ (Embed- │   │ upsert │
└───────────────┘   └────────────────┘   │  -aware) │   │ roles, years …    │   │  der)   │   └────────┘
                                         └──────────┘   └───────────────────┘   └─────────┘
                                              │                                            ┌────────┐
                                              └──────────────────────────────────────────▶ │  BM25  │
                                                                                           └────────┘
Part B – matching
┌────┐   ┌───────────────────┐   ┌─────────────────────┐   ┌─────┐   ┌───────────────┐   ┌───────────┐   ┌───────┐   ┌──────┐
│ JD │──▶│ parse requirements│──▶│ semantic top-N      │──▶│ RRF │──▶│ aggregate per │──▶│ must-have │──▶│ score │──▶│ JSON │
└────┘   │ must / nice / yrs │   │ + BM25 top-N chunks │   └─────┘   │ candidate     │   │ filter    │   │ 0-100 │   └──────┘
         └───────────────────┘   └─────────────────────┘             └───────────────┘   └───────────┘   └───────┘
```

Planned module layout (built slice by slice, see [project-plan.md](project-plan.md)):

| Path | Role |
|---|---|
| `fs_tools.py` | Sandboxed file loading (Milestone 1, unmodified) |
| `rag/loader.py` | Walks `WORKSPACE_ROOT` through `fs_tools` and yields documents |
| `rag/sections.py` | Section header detection (synonym map) |
| `rag/chunking.py` | `SectionAwareChunker`, `FixedSizeChunker` |
| `rag/taxonomy.py` | Skill taxonomy + alias matching |
| `rag/metadata.py` | Rule-based extractor (name, education, skills, roles, years) |
| `rag/dates.py` | Multi-format date parsing + interval merging |
| `rag/embedders/` | `Embedder` ABC, registry, `huggingface`, `gemini`, `fake` |
| `rag/store.py` | Chroma repository (`ResumeStore`) |
| `rag/bm25.py` | Keyword index over the same chunks |
| `rag/jd_parser.py` | Rule-based JD requirement parser |
| `rag/retrieval.py` | Hybrid retrieval, RRF, candidate aggregation |
| `rag/scoring.py` | Score components, must-have filter, reasoning, excerpts |
| `rag/evaluation.py` | qrels loading, IR metrics, latency |
| `resume_rag.py` / `job_matcher.py` | CLIs (thin: argument parsing + rendering) |

## 2. Loading through `fs_tools` (sandbox stays in force)

All resume and JD reads go through `fs_tools.list_files` / `fs_tools.read_file`. There is no
direct `open()` or `pypdf` call in `rag/`. That way the Milestone 1 sandbox (path resolution
inside `WORKSPACE_ROOT`, symlink-escape protection, size limits, the extraction fallbacks)
covers everything Milestone 2 reads.

- `WORKSPACE_ROOT=./data`. Paths passed to the tools are workspace-relative
  (`resumes/…`, `job_descriptions/…`).
- `fs_tools` falls back to its **Milestone 1 default `./sample_data`** when `WORKSPACE_ROOT` is
  unset. We don't edit the copied module, so the CLIs load `.env` and set
  `WORKSPACE_ROOT` to `./data` if it's missing. Tests set it to a `tmp_path` workspace through
  the copied `workspace` fixture.
- `list_files` is **non-recursive**, skips dotfiles (so `.gitkeep` never shows up) and
  **raises** `FsToolError`. `read_file` **never raises** and returns the envelope
  `{success, data: {content, metadata}, error}`. The loader handles both conventions. A failed
  read (corrupt PDF, empty text layer, unsupported type) is logged and counted in the indexing
  report. It never aborts the whole run.
- `metadata.modified` and `size_bytes` are informational. Change detection uses content hashes
  (§6).

## 3. Chunking

### Section detection

Lines are matched against a **header synonym map** after normalisation (case-folded, trailing
`:` and decoration such as `====` or `---` stripped, length-limited so body lines don't match):

| Canonical section | Synonyms (examples) |
|---|---|
| `summary` | Summary, Profile, Professional Summary, About Me, Objective |
| `experience` | Experience, Work Experience, Professional Experience, Employment History |
| `skills` | Skills, Technical Skills, Core Competencies, Tech Stack |
| `education` | Education, Academic Background, Qualifications |
| `projects` | Projects, Selected Projects, Personal Projects |
| `certifications` | Certifications, Licenses & Certifications |
| `other` | anything before the first header, or unrecognised headers |

### Section-aware chunker (default)

- One chunk per section. **Experience is split per role.** A role starts at a line that
  carries a date range (`Jan 2020 – Present`, `2019-2021`, `03/2018 - 06/2020`) or follows the
  `Title | Company` pattern.
- Long sections are split on paragraph or bullet boundaries to stay under a token budget, with
  a small overlap.
- Before embedding, a **contextual header line** is prepended:
  `Candidate: Priya Sharma | Section: Experience | Role: Data Scientist @ Acme (2021-Present)`.
  This gives an isolated bullet enough context to be retrievable. The raw text (without the
  header) is stored separately for excerpts.

### Fixed-size baseline (experiments)

`FixedSizeChunker(size, overlap)` makes character/token windows that ignore structure. It
exists only so the notebook can compare retrieval quality and latency against the
section-aware chunker. Both implement the same `Chunker` protocol and emit the same `Chunk`
dataclass (`text`, `embed_text`, `section`, `role_index`, `ordinal`, `char_span`).

## 4. Metadata extraction (rule-based)

Everything is deterministic and regex/taxonomy-driven. There are no LLM calls on the indexing
path.

- **Name**: first non-empty line that isn't a header, contact line (email, phone, URL) or
  location. Falls back to the filename stem (`resume_priya_sharma` → `Priya Sharma`).
- **Education**: degree keywords (B.Tech, BSc, MSc, MBA, PhD, …) + institution + year from the
  `education` section.
- **Skills**: matched against a **taxonomy with aliases** (`rag/skills.py`). Each canonical
  skill (`JavaScript`, `Kubernetes`, `PostgreSQL`) has aliases (`JS`, `k8s`, `Postgres`).
  Matching uses token boundaries that handle punctuation in names (`C++`, `C#`, `.NET`,
  `Node.js`), so **`Java` never matches `JavaScript`** and `Go` doesn't match `Google`.
  Skills that are also ordinary words (`Go`, `R`, `React`, `Spark`, `Excel`) are flagged
  ambiguous. Their canonical name counts only in its exact case, while their safe aliases
  (`Golang`, `Apache Spark`) count in any case. Mentions count only in skills sections,
  bullets and project descriptions, never in hobbies, coursework, titles, company names or
  institution names. The full contract is in `data/ground_truth/README.md`.
- **Experience entries**: `{title, company, start, end, is_current, text}` for each role.
  **Multi-format date parsing** covers `Jan 2020`, `January 2020`, `01/2020`, `2020-01`,
  `2020`, and `Present` / `Current` / `Now`. Ranges use `-`, `–`, `—` or `to`.
- **Total years**: role intervals are **merged when they overlap** (concurrent jobs aren't
  double-counted). Gaps aren't counted. The result is rounded to one decimal place.
- **Per-skill years**: a skill gets the duration of each role whose text mentions it (merged
  the same way). A skill that only appears in the skills list gets **0 evidenced years**. It
  still counts as "has the skill" for coverage, but not as experience.
- **`as_of` date is injected.** All "present" and duration maths takes an explicit
  `as_of: date`. The CLI reads `AS_OF_DATE` and uses today when it's blank. Tests and the ground
  truth use one fixed date, so results don't change with the calendar.

## 5. Embedders

```python
class Embedder(ABC):
    name: str  # "huggingface:BAAI/bge-small-en-v1.5"
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

Documents and queries are embedded differently, so the two methods stay separate:

| Backend | Documents | Queries |
|---|---|---|
| `huggingface` (default, `BAAI/bge-small-en-v1.5`, 384-d, local) | raw text, normalised | prefixed `"Represent this sentence for searching relevant passages: "` |
| `gemini` (`GEMINI_EMBEDDING_MODEL`) | `gemini-embedding-001`: `task_type=RETRIEVAL_DOCUMENT` | `task_type=RETRIEVAL_QUERY` |
| `fake` (tests only) | deterministic hash-based vectors | same |

- **Gemini note:** the current model is `gemini-embedding-2`. It doesn't accept `task_type`;
  the task instruction goes in the input text. It also returns *one aggregated* embedding when
  given several inputs in a single content, so batches must be sent as separate contents.
  `gemini-embedding-001` (text-only) still supports `task_type`. The Gemini embedder branches
  on the model id.
- **Registry** (`rag/embedders/__init__.py`): `get_embedder(name, **config)`. Backends are
  **imported lazily**, so `fake` never imports `torch` or `google-genai`.
- **FakeEmbedder**: tokens are hashed (stable `hashlib` hash, not Python's salted `hash()`)
  into a fixed-dimension bag-of-words vector and L2-normalised. It's deterministic, needs no
  network, and texts that share words come out similar, which is enough for meaningful
  retrieval tests.

## 6. Vector store (Chroma)

- `ResumeStore` wraps a `chromadb.PersistentClient(path=CHROMA_DIR)` (**repository pattern**).
  Nothing outside `rag/store.py` imports `chromadb`. Telemetry is off via
  `Settings(anonymized_telemetry=False)`.
- **Collection name** comes from the embedder plus the chunker config, e.g.
  `resumes__hf-bge-small-en-v1-5__section-v1` (sanitised to Chroma's naming rules: 3-512 chars,
  `[a-zA-Z0-9._-]`, alphanumeric at both ends). Vectors from different models or chunkers never
  mix, and the notebook can keep several configurations side by side. Distance is cosine
  (`configuration={"hnsw": {"space": "cosine"}}`).
- **Chunk ids are content hashes**: `sha256(resume_path + chunker_config + embed_text)`.
  Upserts are idempotent. A file-level hash is stored on each chunk, so **unchanged files are
  skipped** without re-embedding, and chunks of a changed or removed file are deleted before
  re-upserting.
- **Metadata value types** (checked against the installed `chromadb==1.5.9`, 2026-11-18):
  - Accepted: `str`, `int`, `float`, `bool`, and **homogeneous, non-empty lists** of one of
    those types.
  - Rejected: nested dicts, mixed-type lists (`[1, "a"]`) and **empty lists**. Each raises
    `ValueError`.
  - `None` values are accepted but the key is dropped (a `{"x": None}` metadata reads back as
    `None`).
  - Lists filter with `{"skills": {"$contains": "python"}}`. `$not_contains` also matches
    records that **lack the key**.
- **How skills are stored**: `skills` is a native `list[str]` of canonical taxonomy ids on every
  chunk. The key is **omitted** when a candidate has no skills (empty lists are rejected).
  Pre-filtering on must-haves combines one `$contains` per skill under `$and`. The scorer
  re-checks coverage against the sidecar anyway, so the filter is an optimisation rather than
  the source of truth. Structured metadata that Chroma can't hold (roles, per-skill years,
  education) is kept in a JSON sidecar per collection
  (`.chroma/<collection>/candidates.json`). Chunk metadata only carries flat fields
  (`candidate_name`, `resume_path`, `section`, `file_hash`, `total_years`, `skills`).
  A store test pins this behaviour, so a Chroma upgrade that changes it fails loudly.

## 7. Hybrid retrieval

1. **Query text** = the JD's requirement summary (title + must-haves + nice-to-haves +
   responsibilities), embedded with `embed_query`.
2. **Semantic**: top-N chunks (default N=50) from Chroma by cosine distance.
3. **Keyword**: top-N chunks from a `rank_bm25.BM25Okapi` index built over the same chunk
   texts. It's tokenised with the same taxonomy normalisation, so `k8s` and `Kubernetes`
   collide, and it's rebuilt from the store on load.
4. **Reciprocal Rank Fusion**: `rrf(d) = Σ 1 / (k + rank_i(d))`, with `k = 60` (configurable).
   It only uses ranks, so the two incomparable score scales never get mixed.
5. **Candidate aggregation**: chunks are grouped by `resume_path`. A candidate's retrieval
   score is the max chunk RRF plus a damped sum of the rest, so one strong chunk counts but
   breadth also helps. The **matched sections** (and chunk ids) for each candidate are recorded
   for excerpts and reasoning.

## 8. Scoring (0-100)

```
score = 100 × ( w_sem · semantic + w_must · must_cov + w_nice · nice_cov + w_exp · exp_fit )
        default weights: 0.40 / 0.30 / 0.15 / 0.15 (sum to 1, configurable)
```

- **semantic**: the candidate's best chunk cosine similarity, **rescaled over a fixed
  per-model range** `[lo, hi]` (e.g. bge-small ≈ `[0.30, 0.85]`, calibrated once in the
  notebook) and clamped to `[0, 1]`. Per-query min-max is deliberately *not* used: it would
  always give the top candidate 1.0 and the bottom 0.0, even when nobody fits, and scores
  wouldn't be comparable across JDs.
- **must_cov / nice_cov**: the fraction of must-have / nice-to-have skills found in the
  candidate's extracted skills (taxonomy-normalised). Empty lists count as 1.0.
- **exp_fit**: `min(1, candidate_years / required_years)` for total years, averaged with
  per-skill year requirements ("5+ years of Python" uses *evidenced* Python years).
- **Must-haves are hard filters.** A candidate missing any must-have skill is removed before
  ranking, whatever their score. The output reports how many were filtered (and, at debug
  verbosity, why).
- **Reasoning is templated** from the score components, e.g. *"Has 4/4 must-haves and 2/3
  nice-to-haves (missing: Terraform). 6.5 years of experience vs 5 required. Strongest match
  in Experience (Senior Data Engineer @ Acme)."* It's deterministic and testable, and every
  claim traces back to a component. LLM-written reasoning is an optional bonus.
- **relevant_excerpts**: the top 2-3 matched chunks' raw text (without the contextual header),
  trimmed around the matched terms.

## 9. JD parsing (rule-based)

- Detect sections by headers: *Requirements*, *Must have* / *Required*, *Nice to have* /
  *Preferred* / *Bonus*, *Responsibilities*. A bullet's section decides whether its skills are
  must-have or nice-to-have. Skills outside any section default to nice-to-have.
- **Years patterns**: `N+ years (of) X`, `N-M years`, `at least N years`, `minimum N years`.
  When X resolves to a taxonomy skill it becomes a per-skill requirement. Otherwise (e.g.
  "5+ years of backend development") it's total experience.
- Skills are matched with the same taxonomy and alias rules as resumes (§4), so JD and resume
  vocabularies line up.
- Output: `JobRequirements(title, must_have, nice_to_have, min_years, skill_years, raw_text)`.

## 10. Output format

Exactly as the brief specifies, one JSON object per JD (written to `results/<jd>.json` and
printed with `rich` on request):

```json
{
  "job_description": "Senior Python Backend Engineer ...",
  "top_matches": [
    {
      "candidate_name": "John Doe",
      "resume_path": "resumes/resume_john_doe.pdf",
      "match_score": 87,
      "matched_skills": ["python", "django", "postgresql"],
      "relevant_excerpts": ["Led migration of the billing service to Django 5 ..."],
      "reasoning": "Has 4/4 must-haves and 2/3 nice-to-haves ..."
    }
  ]
}
```

Diagnostics (filtered count, score components, timing) go to stderr or a `--debug` field,
**never** into the brief's schema.

## 11. Evaluation

- **Ground truth** (`data/ground_truth/qrels.json`) holds graded relevance (`0` not relevant,
  `1` partial, `2` strong) for every JD × candidate pair. The grades are the labels assigned
  by hand in `persona_plan.md`; `scripts/build_qrels.py` only copies them and adds each
  candidate's must-have eligibility from the manifest. It is **never generated by, or tuned
  against, the scorer's output**, since that would make the evaluation circular. The
  semantics are in "Relevance semantics" in `data/ground_truth/README.md`.
- **Metrics** (`rag/evaluation.py`), reported in two parts because eligible == strong in this
  dataset. Ranking before the must-have filter: graded nDCG@10; P@5, R@10 and MRR in a strict
  form (grade 2) and a lenient form (grade ≥ 1); and distractor intrusion@10. Filter accuracy:
  keep/drop decisions against `eligible`. Each metric is a pure function over
  `(ranked_ids, qrels)` and is unit-tested on hand-computed examples.
- **Latency**: index build time (total and per document), and per-query p50/p95 **split into
  embed / search (semantic + BM25 + RRF) / score**, measured with `time.perf_counter` over
  repeated runs after one warm-up.
- **Notebook** (`notebooks/experiments.ipynb`) only calls package code (`rag.evaluation`,
  `rag.retrieval`, …) and holds no logic of its own. It compares section-aware vs fixed-size
  chunking, semantic-only vs BM25-only vs hybrid, and huggingface vs gemini.

## 12. Testing

- Unit tests use the **FakeEmbedder** and `tmp_path` workspaces and Chroma dirs. **No network
  calls or model downloads** happen in tests or CI. The `huggingface` and `gemini` embedders are
  tested with their client or model object stubbed. Integration runs with real models are
  manual (notebook / CLI).
- **No hardcoded "today"** and nothing non-deterministic: every date computation takes `as_of`,
  hashes are `hashlib`-based, ordering is always made total (ties broken by `resume_path`), and
  sets are sorted before serialising.
- Sample resumes for tests are built in `tmp_path` with the copied `make_pdf` / `make_docx`
  fixtures. The generated corpus under `data/` is only used for manifest checks and evaluation.
- CI (GitHub Actions, Python 3.14) installs **CPU-only torch** from the PyTorch CPU index, then
  runs `ruff check .`, `ruff format --check .` and `pytest`.
