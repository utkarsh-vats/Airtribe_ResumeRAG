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
| `rag/keyword_index.py` | BM25 keyword index over the same chunks |
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

### Text normalisation (`rag/text.py`)

Extracted text is cleaned into lines before anything else: PDF footers (`<Name> · Page N of
M`) and form-feed page separators are dropped, as are TXT underline rows (`====`, `----`).
Bullet glyphs (`•`, `- `, `* `) and indentation are stripped, tabs become ` | ` (so the modern
DOCX role line looks like the inline form), runs of spaces are collapsed and blank lines go.

### Section detection (`rag/sections.py`)

A line is a heading when the whole line, case-insensitive and with one optional trailing `:`,
is in the heading vocabulary (`HEADINGS`). The vocabulary covers every heading variant the
dataset renders, plus common real-world synonyms. A test keeps it in step with
`scripts/persona_schema.py` and the README's rendering contract.

| Section kind | Headings (examples) |
|---|---|
| `summary` | Summary, Professional Summary, Profile, About Me, Objective |
| `experience` | Experience, Professional Experience, Work History, Employment, Work Experience |
| `skills` | Skills, Technical Skills, Core Competencies, Skills & Tools, Tech Stack |
| `education` | Education, Academic Background, Qualifications |
| `coursework` | Coursework, Relevant Coursework, Key Coursework, Study Modules |
| `projects` | Projects, Selected Projects, Key Projects, Personal Projects |
| `certifications` | Certifications, Licenses & Certifications, Courses & Certifications |
| `hobbies` | Interests, Hobbies, Outside Work |
| `other` | an unknown short ALL-CAPS line (at most 4 words, letters only), e.g. `VOLUNTEERING` |
| `header` | everything before the first heading (name, headline, contact) |

A document with no headings becomes a header (the name plus any contact lines just after it)
and one `other` section.

### Role splitting

`split_roles` walks an experience section and recognises role headers by form, one small
function per form, tried in order (`ROLE_HEADER_FORMS`). A date range is detected by
`rag/dates.py`, which covers the month-name, ISO, numeric and year-only formats, the en dash,
hyphen and `to` separators, and Present / present / Current:

| Form | Lines |
|---|---|
| inline (and modern DOCX after tab normalisation) | `Title \| Company \| <dates>` |
| stacked | `Title`, then `Company \| <dates>` |
| split (modern PDF) | `Title \| Company`, then a line that is only `<dates>` |

Lines after a header belong to that role until the next header.

### Section-aware chunker (default)

- One chunk per section. **Experience is split per role.**
- Oversize guard: a chunk body over `max_words` (default 300; bge-small truncates at 512
  tokens) is split at line boundaries into parts, each with the same contextual header.
  Nothing in the corpus triggers it (the longest chunk is 97 words).
- Lines in an experience section before its first role (an intro sentence) become their own
  experience chunk with no role.
- Before embedding, a **contextual header line** is prepended: `Section: Experience | Role:
  Data Scientist at Acme` for experience chunks and `Section: Skills` (the section kind)
  otherwise. It gives an isolated bullet enough context to be retrievable. It deliberately
  leaves out the candidate's name: a name carries no meaning for matching and only adds
  similarity noise between unrelated chunks. The raw text (without the header) is stored
  separately for excerpts.
- `Chunk.body_start` is the line offset in the chunk text where the body begins: the number
  of role-header lines (1 for inline headers, 2 for stacked or split ones) at the top of a
  role chunk's first part, 0 everywhere else (fixed-size chunks included). It isn't part of
  the chunk id. The keyword index (§7) uses it to take skill tokens from role bodies only.
- **Every section except the header block is indexed**, including hobbies and coursework.
  Low-signal sections are down-weighted at scoring time rather than excluded, so a hobby
  mention can still be shown as context without counting as skill evidence (the mention
  contract already ignores hobbies and coursework as skill sources).

### Fixed-size baseline (experiments)

`FixedSizeChunker(size=120, overlap=30)` makes overlapping word windows over the whole
cleaned text, header included, with no contextual header, as a naive pipeline would. Each
window is labelled with the section most of its words come from, only so excerpts can still
name a section. It exists so the notebook can compare retrieval quality and latency against
the section-aware chunker. Both implement the same `Chunker` protocol (`rag/chunking.py`,
registry `get_chunker(name, **params)`) and emit the same frozen `Chunk` dataclass:
`chunk_id`, `candidate_id`, `resume_path`, `chunker_id`, `section`, `role_index`,
`role_title`, `role_company`, `part`, `text`, `embed_text`, `start_line`, `end_line`.
`chunk_id` is a SHA-256 over (resume path, chunker id, section, role index, part, text),
truncated to 16 hex chars. `chunker_id` (`section-v1`, `section-v1-noctx`,
`fixed-w120-o30`) encodes every parameter that changes the output, so collection names
derive from it.

## 4. Metadata extraction (rule-based)

Everything is deterministic and regex/taxonomy-driven. There are no LLM calls on the indexing
path.

Implemented in `rag/dates.py`, `rag/timeline.py` and `rag/metadata.py` (`extract_metadata(
resume_path, parsed, as_of) -> CandidateMetadata`).

- **Name**: the first header-block line that isn't contact info (no `@`, no `|`, no
  phone-like digit run). Falls back to the file stem (`priya_sharma_2` → `Priya Sharma 2`).
- **No contact details in metadata.** Email and phone are deliberately not extracted: metadata
  ends up in the vector store, and nothing in the brief's output needs them.
- **Education**: each entry is a `Degree in Field` line followed by an institution line
  carrying a date range (`Institution, 2014 – 2018 | Grade: …`). The degree's level comes from
  its abbreviation, matched as a whole token with dots, spaces and `(Hons)` removed (`Ph.D.`,
  `M.Sc.`, `M.S.`, `M.Tech`, `MBA`, `B.Sc.`, `B.S.`, `B.Eng.`, `B.Tech`, `B.A.`, …), or from a
  spelled-out form at the start (`Bachelor of …`, `Master of …`, `Diploma`, `High School`).
  Whole-token matching stops an institution such as "Marina Bay …" reading as `M.A.`. The
  field is the text after ` in `, and the end year is the last year of the range. Highest
  education: PhD > Masters > Bachelors > Diploma > School; a tie goes to the latest end year,
  then to the first listed.
- **Skills**: matched against a **taxonomy with aliases** (`rag/skills.py`). Each canonical
  skill (`JavaScript`, `Kubernetes`, `PostgreSQL`) has aliases (`JS`, `k8s`, `Postgres`).
  Matching uses token boundaries that handle punctuation in names (`C++`, `C#`, `.NET`,
  `Node.js`), so **`Java` never matches `JavaScript`** and `Go` doesn't match `Google`.
  Skills that are also ordinary words (`Go`, `R`, `React`, `Spark`, `Excel`) are flagged
  ambiguous. Their canonical name counts only in its exact case, while their safe aliases
  (`Golang`, `Apache Spark`) count in any case. The full contract is in
  `data/ground_truth/README.md`; `rag/skill_matcher.py` implements it independently of the
  test-side reference matcher, and a test checks that they agree on every resume source and
  every JD. A space inside a multi-word name matches any whitespace, so a name wrapped
  across PDF lines (`GitHub` / `Actions`) still counts.
  - **Sources** (`rag/metadata.py`): skills-section lines, experience role bodies (the
    bullets, not the role header lines) and project sections. Never the header block,
    summary, education, coursework, certifications, hobbies or `other` sections, so a
    "Python" in a hobby, a role title or a company name gives nothing. Skills-table
    category labels and skill-group names are never skills (the generator guarantees it and
    a test confirms the extractor agrees).
  - **Overlaps**: the longest match wins and, between equally long ones, the earlier;
    matches never overlap. `Apache Spark` is one Spark mention, `AWS Lambda` isn't also
    `AWS`, `React Native` isn't also `React`. Implications (`expand_implied`) are applied
    afterwards, so `PySpark` gives Spark and Python.
- **Experience entries**: a `RoleRecord` (`title`, `company`, `date_text`, parsed `dates`,
  `is_internship`) per role from `split_roles`. **Multi-format date parsing** covers `Jan
  2020`, `January 2020`, `01/2020`, `2020-01` and `2020`, with `Present` / `Current` / `Now`
  for an ongoing role, and `-`, `–`, `—` or `to` between the two ends. A year-only start is
  January and a year-only end is December. An ongoing end is the `as_of` month, and an end
  after `as_of` is **clamped** to it (and flagged). A start after the end raises
  `DateRangeError`; such a role is kept but its months don't count.
- **Months are the source of truth.** Role lengths are inclusive month counts
  (`end - start + 1`) and every comparison is on months: `meets_min_years(months, years)` is
  `months >= years * 12`, so 59 months never passes "5+ years".
- **Internships** are detected from the title: `intern` or `internship` as a whole word,
  case-insensitive ("Internal Tools Engineer" and "International Sales" don't match). A test
  checks that this agrees with every persona's employment type. Internships keep their skills
  but add no experience months.
- **Total experience**: role intervals are **merged when they overlap or touch** (concurrent
  jobs aren't double-counted) and gaps aren't counted. `total_years` is for display only:
  months / 12 rounded **half up** to one decimal with `Decimal` (`ROUND_HALF_UP`, so 75
  months is 6.3; Python's `round` would give 6.2). This is the same rule as the ground
  truth.
- **Per-skill months** (`skill_months`): every skill mentioned in a role's body, after
  implications, gets that role's interval if the role counts towards experience (dates
  parsed, not an internship). Intervals are merged per skill, so two overlapping roles that
  both use Python count once. A skill seen only in the skills section, projects or an
  internship gets **0 months**: it still counts as "has the skill" for coverage, but not as
  experience. `skill_years` is the display form (half up, one decimal).
- **Evidence** (`skill_evidence`): for each skill, where it was mentioned: `skills`,
  `experience:<role index>` and/or `projects`. Step 12's reasoning uses it to say where a
  skill came from.
- **Warnings**: one line per problem a reviewer should see, naming the role: dates that
  couldn't be parsed (the role counts 0 months) and end dates after `as_of` (clamped). The
  corpus produces none.
- **`CandidateMetadata.to_dict()` / `from_dict()`** round-trip through JSON; step 8 stores the
  dicts next to the collection. They hold no contact details.
- **`as_of` date is injected.** All "present" and duration maths takes an explicit
  `as_of: date`. The CLI reads `AS_OF_DATE` and uses today when it's blank. Tests and the ground
  truth use one fixed date, so results don't change with the calendar.

## 5. Embedders

```python
class Embedder(ABC):  # rag/embedders/base.py
    name: str  # registry name: "huggingface", "fake", "gemini"
    model: str  # model id within the backend
    dim: int
    max_tokens: int  # longest input before the backend truncates
    embedder_id: str  # "hf-bge-small-en-v1.5", "fake-d64"

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...  # batched
    def embed_query(self, text: str) -> list[float]: ...
    def token_count(self, text: str) -> int: ...
```

- **Contract**: every returned vector is **L2-normalised**, so cosine similarity is the dot
  product. The base class normalises whatever a backend returns, so a backend can't break
  this. Documents and queries may be encoded differently (prefixes, task types), so the two
  methods stay separate. `embed_documents` sends texts to the backend in batches of
  `batch_size` (default 32).
- **`embedder_id`** encodes everything that changes the vectors: backend, model and
  dimension. Step 8 derives collection names from it (with the chunker id), so vectors
  from different embedders or models never share a collection.

| Backend | `embedder_id` | Documents | Queries |
|---|---|---|---|
| `huggingface` (default, local) | `hf-<model name>` | raw text | model's query prefix (below) |
| `gemini` (`GEMINI_EMBEDDING_MODEL`) | `<model>-d<dims>` | see the Gemini table below | see below |
| `fake` (tests, CI) | `fake-d<dim>` | feature hashing | same |

### HuggingFace models (`MODEL_CONFIGS` in `rag/embedders/huggingface.py`)

| Model | Dim | Max tokens | Query prefix | Use |
|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | 512 | `Represent this sentence for searching relevant passages: ` | default |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | none | notebook comparison |
| `BAAI/bge-base-en-v1.5` | 768 | 512 | same as bge-small | notebook comparison |

Documents are never prefixed. Any other sentence-transformers model works with no prefix,
and a warning is logged so a missing instruction isn't silent. Vectors are requested with
`normalize_embeddings=True`. The device is auto-selected (CUDA, then MPS, then CPU) unless
overridden. `torch` and `sentence_transformers` are imported only when the model is first
used: importing `rag` or `rag.embedders` never loads them (a test checks this in a
subprocess).

- **Truncation warning**: sentence-transformers silently truncates inputs longer than the
  model's `max_seq_length`. Before encoding, every text (with its prefix) is counted with the
  model's own tokenizer, special tokens included; anything over the limit is logged with
  its token count and overflow. Section chunks are at most 300 words, comfortably under 512
  tokens, but whole JDs can come close.
- **FakeEmbedder** (`rag/embedders/fake.py`): lowercased word tokens are hashed with SHA-256
  (never Python's salted `hash()`) to a bucket and a sign; the signed counts in `dim`
  buckets (default 64) are L2-normalised. A text with no words hashes the empty token, so it
  still gets a unit vector. It's deterministic across processes and platforms, needs no
  network, and texts that share words come out closer, which is enough for meaningful
  retrieval tests. `token_count` is the word count.
- **Registry** (`rag/embedders/__init__.py`): `get_embedder(name, **params)` and
  `available_embedders()`. Backends are imported by name only when requested, so `fake`
  never imports `torch` or `google-genai`. `from_env()` reads `EMBEDDER` (default
  `huggingface`) and `HF_EMBEDDING_MODEL` (default `BAAI/bge-small-en-v1.5`), as in
  `.env.example`.
- **Smoke test**: `python -m scripts.embedder_smoke --embedder huggingface` (manual, not in
  CI) downloads the model, prints the dimension, each JD's token count against
  `max_tokens`, and a similarity matrix.
### Gemini (`rag/embedders/gemini.py`)

Source: the Gemini API "Embeddings" page (ai.google.dev/gemini-api/docs/embeddings, which
redirects to aistudio.google.com/docs/embeddings) and the two model pages, read in November
2026. Two families are supported; any other model name is refused with an error listing
them.

| | `gemini-embedding-001` | `gemini-embedding-2` |
|---|---|---|
| Input | text, 2,048 tokens | text (also image, audio, video, PDF), 8,192 tokens |
| Dimensions | 3072 default; 128-3072 via `output_dimensionality` | same |
| Task handling | `task_type`: `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` ("Task types with Embeddings 1") | no `task_type`; the task goes in the text ("Task types with Embeddings 2", asymmetric retrieval format) |
| Document text | as is | `title: none \| text: {content}` |
| Query text | as is | `task: search result \| query: {content}` |
| Inputs per request | a list of texts, up to `batch_size` (default 100) | **one**: several inputs in one request are aggregated into a single embedding ("Embedding aggregation") |
| Reduced dimensions | the caller must normalise | renormalised by the API |

- **Normalisation**: the base class L2-normalises every vector anyway, which is what makes
  reduced `gemini-embedding-001` dimensions usable for cosine similarity.
- **Batch size**: the embeddings page doesn't state a per-request limit for the list form,
  so 100 is a conservative default and `batch_size` can lower it. The docs also describe
  wrapping each input in its own `Content` object to get separate embeddings from one
  `gemini-embedding-2` request; the embedder sends one request per input instead, which is
  simpler and can't aggregate by accident.
- **`token_count`** is an offline estimate (about 4 characters per token), with no API
  call. Texts estimated over the family's input limit are logged, since the API may
  truncate them.
- **Retries**: 429 and 5xx responses are retried up to `max_retries` (default 5) with
  exponential backoff and jitter: the delay for retry *n* is drawn from [c/2, c], where
  c = min(`max_delay`, `base_delay` · 2ⁿ) (defaults 1 s and 60 s). Other 4xx errors, and
  errors without an HTTP status, fail at once. A final failure raises `EmbeddingError` with
  the model, the status and the attempt count. The sleep function, random source and clock
  are injectable, so tests never wait. The embeddings page gives no rate-limit or retry
  guidance; these defaults are ordinary practice.
- **Throttle**: `requests_per_minute` (off by default) spaces requests at least
  60 / rpm seconds apart.
- **Config**: `from_env()` reads `GEMINI_API_KEY`, `GEMINI_EMBEDDING_MODEL` (required when
  `EMBEDDER=gemini`) and `GEMINI_EMBEDDING_DIM` (blank = 3072). A missing key fails when the
  embedder is constructed. `google.genai` is imported only when the first request is made.
  `embedder_id` is `<model>-d<dims>`, e.g. `gemini-embedding-001-d768` or
  `gemini-embedding-2-d3072` (`gemini-` is added only to a model name that lacks it).

## 6. Vector store (Chroma) and indexing

### Verified Chroma facts (installed `chromadb==1.5.9`, checked 2026-11-19)

- **Collection names**: 3-512 characters from `[a-zA-Z0-9._-]`, starting and ending with a
  letter or digit, no `..`, and not an IPv4 address (each case raises
  `InvalidArgumentError`).
- **Cosine distance**: `configuration={"hnsw": {"space": "cosine"}}`, the 1.x collection
  configuration. The old `metadata={"hnsw:space": "cosine"}` still works but is the legacy
  form. A query with the same unit vector returns distance ~0.
- **No embedding function**: pass `embedding_function=None` on *every* `create_collection`,
  `get_collection` and `get_or_create_collection` call. Leaving it out of `get_collection`
  attaches Chroma's default ONNX MiniLM embedding function to the handle even for a
  collection created without one. With `None` everywhere, Chroma never constructs or calls
  it (a test patches both the default and the ONNX class to raise and runs a full index,
  reopen, query and rebuild).
- **`get_or_create_collection` keeps existing metadata**: passing different metadata for an
  existing collection doesn't overwrite it, so the stored `dim` check below is reliable.
- **Dimension**: Chroma itself rejects a vector of the wrong length on add.
- **Metadata values**: `str`, `int`, `float`, `bool`, or homogeneous **non-empty** lists of
  one of those. Empty lists raise `ValueError`; a `None` value drops the key. So keys whose
  value is `None` or `[]` are omitted. Lists filter with `{"skills": {"$contains": "Python"}}`;
  `$not_contains` also matches records that lack the key.
- Telemetry is off: `PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))`.

### Collections (`rag/store.py`, `VectorStore`)

- **One collection per (chunker, embedder)**: `resumes__<chunker_id>__<embedder_id>`, e.g.
  `resumes__section-v1__hf-bge-small-en-v1.5`. Characters outside the allowed set become `-`
  and runs of dots collapse. Only a name over 512 characters is cut and given a 12-character
  hash of the full name. Vectors from different models or chunkers never mix, and the
  notebook can keep several configurations side by side.
- **Collection metadata**: `schema_version`, `chunker_id`, `embedder_id`, `dim`. Opening an
  existing collection with a different `dim` raises `DimensionMismatchError`; another
  schema version or another configuration raises `StoreError`.
- **Per-chunk metadata** (flat, Chroma-safe): `candidate_id`, `resume_path`, `name`,
  `section`, `role_index`, `role_title`, `role_company`, `body_start`, `chunker_id`,
  `total_months`,
  `highest_degree_level` (1-5) and `skills` (canonical names). Keys with `None` or an empty
  list are left out. Chunk ids are the chunkers' content hashes (§3), so upserts are
  idempotent.
- **Document**: `chunk.text` (BM25 in step 9 and excerpts in step 12 read it). `embed_text`,
  with its contextual header, is only embedded, never stored.
- Methods: `upsert_chunks`, `delete_candidate`, `chunk_ids`, `update_candidate_metadata`,
  `count`, `list_candidates`, `query(vector, n, where)` (a thin wrapper step 11 builds on),
  `drop`, `reset`. Nothing outside `rag/store.py` imports `chromadb`.

### Sidecar

`<CHROMA_DIR>/sidecar/<collection>.json` holds what Chroma metadata can't: `schema_version`,
`chunker_id`, `embedder_id`, `as_of` (the date durations were computed with),
`pipeline_version`, and per candidate `CandidateMetadata.to_dict()` (roles, per-skill months
and evidence, education), the file's `content_hash` and `indexed_at` (UTC). It's written
atomically (temp file in the same folder, then `os.replace`) and loading checks the schema
version and the configuration.

### Indexing (`rag/indexer.py`, `index_directory`)

- **Loading** goes through `fs_tools.list_files` and `fs_tools.read_file` only, sandbox in
  force. `fs_tools` reads `WORKSPACE_ROOT` on every call (not at import) and falls back to
  its Milestone 1 default `./sample_data` when it's unset, so `index_directory` sets an unset
  or blank `WORKSPACE_ROOT` to `./data` for the duration of the run and restores it after.
- **Per file**: read -> clean / parse -> `extract_metadata(as_of)` -> chunk ->
  `embed_documents(embed_texts)` -> upsert. Only `.pdf`, `.docx` and `.txt` are indexed;
  anything else is reported as skipped (`UNSUPPORTED_FORMAT`), not failed.
- **Change detection**: `content_hash = sha256(PIPELINE_VERSION + extracted text)`.
  `PIPELINE_VERSION` (in `rag/indexer.py`) **must be bumped whenever parsing, metadata or
  chunking output changes for the same text**; a bump re-indexes every file. Version 2 added
  `body_start` to the chunk metadata. The store's `SCHEMA_VERSION` stays 1: an added key
  doesn't change how a collection is read, and a schema bump would make old collections
  refuse to open instead of re-indexing.
  - unchanged (same hash and path, same chunk ids in the store) -> skipped with no embedding.
    Its metadata is still recomputed, and if it differs (a new `as_of` changes durations)
    the chunk metadata is updated in place and the sidecar refreshed (`metadata_refreshed`).
  - changed -> the candidate's old chunks are deleted, then the new ones upserted;
  - a file no longer listed -> its chunks and sidecar entry are pruned (`removed`). Chunks in
    the store with no sidecar entry (an interrupted run) are pruned or replaced the same way.
- **Failures**: a file that fails is recorded with a code and message, and the run continues:
  fs_tools' own codes (`EXTRACTION_FAILED`, `UNSUPPORTED_FORMAT`, …), `EMPTY_TEXT` (fs_tools
  returns success with a "File is empty." warning, which becomes the message), `NO_SECTIONS`,
  `NO_CHUNKS`, `DUPLICATE_CANDIDATE` (two files with the same stem) or the exception type
  (for example an embedding error). A failed file keeps whatever it had indexed before.
- **`IndexReport`**: `added`, `updated`, `unchanged`, `removed`, `skipped`, `failed`,
  `metadata_refreshed`, `chunks_written`, and per-stage timings (`read`, `parse_extract`,
  `chunk`, `embed`, `upsert`) as totals and per file. Step 13's latency metrics reuse them.
- **`rebuild=True`** drops and recreates the collection and the sidecar first.

### CLI (`resume_rag.py`, Part A)

```text
python resume_rag.py index [--dir resumes] [--chunker section|fixed] [--chunker-opt KEY=VALUE ...]
                           [--embedder fake|huggingface|gemini] [--model ID] [--rebuild]
                           [--as-of YYYY-MM-DD] [--json]
python resume_rag.py stats
python resume_rag.py candidates [--chunker ...] [--embedder ...] [--model ID]
python resume_rag.py show CANDIDATE_ID [--chunker ...] [--embedder ...] [--model ID] [--json]
```

- **Settings**: CLI flags, then the environment (`.env` is loaded with python-dotenv and never
  overrides variables that are already set), then defaults. `WORKSPACE_ROOT` defaults to
  `./data` and `CHROMA_DIR` to `./.chroma`. `--chunker-opt` passes chunker parameters
  (`max_words=200`, `contextual_header=false`, `size=200`, `overlap=50`); `--model` sets
  `HF_EMBEDDING_MODEL` or `GEMINI_EMBEDDING_MODEL` for the chosen embedder.
- **`as_of`**: `--as-of`, else `AS_OF_DATE`, else today. The CLI is the only place that
  reads today's date (injectable in tests); everything below it takes `as_of` as an argument,
  and the sidecar records the date used.
- **`index`** prints the collection, a counts table (added / updated / unchanged / removed /
  skipped / failed / chunks written), a table of failed and skipped files (path, code,
  message), the files whose metadata was refreshed and the per-stage timings. `--json`
  prints the `IndexReport` (plus `as_of`) instead.
- **`stats`** lists every collection under `CHROMA_DIR` with its chunker, embedder,
  dimension, chunk and candidate counts, chunks by section, and the sidecar's `as_of` and
  pipeline version. It needs no embedder and creates nothing.
- **`candidates`** and **`show`** read the sidecar of the collection for the chosen chunker
  and embedder (no model is loaded): a table of id, name, total years, highest degree and
  skill count; or one candidate's education, roles (dates, months, internship flag), skills
  with years and evidence, and warnings. `show --json` prints the stored metadata dict.
- **Exit codes**: 0 success; 1 indexing finished but some files failed; 2 usage or
  configuration errors (bad date, unknown embedder or chunker, bad chunker option, missing
  Gemini settings, missing directory, unknown candidate, no index yet), reported as one line
  on stderr with no traceback.
- Rendering lives in `rag/console.py` (rich) so `job_matcher.py` can reuse it.

First real run on this machine (bge-small on CPU, 35 resumes): section chunker 228 chunks in
32 s of pipeline time (embedding 23 s, including the first model load), fixed chunker 86
chunks in 28 s; a second run finds all 35 unchanged in 0.7 s of pipeline time.

## 7. Hybrid retrieval

1. **Query text** = the JD's title, overview, responsibilities, requirements and
   nice-to-have sections, embedded with `embed_query` (decision below).
2. **Semantic**: top-N chunks (default N=50) from Chroma by cosine distance.
3. **Keyword**: top-N chunks from the BM25 index over the same chunks (below).
4. **Reciprocal Rank Fusion**: `rrf(d) = Σ 1 / (k + rank_i(d))`, with `k = 60` (configurable).
   It only uses ranks, so the two incomparable score scales never get mixed.
5. **Candidate aggregation**: chunks are grouped by `resume_path`. A candidate's retrieval
   score is the max chunk RRF plus a damped sum of the rest, so one strong chunk counts but
   breadth also helps. The **matched sections** (and chunk ids) for each candidate are recorded
   for excerpts and reasoning.

### Query text for the semantic side (step 10 decision)

The query embeds the JD's title, overview, responsibilities, requirements and nice-to-have
sections, not the whole JD. The whole of `devops_sre.txt` is 513 bge tokens against bge-small's
512 limit (step 8b smoke run), so its tail would be silently truncated; and the benefits and
location text (pension, holidays, laptops) only adds noise that pulls every JD towards every
resume. The whole-JD variant stays available so the notebook can compare the two.

### Keyword index (`rag/keyword_index.py`)

- **Built from the store**: `KeywordIndex.from_store(store)` reads every chunk's document
  (`chunk.text`) and metadata from the collection and builds a `rank_bm25.BM25Okapi` model in
  memory (`k1` 1.5 and `b` 0.75 by default, both configurable). Nothing is persisted, so the
  index can never drift from the collection; on the real section collection (228 chunks) the
  build takes milliseconds. Build time and per-query latency are recorded (`build_seconds`,
  `query_seconds`) for step 13. A collection written before `body_start` existed is rejected
  with "re-index it".
- **Tokenizer modes**:
  - `words`: case-folded word tokens over the whole chunk text. Names with symbols stay whole
    (`c++`, `c#`, `.net`, `node.js`, `asp.net`); a small English stopword list (function words
    only, never `c`, `r` or `go`) is dropped; no stemming.
  - `skills`: only canonical skill tokens (`skill:kubernetes`, `skill:rest_apis`), one per
    mention found by `rag.skill_matcher`.
  - `words+skills` (default): both in one bag.
  Queries use the same mode; for a query's skill tokens the whole query text is a source.
  `search_skills(names)` builds the query from canonical names directly (step 11 uses it for
  the critical skills): word tokens of the names, the skill tokens, or both.
- **Skill tokens follow the extraction contract** (§4): only chunks from skill-source sections
  (skills, experience, projects) get them, only from the text after `body_start`, with the
  same alias, case and boundary rules and no implications. So `k8s` and `Kubernetes` become one
  token, a lowercase `spark` or `airflow` is not a mention, `JavaScript` never yields
  `skill:java`, Peter Hughes's employer "Spark Energy Ltd" (a role-header line) yields no
  `skill:spark`, and a Python hobby gets word tokens but no skill token. The keyword side then
  agrees with what metadata extraction counts as evidence, instead of rewarding a word that
  happens to look like a skill. Words still catch what the taxonomy doesn't know.
- **Search**: `search(text, n, where)` and `search_skills(skills, n, where)` return
  `(chunk_id, score)` pairs, best first, ties broken by chunk id, zero scores dropped. `where`
  filters by `candidate_ids` and/or `sections` after scoring (IDF stays corpus-wide).
- **Length normalisation favours tiny chunks**. BM25's `b` term divides by chunk length over
  the average, so a 3-word chunk that contains a query term outscores a 56-word role chunk
  with the same term. Hobbies chunks have a median of 3.5 words (summary 24, experience 56),
  so in `words` mode a hobby line such as "Python scripting for a Raspberry Pi weather station"
  can rank near the top for "python". Step 11 offsets this with per-section weights when it
  aggregates chunks into candidates; `skills` mode avoids it for skill queries because
  non-source sections carry no skill tokens.
- **rank-bm25 quirk**: `BM25Okapi` floors the IDF of a term found in more than half the chunks at
  `0.25 × average IDF`, so a term that is everywhere contributes almost nothing (and in a tiny
  corpus can make every score non-positive, which is then dropped).
- **Fixed-size chunks** have `body_start` 0, so their role-header lines do count for skill
  tokens. That is part of what the notebook's chunker comparison measures.

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
