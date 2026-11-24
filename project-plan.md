# Project plan

Vertical slices. Each one leaves `ruff check .`, `ruff format --check .` and `pytest` green. See
[DESIGN.md](DESIGN.md) for the decisions behind each step.

- [x] **1. Scaffold + Milestone 1 reuse + docs + CI** – layout, pinned requirements,
      `pyproject.toml`, `.env.example`, `.gitignore`, `fs_tools.py` and its tests copied
      unmodified, DESIGN.md, this plan, GitHub Actions (CPU torch, ruff, pytest).
- [x] **2a. Skill taxonomy + persona schema + manifest** – `rag/skills.py` (canonical skills,
      aliases, implications), `scripts/persona_schema.py`, `scripts/build_manifest.py`
      (`data/ground_truth/manifest.json`), the 8 Milestone 1 personas carried over,
      `data/ground_truth/persona_plan.md` (35 slots) and ground-truth conventions + tests.
- [x] **2b. Mention rules + JD themes + first 12 new personas** – skill-mention contract in
      `data/ground_truth/README.md`, taxonomy ambiguity audit, reference matcher
      (`tests/skill_mentions.py`) with forward and reverse bullet checks,
      `data/ground_truth/jd_themes.json` + `scripts/jd_themes.py`, label-consistency tests
      (strong / distractor vs must-haves), and 12 personas authored from the plan.
- [x] **2c. Author the remaining 15 personas** from `persona_plan.md`; rebuild the manifest;
      the full plan-coverage test stops skipping. Also adds `first_release_year` to the
      taxonomy with a release-year realism check on every role, and the ground-truth "Known
      limitations".
- [x] **3a. Resume generator core** – format-agnostic block model (`scripts/resume_doc.py`),
      the 8 layouts (`scripts/layouts.py`), TXT and DOCX renderers,
      `python -m scripts.generate_resumes`, the "Rendering variants" contract in
      `data/ground_truth/README.md`, a cp1252 encoding guard, and the 21 TXT/DOCX resumes in
      `data/resumes/`.
- [x] **3b. PDF renderer** – reportlab (standard Helvetica) for the 14 PDF personas, including
      the two-page `john_doe`; byte-deterministic PDF and DOCX output; all 35 resumes in
      `data/resumes/`, with `python -m scripts.generate_resumes --check` and a byte-identity
      test.
- [x] **4. Job descriptions + qrels** – 6 JDs in `data/job_descriptions/` with their parsing
      contract (`README.md`) and tests, and `data/ground_truth/qrels.json` built by
      `python -m scripts.build_qrels` (labels and 0/1/2 grades from `persona_plan.md`,
      eligibility and unmet must-haves from the manifest).
- [x] **5a. Section detection + role splitting** – `rag/text.py` (normalising extracted
      text), `rag/dates.py` (date-range detection), `rag/sections.py` (`parse_resume`,
      `split_roles`) + unit tests and checks over all 35 resumes.
- [x] **5b. Chunkers** – `rag/chunking.py`: section-aware chunker (per-role experience chunks
      with contextual headers, oversize guard, `-noctx` ablation) and the fixed-size word-window
      baseline, a small registry, deterministic chunk ids + tests over all 35 resumes.
- [x] **6a. Dates, timeline, name, education** – `parse_date_range` (`rag/dates.py`),
      `rag/timeline.py` (merged months, internships, `meets_min_years`), `rag/metadata.py`
      (`extract_metadata`: name, roles, total months/years, education, highest education) +
      tests against the manifest for all 35 resumes. Ground-truth rounding switched to
      `ROUND_HALF_UP`.
- [x] **6b. Skills** – `rag/skill_matcher.py` (the mention contract, cross-checked against
      the reference matcher), skill sources, per-skill months and evidence, warnings and the
      JSON round-trip in `rag/metadata.py` + tests matching the manifest for all 35 resumes,
      and `tests/test_repo_hygiene.py`.
- [x] **7a. Embedders: base, fake, HuggingFace** – `Embedder` ABC (normalised vectors,
      batching, `embedder_id`), `FakeEmbedder`, lazy `HuggingFaceEmbedder` with a model
      config table and truncation warnings, registry + `from_env`, `scripts/embedder_smoke.py`
      + offline tests.
- [x] **7b. Gemini embedder** – `GeminiEmbedder` for `gemini-embedding-001` (task types,
      batched) and `gemini-embedding-2` (task prefixes, one input per request), retries with
      backoff, optional throttle, `from_env` + `.env.example`, smoke script support + tests
      against a fake client.
- [x] **8a. Vector store + indexing** – `rag/store.py` (`VectorStore`, collection naming,
      dimension checks, atomic JSON sidecar), `rag/indexer.py` (`index_directory` with
      content-hash change detection, `PIPELINE_VERSION`, pruning, per-file failures and stage
      timings) + tests, including a full-corpus run with `FakeEmbedder`.
- [x] **8b. Indexing CLI** – `resume_rag.py` (`index`, `stats`, `candidates`, `show`), settings
      precedence and exit codes, rendering in `rag/console.py` + tests; first real bge run.
- [x] **9. BM25 keyword index** – `Chunk.body_start` (pipeline v2), `rag/keyword_index.py`
      (`words` / `skills` / `words+skills` tokenizers, `search`, `search_skills`, filters,
      timings) + synthetic and corpus tests; step 10 query-text decision recorded.
- [x] **10. JD requirement parser** – `rag/jd_parser.py` (sections, units, years, any_of grouping,
      no-requirements fallback, `query_text`, `load_job_description`) + tests; smoke script
      prints query_text token counts.
- [x] **11. Hybrid retrieval** – `rag/retrieval.py` (semantic + BM25 lists, section weights,
      best + second-chunk aggregation, weighted RRF, backfill, timings) +
      `scripts/retrieval_report.py` + tests.
- [x] **12a. Matching + scoring** – `rag/requirements_check.py` (qrels-identical must-have
      checks), `rag/calibration.py` + `resume_rag.py calibrate`, `rag/scoring.py`,
      `rag/reasoning.py`, `rag/matcher.py` (strict filter-then-retrieve, soft), retrieval
      `keyword_mode` and `candidate_ids` + tests.
- [x] **12b. Output** – `rag/excerpts.py`, `rag/output.py` (brief + full JSON),
      `job_matcher.py` (`match`, `batch`, `parse`), shared `rag/cli.py`, Python-version guard,
      committed `results/samples/` + tests.
- [x] **13a. Evaluation API** – `rag/evaluation/` (metrics, ground-truth loading, ranking /
      filter / extraction / end-to-end layers, label isolation) + tests.
- [ ] **13b. Evaluation CLI + latency** – eval CLI, latency p50/p95 + tests.
- [ ] **14. Experiments notebook** – chunkers, retrieval modes, embedders (package code only).
- [ ] **15. README + docs polish.**

**Bonus (cut if needed):** Gemini-based JD parser / LLM-written reasoning; a `match_jobs` tool
for the Milestone 1 assistant.
