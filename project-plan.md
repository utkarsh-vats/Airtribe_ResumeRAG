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
- [ ] **3. Resume generator** – `scripts/` generator with several layouts, PDF/DOCX/TXT, and
      deliberate edge cases (overlapping roles, odd date formats, skills-list-only mentions,
      missing headers) → `data/resumes/`.
- [ ] **4. Job descriptions + qrels** – 6 JDs in `data/job_descriptions/` and
      `data/ground_truth/qrels.json` (hand-authored, reviewed, 0/1/2 per JD × candidate).
- [ ] **5. Chunking** – section detection + section-aware and fixed-size chunkers + tests.
- [ ] **6. Metadata extractor** – name, education, skills, roles, merged years, per-skill years
      (injected `as_of`) + tests against the manifest.
- [ ] **7. Embedders** – `Embedder` ABC + lazy registry (`huggingface`, `gemini`, `fake`) + tests.
- [ ] **8. Vector store + indexing** – Chroma `ResumeStore`, indexing pipeline, `resume_rag.py`
      CLI (`index`, `stats`) + tests, including one that pins the Chroma metadata-type
      behaviour.
- [ ] **9. BM25 keyword index** + tests.
- [ ] **10. JD requirement parser** + tests.
- [ ] **11. Hybrid retrieval** – semantic + BM25 + RRF + candidate aggregation + tests.
- [ ] **12. Scoring + output** – score components, must-have filter, reasoning/excerpts, JSON
      output, `job_matcher.py` CLI + tests.
- [ ] **13. Evaluation** – metrics (P@5, R@10, MRR, nDCG@10) + latency, eval CLI + tests.
- [ ] **14. Experiments notebook** – chunkers, retrieval modes, embedders (package code only).
- [ ] **15. README + docs polish.**

**Bonus (cut if needed):** Gemini-based JD parser / LLM-written reasoning; a `match_jobs` tool
for the Milestone 1 assistant.
