# Project plan

Vertical slices. Each one leaves `ruff check .`, `ruff format --check .` and `pytest` green. See
[DESIGN.md](DESIGN.md) for the decisions behind each step.

- [x] **1. Scaffold + Milestone 1 reuse + docs + CI** – layout, pinned requirements,
      `pyproject.toml`, `.env.example`, `.gitignore`, `fs_tools.py` and its tests copied
      unmodified, DESIGN.md, this plan, GitHub Actions (CPU torch, ruff, pytest).
- [ ] **2. Skill taxonomy + personas** – `rag/taxonomy.py` (canonical skills + aliases) and 35
      personas (8 carried over from Milestone 1 + 27 new) as structured data, plus the
      ground-truth manifest (`data/ground_truth/`).
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
