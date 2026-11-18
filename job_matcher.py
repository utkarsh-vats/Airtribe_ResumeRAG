"""Part B: match a job description against the indexed resumes.

Planned usage (see DESIGN.md and project-plan.md, step 12): parse the JD's requirements, run
hybrid retrieval (semantic + BM25 fused with RRF), aggregate per candidate, apply must-have
filters, score 0-100 and emit the ranked matches as JSON.
"""
