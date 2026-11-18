# Ground truth

The answer key for extraction and evaluation. Everything here is either hand-authored or
derived from hand-authored data by `scripts/build_manifest.py`. **Nothing is ever produced by
the runtime pipeline in `rag/`**, because grading the system against its own output would be
circular.

| Path | Contents | Authored by |
|---|---|---|
| `personas/<id>.json` | One persona per candidate (schema: `scripts/persona_schema.py`) | hand |
| `persona_plan.md` | 35 slots, intended JD relevance, edge-case placement, layouts | hand |
| `manifest.json` | Per-candidate skills, years and education | `python -m scripts.build_manifest` |
| `qrels.json` (step 4) | Graded relevance (0/1/2) per JD × candidate | hand |

After editing any persona, run `python -m scripts.build_manifest`. The test suite fails if
`manifest.json` is stale; `python -m scripts.build_manifest --check` gives the same check
from the command line.

## Conventions

These are enforced in code (`scripts/build_manifest.py`, `tests/test_personas.py`).

- **`AS_OF = 2026-11-01`**: one fixed reference date (`scripts.persona_schema.AS_OF`) for
  every ground-truth computation. Nothing reads the real "today".
- **Role months are inclusive**: `(end - start) + 1`, so a role from `2020-01` to `2020-01`
  is 1 month. `"present"` means the `AS_OF` month (`2026-11`).
- **Total years** = the merged (non-overlapping) months of all **non-internship** roles,
  divided by 12 and rounded to 1 decimal place. Concurrent roles count once and gaps don't
  count. A month count divided by 12 never lands exactly on a rounding tie.
- **Per-skill years** = the merged months of the non-internship roles whose `skills_used`,
  after `expand_implied`, include the skill. Every candidate skill appears in
  `skill_years`. A skill that's listed but used in no role gets `0.0`.
- **Candidate skills** = `expand_implied(skills_section ∪ role skills_used ∪ project
  skills_used)`. **Hobbies and coursework are NOT skills**; they're deliberate traps. A role
  that's an internship still contributes its skills, just no years. Certifications are plain
  text and contribute nothing.
- **`date_format: year_only`**: since the resume shows only years, the persona's role
  starts must be `-01` and its ends `-12` (or `present`). That way the data never contains a
  month that the rendered resume hides.
- **Highest education** = the highest degree level (PhD > Masters > Bachelors > Diploma >
  School). Degree strings map to levels in `DEGREE_LEVELS`; on a tie, the most recent
  `end_year` wins.
- Skills are canonical names from `rag/skills.py`. `render.skill_labels` can change how a
  skill is *displayed* (for example a lowercase `python`, or an alias), but each label must
  canonicalize back to its skill.

## Authoring rules

- `id` is the slug of `name` (`firstname_lastname`, lowercased, non-alphanumerics become
  `_`). A second persona with the same name gets the suffix `_2`. The filename is `<id>.json`.
- Contact details are obviously fictional: an `@example.com` email and a `+<cc> 555-01NN`
  phone number, both unique.
- Dates: `start <= end <= AS_OF`, and education `start_year <= end_year <= 2026`.
- Every non-internship role lists at least one skill in `skills_used` (non-tech roles use
  business skills such as Excel or Salesforce).
- Every skill in a role's `skills_used` is mentioned in that role's bullets, by canonical
  name or alias. Bullets **should not** mention skills the role doesn't use. This is not
  enforced yet (ambiguous names make it noisy), but the step 6 extractor tests against the
  manifest will surface any mismatch.
- `skills_section` is empty if and only if the persona is tagged `no_skills_section`.
  `layout: table-skills` comes with the tag `skills_table`. `internships_only` means every
  role is an internship.
- Edge-case tags come from `EDGE_CASES` in `scripts/persona_schema.py`.
