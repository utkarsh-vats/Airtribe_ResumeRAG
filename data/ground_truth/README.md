# Ground truth

The answer key for extraction and evaluation. Everything here is either hand-authored or
derived from hand-authored data by `scripts/build_manifest.py`. **Nothing is ever produced by
the runtime pipeline in `rag/`**, because grading the system against its own output would be
circular.

| Path | Contents | Authored by |
|---|---|---|
| `personas/<id>.json` | One persona per candidate (schema: `scripts/persona_schema.py`) | hand |
| `persona_plan.md` | 35 slots, intended JD relevance, edge-case placement, layouts | hand |
| `jd_themes.json` | The six JD themes as structured must-haves / nice-to-haves (`scripts/jd_themes.py`) | hand |
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
  skill is *displayed* (for example a lowercase `python`, or an alias), but the label must
  still be recognised as that skill under the mention rules below. An ambiguous skill
  therefore can't get a lowercase label.

## Skill-mention rules (the extractor contract)

These rules say when text counts as a mention of a skill. The step 6 extractor in `rag/`
must implement them. `tests/skill_mentions.py` is an independent reference implementation
that the persona tests use, so the persona text and `manifest.json` are guaranteed to agree
with the rules.

1. **Where mentions count**: Skills sections, role bullets and project descriptions. Nothing
   else is a skill source for ground truth.
2. **Never counted**: hobbies, education coursework, role title lines, company names and
   institution names. Summary, headline and certification text aren't skill sources either.
3. **Non-ambiguous skills** match by canonical name or any alias, case-insensitively. The
   match must sit on word boundaries: no letter, digit or `_` directly before or after it.
   This boundary rule has to work for names with symbols (`C++`, `C#`, `.NET`, `Node.js`,
   `CI/CD`, `A/B tests`), so `Java` never matches inside `JavaScript`.
4. **Ambiguous skills** (`ambiguous=True` in `rag/skills.py`; ordinary words or very short
   names such as `Spark`, `Airflow`, `React`, `Go`, `R`, `Excel`):
   - *Safe aliases* (`Apache Spark`, `Golang`, `ReactJS`, `Microsoft Excel`) match
     case-insensitively anywhere a mention can count.
   - The *canonical name* matches only in its exact case (`Spark`, never `spark`), in skills
     sections, bullets and project descriptions alike.
5. **Overlaps**: when matches overlap, the longest wins. `React Native` is not also `React`,
   `AWS Lambda` is not also `AWS`, and `Tailwind CSS` is not also `CSS`. The implications
   (`AWS Lambda` → `AWS`) then add the shorter skill back where it is genuinely implied.
6. **Implications** (`expand_implied`) are applied after matching. A role whose bullets say
   "Django" evidences Python as well.

Consequences for authoring: a bullet must not use a capitalised ambiguous name in its plain
English sense (for example "React quickly to…" at the start of a sentence, "Series C", or
"R&D"). Use lowercase or reword it.

## Authoring rules

- `id` is the slug of `name` (`firstname_lastname`, lowercased, non-alphanumerics become
  `_`). A second persona with the same name gets the suffix `_2`. The filename is `<id>.json`.
- Contact details are obviously fictional: an `@example.com` email and a `+<cc> 555-01NN`
  phone number, both unique.
- Dates: `start <= end <= AS_OF`, and education `start_year <= end_year <= 2026`.
- Every non-internship role lists at least one skill in `skills_used` (non-tech roles use
  business skills such as Excel or Salesforce).
- **Forward rule**: every skill in a role's (or project's) `skills_used` is mentioned in its
  bullets (or description) under the mention rules, directly or through an implication of a
  mentioned skill.
- **Reverse rule**: bullets and project descriptions mention no skill outside what their
  `skills_used` covers after `expand_implied`.
- Each role has 3-5 bullets (Milestone 1 personas keep their original 1-3).
- Persona labels in `persona_plan.md` must agree with `jd_themes.json`: strong means every
  must-have is met, distractor means at least one fails, and meeting them all means strong
  or partial.
- `skills_section` is empty if and only if the persona is tagged `no_skills_section`.
  `layout: table-skills` comes with the tag `skills_table`. `internships_only` means every
  role is an internship.
- Edge-case tags come from `EDGE_CASES` in `scripts/persona_schema.py`.
