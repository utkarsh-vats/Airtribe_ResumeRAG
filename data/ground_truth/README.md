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
- **Release years**: no role's `skills_used` (after `expand_implied`) may include a skill
  whose `first_release_year` in `rag/skills.py` is later than the role's start year.
  Projects carry no dates and aren't checked.
- **Encoding**: every persona string encodes in cp1252, because the PDF renderer (step 3b)
  uses reportlab's standard Helvetica. Write ASCII equivalents (`->`, `>=`) for anything
  outside it.

## Rendering variants

`scripts/layouts.py` turns a persona into format-agnostic blocks (`scripts/resume_doc.py`);
`scripts/render_txt.py` and `scripts/render_docx.py` draw them (PDF in step 3b). Regenerate
with `python -m scripts.generate_resumes [--only <id>]`. Section detection and parsing in
steps 5-6 parse against the contract below; `tests/test_generator.py` checks the rendered
files against it and checks the layout table below against the code.

### Section headings

The base heading string comes from `render.header_variant` (`HEADER_VARIANTS` in
`scripts/persona_schema.py`):

| Section | `standard` | `professional` | `history` | `employment` |
|---|---|---|---|---|
| summary | Summary | Professional Summary | Profile | About Me |
| experience | Experience | Professional Experience | Work History | Employment |
| skills | Skills | Technical Skills | Core Competencies | Skills & Tools |
| education | Education | Education | Academic Background | Qualifications |
| coursework | Coursework | Relevant Coursework | Key Coursework | Study Modules |
| projects | Projects | Selected Projects | Key Projects | Personal Projects |
| certifications | Certifications | Certifications | Licenses & Certifications | Courses & Certifications |
| hobbies | Interests | Hobbies | Interests | Outside Work |

The layout then styles it: **caps** upper-cases the whole string (`WORK HISTORY`), **title**
keeps it as above, and **colon** appends `:` (`SKILLS & TOOLS:`). A heading is always alone
on its line. A section without content (no projects, no certifications, an empty skills
section) emits no heading at all.

### Layouts

<!-- layout-table: parsed by tests/test_generator.py -->
| Layout | Heading style | TXT underline | DOCX heading | Role header | Date separator | Ongoing | Skills | Section order |
|---|---|---|---|---|---|---|---|---|
| `classic` | caps | none | styled | stacked | en dash | Present | list | summary, experience, skills, education, coursework, projects, certifications, hobbies |
| `modern` | title | none | styled | stacked | hyphen | Present | grouped | summary, skills, experience, projects, education, coursework, certifications, hobbies |
| `compact` | caps + colon | none | bold | inline | en dash | Current | list | skills, summary, experience, education, coursework, projects, certifications, hobbies |
| `sidebar` | title | none | styled | stacked | to | Present | list | [panel: skills, education, coursework], summary, experience, projects, certifications, hobbies |
| `table-skills` | caps | none | styled | stacked | hyphen | Current | table | summary, skills, experience, education, coursework, projects, certifications, hobbies |
| `academic` | title + colon | none | styled | stacked | to | present | grouped | education, coursework, summary, projects, experience, skills, certifications, hobbies |
| `plain` | caps | = | bold | stacked | hyphen | present | list | summary, skills, experience, projects, education, coursework, certifications, hobbies |
| `minimal` | title | - | bold | inline | to | Current | list | summary, experience, skills, education, coursework, projects, certifications, hobbies |

- **TXT underline**: `=` or `-` means a row of that character, the same length as the heading,
  on the next line; `none` means no underline (those layouts aren't used for TXT personas). Headings are preceded by one blank line.
- **DOCX heading**: *styled* uses Word's `Heading 1` style (the name uses `Title`); *bold* is a
  plain paragraph with a bold run (16 pt for the name, 12 pt for headings).
- **Date separator**: en dash is `" – "` (U+2013 with spaces), hyphen is `" - "`, to is
  `" to "`. Education years use the same separator. **Ongoing** is the word that replaces
  the end date of a `present` role.

### Dates (`render.date_format`)

| `date_format` | Month | Example (classic) |
|---|---|---|
| `month_name` | `Jan 2020` (3-letter English month) | `Jan 2020 – Present` |
| `iso` | `2020-01` | `2020-01 – Present` |
| `numeric` | `01/2020` | `01/2020 – Present` |
| `year_only` | `2020` | `2020 – Present` |

### Blocks within sections

- **Header**: the name on the first line, then the headline (every layout except `minimal`),
  then `email | phone | location` on one line. In `sidebar` the email, phone and location are
  three separate lines inside the panel.
- **Role header**: *stacked* is the title on one line, then `Company | <date range>` (DOCX:
  the title bold, the company line italic). *inline* is `Title | Company | <date range>` on
  one line. Bullets follow: TXT `- ` (`minimal`: `* `), DOCX `List Bullet` paragraphs. Each
  bullet is the persona text verbatim, one per line. Roles are separated by a blank line.
- **Skills**: *list* is one comma-separated line in persona order; *grouped* is one
  `Group: a, b, c` line per category; *table* is a real DOCX table (TXT: `Group | a, b, c`
  lines) with one `Group | skills` row per category and no header row. Groups appear in
  first-use order and are named Languages, Backend, Frontend, Mobile, Data, AI & Modelling,
  Databases, Cloud, DevOps & Tooling, Testing, Practices, Business Tools (none of these is a
  skill mention). Skills are shown through `render.skill_labels` where set.
- **Education**: per entry, `Degree in Field` (bold), then `Institution, <start><sep><end>`
  with ` | Grade: <grade>` when there is a grade.
- **Coursework**: its own section after Education: one `Degree in Field: course, course` line
  per entry that has coursework. Coursework is never a skill source.
- **Projects**: the project name (bold), then the description verbatim.
- **Certifications**: bullets. **Hobbies**: one comma-separated line.
- **Summary**: one paragraph, verbatim.
- **Sidebar panel** (`sidebar` only): contact lines plus the panel sections, as shaded
  paragraphs in DOCX (and boxed with `+---` / `| ` in TXT). It is a visual treatment: the
  content stays single-column, in the order shown above.
- **DOCX extraction order**: `fs_tools.read_file` walks the body with
  `iter_inner_content()`, so table rows come out in document order, as `cell | cell` lines,
  at the table's position (for `table-skills`, between the skills heading and the next
  section heading).
- **DOCX metadata**: core properties are fixed (author and last-modified-by are the
  candidate's name; created, modified and last printed are 2026-11-01T00:00:00Z; revision 1).

## Known limitations

- **Short aliases match in any case.** `ML`, `NLP`, `LLM`, `JS` and `TS` are safe aliases, so
  on real resumes "500 ml", an LLM law degree or a "TS clearance" would false-match. This
  dataset is protected only because the reverse-mention check rejects any persona text that
  mentions a skill its `skills_used` doesn't cover.
- **Single letters and unit-like tokens.** The exact-case rule for ambiguous names doesn't
  help when the plain-English use is also capitalised: "21 °C", "Grade C", "C-suite" and
  "Series C" all match `C`; "R-32 refrigerant" and "R&D" match `R`; "Go-live" at the start
  of a sentence matches `Go`. The personas avoid these phrasings; a real-resume extractor
  would need context rules.
- **Capitalised sentence starts.** "Spark", "Airflow" or "React" opening a sentence in their
  English sense count as the skill. Personas (notably `peter_hughes`) write the English word
  in lowercase instead.
- **Common words that are non-ambiguous skills.** `Hibernate`, `Azure`, `pandas`, `Tableau`,
  `Bash` and `Looker` match case-insensitively, so "hibernate", "azure sea" or "giant pandas"
  would count. None is flagged ambiguous because the skill name is the common
  spelling; the personas just don't use those words otherwise.
- **Version-suffixed names.** "Vue 3" matches `Vue.js` through the `Vue` alias, but
  "Angular 17" matches `Angular` only because the canonical name is written in exact case;
  "angular 17" would not match.
- **No dates on projects**, so the release-year check covers roles only, and project skills
  add no years.
- **Release years are coarse.** They are year-only and use the first public release (for
  example Spark 2010, before Apache Spark 1.0 in 2014; Kotlin 2011, before 1.0 in 2016), so
  the check catches anachronisms but not early-adopter edge cases within the release year.
- **No real two-column layouts.** `sidebar` is a shaded single-column block followed by the
  main column, not a true two-column page. Extracting text from genuine two-column PDFs
  (where lines from both columns interleave) is out of scope for this dataset.
- **Implications are fixed rules.** `Angular` implies `TypeScript`, but `Kotlin` and `Gradle`
  don't imply `Java`, and `Microservices` implies no language. An extractor that reads
  "JavaFX" or "Java Swing" sees `Java` only when the standalone token appears.
