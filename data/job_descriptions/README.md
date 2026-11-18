# Job descriptions

Six plain-text JDs, one per theme in `data/ground_truth/jd_themes.json`, named
`<theme_id>.txt`. They're hand-written to read like real postings. The JD parser (step 10)
must recover each theme's must-haves, years and nice-to-haves from the text alone; it never
reads `jd_themes.json`. Only `*.txt` files in this folder are JDs (this README is not one).

`tests/test_job_descriptions.py` checks every rule below against the files, using the
reference splitter in `tests/jd_sections.py` and the reference skill matcher in
`tests/skill_mentions.py`.

## File format

- ASCII, LF line endings, 200-450 words.
- Line 1 is the job title and line 2 is `Company - Location`. Everything before the first
  heading is the **header**.
- A **heading** is a line that, after trimming whitespace and one trailing `:`, equals one of
  the strings below (case-insensitive). Some JDs end headings with a colon (`What you'll
  need:`); most don't.
- A section runs from its heading to the next heading. Bullets start with `- `.

## Section headings

| Section type | Heading strings |
|---|---|
| Requirements (must-haves) | `Requirements`, `What you'll need`, `Must have`, `You have`, `What we're looking for` |
| Nice-to-haves | `Nice to have`, `Bonus points`, `Preferred`, `Preferred qualifications` |
| Everything else | `About us`, `Who we are`, `About the role`, `The role`, `The team`, `What you'll do`, `Responsibilities`, `Your day-to-day`, `Location and benefits`, `Where you'll work`, `Benefits`, `Perks` |

Each JD has exactly one requirements section and one nice-to-have section. Requirements
are bullets, except in `data_engineer.txt`, where they are a prose paragraph.

## Where requirements come from

**Only the requirements section and the nice-to-have section are sources of
requirements.** Skill mentions anywhere else (the title, the company blurb, the role
overview, responsibilities, benefits) are context, not requirements. Every JD plants at least
two such skills on purpose, as keyword-search noise:

| JD | Noise skills outside the requirement sections |
|---|---|
| `python_backend` | React ("our React frontend team"), BigQuery (the data platform group) |
| `ml_engineer` | TypeScript (clinician dashboards), Kubernetes (the platform team) |
| `devops_sre` | Java and Python (what the services are written in), GitHub Actions (the deploy pipeline) |
| `frontend` | Figma (designers), Node.js (the API team) |
| `java_backend` | Kubernetes (the shared platform; also a nice-to-have), Angular (the claims tool), Jira (planning) |
| `data_engineer` | Tableau and Excel (what the analysts use) |

Within the two requirement sections, skills follow the mention contract in
`data/ground_truth/README.md`: the requirements section mentions exactly the theme's
must-have skills and the nice-to-have section exactly its nice-to-haves, with nothing else
from the taxonomy. Ambiguous skills use a safe alias or their exact-case name (`Apache Spark`,
`Apache Airflow`, `Helm`, `Go`, `dbt`, `Snowflake`).

## Years

A year phrase is a number followed by `+` or `or more` (optional) and `years`, `year` or
`yrs`. It belongs to the sentence it appears in (a bullet line, or a sentence of a prose
paragraph):

- if that sentence mentions a skill, it is that skill's **minimum years**;
- if the sentence mentions no skill, it is the **minimum total experience**.

Phrasings used:

| JD | Phrase | Meaning |
|---|---|---|
| `python_backend` | `5+ years of professional Python experience` | Python, 5 years |
| `ml_engineer` | `At least 3 years' experience writing production Python` | Python, 3 years |
| `frontend` | `React (3+ yrs) in production apps` | React, 3 years |
| `java_backend` | `5 or more years of Java development` | Java, 5 years |
| `devops_sre` | `A minimum of 4 years in DevOps, SRE or infrastructure roles` | 4 years total experience (no skill in the sentence) |
| `data_engineer` | none | no year minimums |

Total-experience minimums are phrased `A minimum of N years in <field> roles`; only
`devops_sre` has one.

## Alternatives (any_of)

A must-have satisfied by any one of several skills is written as those skills in one
sentence joined by `or` (`Production experience with Django or FastAPI.`); a longer list
would be `A, B or C`. Only `python_backend` has one (Django or FastAPI). Each other
requirement line names one skill.
