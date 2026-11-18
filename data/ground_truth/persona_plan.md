# Persona plan (35 slots)

Design for the synthetic resume corpus: 8 personas carried over from Milestone 1 (already in
`personas/`) plus 27 to author in step 2b. `tests/test_personas.py` parses the **Slots** table
below. It checks the coverage rules against this table now, and checks each persona file
against its row as personas are added.

## JD themes and intended must-haves

| Key | JD | Must-haves |
|---|---|---|
| `py_backend` | Senior Python Backend Engineer | 5+ yrs Python; Django **or** FastAPI; PostgreSQL |
| `ml` | Machine Learning Engineer | 3+ yrs Python; PyTorch; Machine Learning |
| `devops` | DevOps / SRE | 4+ yrs experience; Kubernetes; Terraform; AWS |
| `frontend` | Frontend Engineer | 3+ yrs React; TypeScript |
| `java` | Java Backend Engineer | 5+ yrs Java; Spring Boot; Microservices |
| `data_eng` | Data Engineer | SQL; Spark; Airflow |

Relevance labels (intended; the hand-authored qrels in step 4 are the real ground truth):

- **strong** – meets every must-have, including the years.
- **partial** – a plausible, adjacent candidate missing a must-have or the years.
- **distractor** – built to share surface keywords or a title with the JD while failing its
  must-haves. It must rank below strong and partial candidates.
- **none** – unrelated.

## Slots

| id | archetype | seniority | years | format | layout | date_format | edge_cases | py_backend | ml | devops | frontend | java | data_eng |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `john_doe` | Senior Python backend (M1) | senior | 8.9 | pdf | classic | year_only | two_page | strong | none | partial | none | none | none |
| `priya_sharma` | Data scientist (M1) | mid | 5.4 | pdf | modern | month_name | implied_skill_evidence | none | partial | none | none | none | partial |
| `marcus_chen` | DevOps / platform (M1) | senior | 7.5 | pdf | compact | numeric | - | none | none | strong | none | none | none |
| `elena_rodriguez` | Java / Spring (M1) | mid | 6.2 | docx | table-skills | month_name | skills_table | none | none | none | none | strong | none |
| `aisha_okafor` | Frontend TS / React (M1) | mid | 3.4 | docx | sidebar | iso | - | none | none | none | strong | none | none |
| `liam_walsh` | QA automation (M1) | mid | 4.5 | docx | modern | month_name | - | none | none | distractor | distractor | none | none |
| `sofia_rossi` | Data analyst SQL / R (M1) | junior | 2.9 | txt | plain | year_only | - | none | none | none | none | none | partial |
| `kenji_tanaka` | Staff ML engineer (M1) | staff | 10.9 | txt | plain | year_only | lowercase_skill | none | strong | partial | none | none | partial |
| `ananya_iyer` | Python backend, FastAPI + PostgreSQL | senior | 7 | docx | modern | iso | - | strong | none | none | none | none | none |
| `noah_williams` | Python backend writing only aliases (Postgres, k8s, JS, ML) | senior | 6 | txt | plain | numeric | aliases_only | strong | partial | partial | none | none | none |
| `mateo_garcia` | Python dev, full-time + concurrent freelance (merged ~4.3, summed ~5.6) | mid | 4.3 | pdf | compact | month_name | overlapping_roles | partial | none | none | none | none | none |
| `daniel_osei` | Teacher turned Django developer (~2.5 Python yrs) | mid | 11 | docx | classic | month_name | career_changer | partial | none | none | none | none | none |
| `omar_haddad` | PHP / Laravel dev listing Python + Django, never used | mid | 6 | pdf | sidebar | numeric | listed_never_used | distractor | none | none | none | none | none |
| `wei_zhang` | ML engineer, PyTorch CV, FastAPI model serving | senior | 6 | pdf | modern | iso | - | partial | strong | none | none | none | none |
| `fatima_al_sayed` | NLP research engineer, PhD, PyTorch + Hugging Face | mid | 4 | pdf | academic | month_name | - | none | strong | none | none | none | none |
| `ishaan_gupta` | Fresh graduate, ML internships only | graduate | 0 | docx | academic | month_name | internships_only | none | partial | none | none | none | none |
| `hannah_berg` | Business analyst; ML + PyTorch in MSc coursework only | mid | 5 | docx | classic | year_only | coursework_only_skill | none | distractor | none | none | none | distractor |
| `olga_petrova` | SRE, Kubernetes + Terraform + AWS, 2-year career gap | senior | 9 | pdf | modern | month_name | career_gap | none | none | strong | none | none | none |
| `david_okonkwo` | Principal cloud engineer, Java developer early career | principal | 17 | pdf | classic | year_only | - | none | none | strong | none | partial | none |
| `tomas_novak` | Linux sysadmin titled "Infrastructure Engineer" (Ansible, Bash) | senior | 10 | txt | plain | numeric | - | none | none | distractor | none | none | none |
| `lucas_martin` | Senior frontend, React + TypeScript + Next.js | senior | 8 | pdf | sidebar | month_name | - | none | none | none | strong | none | none |
| `emily_clarke` | JavaScript-heavy full-stack (Node, Express, React, TS) | mid | 5 | docx | compact | numeric | js_for_java_jd | none | none | none | strong | distractor | none |
| `priya_sharma_2` | Vue.js + TypeScript frontend dev, same name as `priya_sharma` | mid | 4 | txt | plain | iso | duplicate_name | none | none | none | partial | none | none |
| `mohammed_rahman` | Java Swing / JavaFX desktop UI developer | senior | 7 | docx | modern | iso | java_for_frontend_jd | none | none | none | distractor | partial | none |
| `zoe_turner` | Junior React dev, JavaScript only, very short resume | junior | 1.5 | txt | minimal | iso | minimal | none | none | none | partial | none | none |
| `rahul_verma` | Senior Java backend, Spring Boot microservices | senior | 9 | pdf | compact | month_name | - | none | none | none | none | strong | none |
| `hans_mueller` | Java architect, long career | principal | 16 | docx | classic | year_only | - | none | none | none | none | strong | none |
| `kwame_mensah` | .NET / C# microservices + Angular / TypeScript | senior | 8 | pdf | classic | numeric | - | none | none | none | partial | partial | none |
| `yuki_sato` | Android developer, Kotlin + Gradle | mid | 6 | txt | plain | month_name | - | none | none | none | none | distractor | none |
| `nina_kowalski` | Data engineer, Spark + Airflow + SQL, no Skills section | senior | 6 | txt | plain | numeric | no_skills_section | none | none | none | none | none | strong |
| `arjun_nair` | Big-data engineer, PySpark + Airflow + Kafka on AWS | senior | 8 | pdf | modern | iso | - | none | none | none | none | none | strong |
| `grace_kim` | Analytics engineer, dbt + Snowflake + Spark + Airflow | mid | 4 | docx | table-skills | month_name | skills_table | none | none | none | none | none | strong |
| `peter_hughes` | HVAC design engineer ("airflow", Spark Energy Ltd); Python hobby | senior | 11 | pdf | classic | year_only | python_in_hobbies, ambiguous_terms | distractor | distractor | none | none | none | distractor |
| `maria_santos` | HR generalist | senior | 9 | docx | sidebar | iso | non_tech | none | none | none | none | none | none |
| `james_o_brien` | Sales account executive (Salesforce, HubSpot) | mid | 7 | txt | plain | month_name | non_tech | none | none | none | none | none | none |

Years are approximate for planned slots (±1 is accepted against the computed manifest) and
exact for the 8 carried over.

### Coverage check

| JD | strong | partial | distractor |
|---|---|---|---|
| `py_backend` | john_doe, ananya_iyer, noah_williams | mateo_garcia, daniel_osei, wei_zhang | omar_haddad, peter_hughes |
| `ml` | kenji_tanaka, wei_zhang, fatima_al_sayed | priya_sharma, noah_williams, ishaan_gupta | hannah_berg, peter_hughes |
| `devops` | marcus_chen, olga_petrova, david_okonkwo | john_doe, kenji_tanaka, noah_williams | liam_walsh, tomas_novak |
| `frontend` | aisha_okafor, lucas_martin, emily_clarke | priya_sharma_2, zoe_turner, kwame_mensah | liam_walsh, mohammed_rahman |
| `java` | elena_rodriguez, rahul_verma, hans_mueller | david_okonkwo, mohammed_rahman, kwame_mensah | emily_clarke, yuki_sato |
| `data_eng` | nina_kowalski, arjun_nair, grace_kim | priya_sharma, sofia_rossi, kenji_tanaka | hannah_berg, peter_hughes |

- Formats: 14 pdf / 12 docx / 9 txt.
- Date formats: month_name 13, iso 8, numeric 7, year_only 7.
- Non-tech, irrelevant to every JD: `maria_santos`, `james_o_brien`.
- Seniority: `ishaan_gupta` (internships only, 0 years) up to `david_okonkwo` (17) and
  `hans_mueller` (16).

### Edge-case placement

| Edge case | Slot | What it tests |
|---|---|---|
| Python only in hobbies | `peter_hughes` | Hobbies are not skills; a keyword hit must not make him a Python candidate |
| JavaScript-heavy candidate for the Java JD | `emily_clarke` | `JavaScript` must not satisfy `Java` |
| Java candidate for the frontend JD | `mohammed_rahman` | `Java` + "UI" must not satisfy React / TypeScript |
| No Skills section | `nina_kowalski` | Skills extracted from experience text only |
| Skills in a DOCX table | `elena_rodriguez`, `grace_kim` | Table text extraction (fs_tools renders tables as lines) |
| Aliases only | `noah_williams` | `Postgres`, `k8s`, `JS`, `ML` canonicalised (render.skill_labels + alias wording in bullets) |
| Overlapping concurrent roles | `mateo_garcia` | Merged ~4.3 years must fail "5+ years" (naive sum ~5.6 would pass) |
| Career gap | `olga_petrova` | The gap is not counted as experience |
| Career changer | `daniel_osei` | ~11 total years but ~2.5 Python years: fails "5+ yrs Python" |
| Skill in coursework only | `hannah_berg` | PyTorch / ML in coursework are not skills |
| Listed but never used | `omar_haddad` | Python + Django listed, 0 evidenced years: fails "5+ yrs Python" |
| Two candidates with the same name | `priya_sharma` + `priya_sharma_2` | Results keyed by resume path, not by name |
| Long two-page resume | `john_doe` (Milestone 1: page 2 holds Projects, Certifications, Education) | Multi-page PDF extraction and chunking |
| Minimal resume | `zoe_turner` | Few chunks, no summary, still scored sensibly |
| Ambiguous terms | `peter_hughes` | `airflow` (HVAC) and `Spark` (company name) outside a skills section |
| Implied evidence | `priya_sharma` | Python never in bullets; evidenced through pandas / scikit-learn / Airflow |
| Lowercase skill | `kenji_tanaka` | `python` (Milestone 1 fact) |

## Layouts (generator, step 3)

| Layout | Description | Formats |
|---|---|---|
| `classic` | Single column; name, contact line, caps headings, bulleted roles | pdf, docx |
| `modern` | Name banner + headline, ruled headings, skills grouped by category inline | pdf, docx |
| `compact` | Dense; role, company and dates on one line, skills as one comma list | pdf, docx |
| `sidebar` | Two-column design linearised to one column: contact, skills, education first, then roles | pdf, docx |
| `table-skills` | Classic, but the Skills section is a DOCX table (category / skills) | docx |
| `academic` | Education first, coursework and projects prominent; long-form | pdf, docx |
| `plain` | Plain text, headings underlined with `=` / `-`, `-` bullets | txt |
| `minimal` | Name, contact, experience and education only; no summary or extras | txt, pdf, docx |

Section-heading variants (`render.header_variant`, defined in `scripts/persona_schema.py`):

| Variant | Summary | Experience | Skills | Education | Projects | Certifications | Hobbies |
|---|---|---|---|---|---|---|---|
| `standard` | Summary | Experience | Skills | Education | Projects | Certifications | Interests |
| `professional` | Professional Summary | Professional Experience | Technical Skills | Education | Selected Projects | Certifications | Hobbies |
| `history` | Profile | Work History | Core Competencies | Academic Background | Key Projects | Licenses & Certifications | Interests |
| `employment` | About Me | Employment | Skills & Tools | Qualifications | Personal Projects | Courses & Certifications | Outside Work |

## Notes for step 2b

- `peter_hughes`'s company "Spark Energy Ltd" and his "airflow" bullets use ambiguous names
  (`Spark`, `Airflow`) outside any skills section. Leave them out of `skills_used`; the
  bullet check ignores unlisted mentions.
- `noah_williams` needs `render.skill_labels` (e.g. `{"PostgreSQL": "Postgres",
  "Kubernetes": "k8s"}`) and bullets that use the aliases.
- `daniel_osei`'s teaching roles need non-empty `skills_used` (e.g. Excel) because every
  non-internship role must list skills.
- `david_okonkwo`'s early Java roles must come without Spring Boot so the Java JD stays partial.
