"""rag.dates parsing, rag.timeline and rag.metadata: unit tests plus the 35-resume corpus."""

import itertools
import json
from datetime import date

import pytest

import fs_tools
from rag.dates import DateRange, DateRangeError, YearMonth, parse_date_range
from rag.metadata import (
    DegreeLevel,
    Education,
    degree_level,
    extract_metadata,
    extract_name,
    highest_education,
    parse_education,
)
from rag.sections import Section, SectionKind, parse_resume
from rag.timeline import (
    build_timeline,
    is_internship_title,
    meets_min_years,
    merge_intervals,
    merged_months,
    months_to_years,
)
from scripts.build_manifest import MANIFEST_PATH
from scripts.persona_schema import PRESENT, ROOT, EmploymentType, load_personas

AS_OF = date(2026, 11, 1)
NOW = YearMonth(2026, 11)
PERSONAS = load_personas()
BY_ID = {p.id: p for p in PERSONAS}
any_persona = pytest.mark.parametrize("persona", PERSONAS, ids=[p.id for p in PERSONAS])


# --- Date parsing -------------------------------------------------------------------------

STARTS = {"month_name": "Mar 2020", "iso": "2020-03", "numeric": "03/2020", "year_only": "2020"}
ENDS = {"month_name": "Feb 2023", "iso": "2023-02", "numeric": "02/2023", "year_only": "2023"}
SEPARATORS = (" \u2013 ", " - ", " to ")
ONGOING = ("Present", "present", "Current", "Now")


@pytest.mark.parametrize(
    ("fmt", "separator"), list(itertools.product(STARTS, SEPARATORS)), ids=lambda v: repr(v)
)
def test_every_format_and_separator(fmt, separator):
    closed = parse_date_range(f"{STARTS[fmt]}{separator}{ENDS[fmt]}", AS_OF)
    if fmt == "year_only":
        assert (closed.start, closed.end) == (YearMonth(2020, 1), YearMonth(2023, 12))
    else:
        assert (closed.start, closed.end) == (YearMonth(2020, 3), YearMonth(2023, 2))
    assert not closed.ongoing and not closed.clamped
    for word in ONGOING:
        current = parse_date_range(f"{STARTS[fmt]}{separator}{word}", AS_OF)
        assert current.end == NOW and current.ongoing and not current.clamped


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("January 2020 - March 2021", (2020, 1), (2021, 3)),
        ("Sept 2019 to Jun 2020", (2019, 9), (2020, 6)),
        ("Sep. 2019 \u2014 Dec. 2019", (2019, 9), (2019, 12)),
        ("1/2018 - 12/2018", (2018, 1), (2018, 12)),
        ("2016-2020", (2016, 1), (2020, 12)),
    ],
)
def test_other_spellings(text, start, end):
    parsed = parse_date_range(text, AS_OF)
    assert (parsed.start, parsed.end) == (YearMonth(*start), YearMonth(*end))


def test_year_only_conventions():
    parsed = parse_date_range("2020 - 2020", AS_OF)
    assert (parsed.start, parsed.end, parsed.months) == (
        YearMonth(2020, 1),
        YearMonth(2020, 12),
        12,
    )
    assert parse_date_range("2026 - Present", AS_OF).months == 11  # Jan..Nov 2026


def test_end_after_as_of_is_clamped():
    parsed = parse_date_range("Jan 2025 - Dec 2027", AS_OF)
    assert parsed.end == NOW and parsed.clamped and not parsed.ongoing
    assert parse_date_range("2025 - 2026", AS_OF).clamped  # year-only end means Dec 2026


def test_start_after_end_raises():
    with pytest.raises(DateRangeError, match="after"):
        parse_date_range("Mar 2023 - Jan 2023", AS_OF)
    with pytest.raises(DateRangeError, match="after"):
        parse_date_range("2027-01 - Present", AS_OF)  # start after as_of
    with pytest.raises(DateRangeError, match="not a date range"):
        parse_date_range("Acme | Jan 2020 - Present", AS_OF)


def test_months_are_inclusive():
    assert parse_date_range("Jan 2020 - Jan 2020", AS_OF).months == 1
    assert parse_date_range("Jan 2020 - Dec 2020", AS_OF).months == 12
    assert DateRange(YearMonth(2019, 11), YearMonth(2020, 2)).months == 4


def test_as_of_is_injected():
    early = parse_date_range("Jan 2020 - Present", date(2021, 6, 15))
    assert early.end == YearMonth(2021, 6) and early.months == 18


# --- Timeline -----------------------------------------------------------------------------


def test_merging_overlap_adjacent_and_gap():
    assert merge_intervals([(0, 11), (6, 17)]) == [(0, 17)]  # overlap
    assert merge_intervals([(0, 5), (6, 11)]) == [(0, 11)]  # adjacent months
    assert merge_intervals([(24, 35), (0, 11)]) == [(0, 11), (24, 35)]  # gap stays a gap
    assert merge_intervals([(0, 59), (12, 17)]) == [(0, 59)]  # contained
    assert merged_months([(0, 11), (24, 35)]) == 24
    assert merged_months([]) == 0


@pytest.mark.parametrize(
    ("title", "intern"),
    [
        ("Software Engineering Intern", True),
        ("Data Science Internship", True),
        ("intern, platform team", True),
        ("Summer Intern (ML)", True),
        ("Internal Tools Engineer", False),
        ("International Sales Manager", False),
        ("Internist", False),
        ("Senior Engineer", False),
    ],
)
def test_internship_titles(title, intern):
    assert is_internship_title(title) is intern


def test_meets_min_years_boundary():
    assert not meets_min_years(59, 5)
    assert meets_min_years(60, 5)
    assert meets_min_years(36, 3) and not meets_min_years(35, 3)
    assert months_to_years(59) == 4.9  # display rounding never fakes a threshold


@pytest.mark.parametrize(
    ("months", "years"),
    [(3, 0.3), (9, 0.8), (15, 1.3), (21, 1.8), (75, 6.3), (51, 4.3), (6, 0.5), (0, 0.0)],
)
def test_years_round_half_up(months, years):
    assert months_to_years(months) == years


def test_timeline_excludes_internships_and_merges():
    doc = (
        "A\nEXPERIENCE\n"
        "Engineer\nAcme | Jan 2020 - Dec 2021\nBuilt.\n"
        "Freelance Developer\nSelf-employed | Jul 2021 - Jun 2022\nBuilt more.\n"
        "Software Intern\nBeta | Jun 2019 - Aug 2019\nLearned.\n"
    )
    metadata = extract_metadata("resumes/a.txt", parse_resume(doc), AS_OF)
    assert [r.is_internship for r in metadata.roles] == [False, False, True]
    assert [r.months for r in metadata.roles] == [24, 12, 3]
    assert metadata.total_months == 30  # Jan 2020 .. Jun 2022 merged; internship excluded
    assert metadata.total_years == 2.5


def test_unparseable_role_dates_are_kept_but_not_counted():
    from rag.sections import Role

    roles = [
        Role("Engineer", "Acme", "Mar 2023 - Jan 2023", ()),
        Role("Dev", "B", "2020 - 2020", ()),
    ]
    timeline = build_timeline(roles, AS_OF)
    assert timeline.roles[0].dates is None and timeline.roles[0].months == 0
    assert timeline.total_months == 12


# --- Name and education -------------------------------------------------------------------


def test_name_skips_contact_lines():
    assert extract_name(("ann@example.com", "+44 555-0100", "Ann Lee", "Engineer")) == "Ann Lee"
    assert extract_name(("a@b.com | +1 555-0100 | Austin", "Ann Lee")) == "Ann Lee"
    assert extract_name((), "resumes/priya_sharma_2.pdf") == "Priya Sharma 2"


@pytest.mark.parametrize(
    ("degree", "level"),
    [
        ("PhD", DegreeLevel.PHD),
        ("Ph.D.", DegreeLevel.PHD),
        ("Doctor of Philosophy", DegreeLevel.PHD),
        ("M.Sc.", DegreeLevel.MASTERS),
        ("M.S.", DegreeLevel.MASTERS),
        ("M.A.", DegreeLevel.MASTERS),
        ("M.Eng.", DegreeLevel.MASTERS),
        ("M.Tech", DegreeLevel.MASTERS),
        ("MBA", DegreeLevel.MASTERS),
        ("MCA", DegreeLevel.MASTERS),
        ("Master of Science", DegreeLevel.MASTERS),
        ("B.Sc.", DegreeLevel.BACHELORS),
        ("B.Sc. (Hons)", DegreeLevel.BACHELORS),
        ("B.S.", DegreeLevel.BACHELORS),
        ("B.Eng.", DegreeLevel.BACHELORS),
        ("B.Tech", DegreeLevel.BACHELORS),
        ("B.A.", DegreeLevel.BACHELORS),
        ("BCA", DegreeLevel.BACHELORS),
        ("Bachelor of Arts", DegreeLevel.BACHELORS),
        ("Diploma", DegreeLevel.DIPLOMA),
        ("High School", DegreeLevel.SCHOOL),
        ("Marina Bay Polytechnic University", None),  # not "M.A."
        ("Baltic State Technical University", None),  # not "B.A."
        ("Masterclass in Pottery", None),
    ],
)
def test_degree_levels(degree, level):
    assert degree_level(degree) is level


def test_parse_education_entries():
    lines = (
        "M.Sc. in Artificial Intelligence",
        "Straits Institute of Technology, 2018 - 2020 | Grade: Distinction",
        "B.Eng. in Computer Engineering",
        "Jiangnan University, 2014 \u2013 2018",
    )
    section = Section(SectionKind.EDUCATION, "Education", lines, 0, 5)
    assert parse_education(section) == [
        Education(
            "M.Sc.",
            DegreeLevel.MASTERS,
            "Artificial Intelligence",
            "Straits Institute of Technology",
            2020,
        ),
        Education(
            "B.Eng.", DegreeLevel.BACHELORS, "Computer Engineering", "Jiangnan University", 2018
        ),
    ]


def test_highest_education_tie_break():
    bsc = Education("B.Sc.", DegreeLevel.BACHELORS, "CS", "U1", 2015)
    beng = Education("B.Eng.", DegreeLevel.BACHELORS, "EE", "U2", 2018)
    msc = Education("M.Sc.", DegreeLevel.MASTERS, "CS", "U3", 2016)
    diploma = Education("Diploma", DegreeLevel.DIPLOMA, "Web", "U4", 2024)
    assert highest_education([bsc, beng]) is beng  # tie -> latest end year
    assert highest_education([diploma, bsc, msc]) is msc
    assert highest_education([bsc, Education("B.A.", DegreeLevel.BACHELORS, "x", "y", 2015)]) is bsc
    assert highest_education([]) is None


# --- Corpus -------------------------------------------------------------------------------


def test_internship_titles_agree_with_employment_type():
    # The corpus check behind title-based internship detection.
    for persona in PERSONAS:
        for role in persona.experience:
            is_intern = role.employment_type is EmploymentType.INTERNSHIP
            assert is_internship_title(role.title) is is_intern, (persona.id, role.title)


@pytest.fixture(scope="module")
def manifest() -> dict[str, dict]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return {c["candidate_id"]: c for c in json.load(handle)["candidates"]}


@pytest.fixture(scope="module")
def metadata(manifest) -> dict:
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WORKSPACE_ROOT", str(ROOT / "data"))
        for persona in PERSONAS:
            path = manifest[persona.id]["resume_path"]
            result = fs_tools.read_file(path)
            assert result["success"], result["error"]
            out[persona.id] = extract_metadata(path, parse_resume(result["data"]["content"]), AS_OF)
    return out


@any_persona
def test_corpus_metadata_matches_ground_truth(persona, metadata, manifest):
    found = metadata[persona.id]
    truth = manifest[persona.id]
    assert found.candidate_id == persona.id and found.resume_path == truth["resume_path"]
    assert found.name == persona.name
    assert found.total_years == truth["total_years"]
    assert len(found.roles) == len(persona.experience)
    for record, role in zip(found.roles, persona.experience, strict=True):
        assert str(record.dates.start) == role.start
        assert str(record.dates.end) == (str(NOW) if role.end == PRESENT else role.end)
        assert record.dates.ongoing is (role.end == PRESENT)
        assert record.is_internship is (role.employment_type is EmploymentType.INTERNSHIP)
    best = found.highest_education
    expected = truth["highest_education"]
    got = (
        None
        if best is None
        else {
            "degree": best.degree,
            "field": best.field,
            "institution": best.institution,
            "end_year": best.end_year,
        }
    )
    assert got == expected
    assert len(found.education) == len(persona.education)


def test_corpus_edge_cases(metadata):
    assert metadata["mateo_garcia"].total_years == 4.3  # merged, not the 6.3 summed
    assert sum(r.months for r in metadata["mateo_garcia"].roles) == 76
    assert metadata["ishaan_gupta"].total_months == 0  # internships only
    assert metadata["ishaan_gupta"].total_years == 0.0
    olga = metadata["olga_petrova"]
    span = olga.roles[0].dates.end.index - olga.roles[-1].dates.start.index + 1
    assert olga.total_months == span - 25  # the Feb 2020 - Feb 2022 gap isn't counted
    daniel = metadata["daniel_osei"]
    assert daniel.total_years == 11.6  # teaching years count as experience
    assert any("Teacher" in r.title or "Maths" in r.title for r in daniel.roles)


def test_metadata_holds_no_contact_details(metadata):
    for persona in PERSONAS:
        text = repr(metadata[persona.id])
        assert persona.email not in text and persona.phone not in text
