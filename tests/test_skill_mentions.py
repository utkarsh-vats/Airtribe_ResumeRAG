import pytest
from skill_mentions import find_mentions, mentioned_skills


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Wrote services in C++ and C#.", {"C++", "C#"}),
        ("Ported ASP.NET apps to .NET 8.", {".NET"}),
        ("Built a Node.js API.", {"Node.js"}),
        ("Java 17, not JavaScript.", {"Java", "JavaScript"}),
        ("JavaScript only", {"JavaScript"}),
        ("Shipped React Native apps", {"React Native"}),
        ("Wrote AWS Lambda handlers", {"AWS Lambda"}),
        ("Tuned Microsoft SQL Server and Postgres", {"SQL Server", "PostgreSQL"}),
        ("Deployed on k8s with Helm charts", {"Kubernetes", "Helm"}),
        ("python, PYTHON and Python", {"Python"}),
        ("Styled pages with Tailwind CSS", {"Tailwind CSS"}),
        ("Ran 40 A/B tests and wrote CI/CD pipelines", {"A/B Testing", "CI/CD"}),
        ("Migrated the monolith to microservices", {"Microservices"}),
    ],
)
def test_non_ambiguous_forms_and_boundaries(text, expected):
    assert mentioned_skills(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Batch jobs on Spark", {"Spark"}),
        ("a spark of curiosity", set()),
        ("ran apache spark jobs", {"Spark"}),
        ("Modelled airflow through ducts", set()),
        ("Scheduled DAGs in Airflow", {"Airflow"}),
        ("Services written in Go", {"Go"}),
        ("go to market; golang services", {"Go"}),
        ("Quick to react to incidents", set()),
        ("Built UIs in React", {"React"}),
        ("built uis in reactjs", {"React"}),
        ("Rest assured, the backlog shrank", set()),
        ("API tests in REST Assured", {"REST Assured"}),
        ("excel at stakeholder management", set()),
        ("Modelled cash flow in Excel", {"Excel"}),
        ("a tailwind for growth", set()),
        ("the salesforce grew to 40 reps", set()),
        ("delivered with a swift turnaround", set()),
    ],
)
def test_ambiguous_canonical_exact_case_aliases_anywhere(text, expected):
    assert mentioned_skills(text) == expected


def test_longest_match_reports_form_and_span():
    [mention] = find_mentions("Shipped on Amazon Web Services.")
    assert mention.skill == "AWS"
    assert mention.text == "Amazon Web Services"
    assert mention.via_alias
    assert (mention.start, mention.end) == (11, 30)


def test_mentions_are_in_text_order():
    assert [m.skill for m in find_mentions("SQL then Python then Docker")] == [
        "SQL",
        "Python",
        "Docker",
    ]
