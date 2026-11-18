"""Skill taxonomy: canonical skills with aliases and implications, plus lookup helpers.

This module is data plus exact-name lookups only. Finding skills inside resume or JD text
(word boundaries, ambiguous names, section context) is the metadata extractor's job.

- ``aliases`` are alternative spellings that mean the same skill (``k8s`` -> Kubernetes).
  Lookups are case-insensitive, so case variants are not listed.
- ``implies`` lists skills you necessarily use when you use this one: a framework implies its
  language (Django -> Python), a specific RDBMS implies SQL. Implications are transitive and
  must stay acyclic.
- ``ambiguous`` marks canonical names that are also ordinary English words or very short
  (Go, R, C, React, Spark, Excel, "rest assured"). Their canonical name only counts in its exact
  case. Aliases are always safe (``Golang``, ``Apache Spark``, ``ReactJS``) and match in any case.
  The full mention contract is in ``data/ground_truth/README.md``.
- ``first_release_year`` is the year of the first public release, set for frameworks, tools,
  platforms and newer languages (not long-established languages). The persona tests use it so
  no role uses a skill before it existed. ``Angular`` (2016) and ``AngularJS`` (2010) are
  separate skills for that reason.

Deliberately missing aliases: ``CV`` for Computer Vision (it also means "resume"), ``Node`` for
Node.js, ``Lambda`` for AWS Lambda, ``Express`` for Express.js and ``Tailwind`` for Tailwind CSS
(all ordinary words or too generic).
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class Category(StrEnum):
    """Broad grouping of a skill, used for reporting and persona design."""

    LANGUAGE = "language"
    BACKEND = "backend"
    FRONTEND = "frontend"
    MOBILE = "mobile"
    DATA = "data"
    ML = "ml"
    DATABASE = "database"
    CLOUD = "cloud"
    DEVOPS = "devops"
    TESTING = "testing"
    PRACTICE = "practice"
    BUSINESS = "business"


@dataclass(frozen=True)
class Skill:
    """One canonical skill in the taxonomy."""

    name: str
    category: Category
    aliases: tuple[str, ...] = ()
    implies: tuple[str, ...] = ()
    ambiguous: bool = False
    first_release_year: int | None = None


_L = Category.LANGUAGE
_B = Category.BACKEND
_F = Category.FRONTEND
_M = Category.MOBILE
_D = Category.DATA
_ML = Category.ML
_DB = Category.DATABASE
_C = Category.CLOUD
_O = Category.DEVOPS
_T = Category.TESTING
_P = Category.PRACTICE
_X = Category.BUSINESS

SKILLS: tuple[Skill, ...] = (
    # Languages
    Skill("Python", _L, ("Python3", "Python 3")),
    Skill("Java", _L),
    Skill("JavaScript", _L, ("JS", "ECMAScript", "ES6")),
    Skill("TypeScript", _L, ("TS",), ("JavaScript",), first_release_year=2012),
    Skill("Go", _L, ("Golang",), ambiguous=True, first_release_year=2009),
    Skill("C", _L, ambiguous=True),
    Skill("C++", _L, ("CPP", "cplusplus")),
    Skill("C#", _L, ("CSharp", "C Sharp")),
    Skill("Kotlin", _L, first_release_year=2011),
    Skill("Scala", _L),
    Skill("Rust", _L, ("Rust lang",), ambiguous=True, first_release_year=2010),
    Skill("PHP", _L),
    Skill("Swift", _L, ("Swift 5",), ambiguous=True, first_release_year=2014),
    Skill("R", _L, ("R language", "RStudio"), ambiguous=True),
    Skill("SQL", _L),
    Skill("Bash", _L, ("Shell scripting", "Bash scripting")),
    Skill("HTML", _L, ("HTML5",)),
    Skill("CSS", _L, ("CSS3",)),
    # Backend frameworks, APIs and messaging
    Skill("Django", _B, (), ("Python",), first_release_year=2005),
    Skill("Django REST Framework", _B, ("DRF",), ("Django",), first_release_year=2011),
    Skill("FastAPI", _B, (), ("Python",), first_release_year=2018),
    Skill("Flask", _B, (), ("Python",), ambiguous=True, first_release_year=2010),
    Skill("Celery", _B, (), ("Python",), ambiguous=True, first_release_year=2009),
    Skill("Spring Boot", _B, ("SpringBoot",), ("Java",), first_release_year=2014),
    Skill("Spring Cloud", _B, (), ("Spring Boot",), first_release_year=2015),
    Skill("Hibernate", _B, (), ("Java",), first_release_year=2001),
    Skill("Node.js", _B, ("NodeJS", "Node JS"), ("JavaScript",), first_release_year=2009),
    Skill("Express.js", _B, ("ExpressJS", "Express JS"), ("Node.js",), first_release_year=2010),
    Skill("Laravel", _B, (), ("PHP",), first_release_year=2011),
    Skill(
        ".NET",
        _B,
        ("dotnet", ".NET Core", "ASP.NET", "ASP.NET Core"),
        ("C#",),
        first_release_year=2002,
    ),
    Skill("GraphQL", _B, first_release_year=2015),
    Skill("REST APIs", _P, ("REST API", "RESTful APIs", "RESTful API", "RESTful")),
    Skill("Microservices", _P, ("microservice", "microservice architecture")),
    Skill("Kafka", _B, ("Apache Kafka",), first_release_year=2011),
    Skill("RabbitMQ", _B, first_release_year=2007),
    # Frontend
    Skill(
        "React",
        _F,
        ("React.js", "ReactJS"),
        ("JavaScript",),
        ambiguous=True,
        first_release_year=2013,
    ),
    Skill("Next.js", _F, ("NextJS",), ("React",), first_release_year=2016),
    Skill("Vue.js", _F, ("Vue", "VueJS"), ("JavaScript",), first_release_year=2014),
    Skill("Angular", _F, ("Angular 2+",), ("TypeScript",), ambiguous=True, first_release_year=2016),
    Skill("AngularJS", _F, ("Angular.js",), ("JavaScript",), first_release_year=2010),
    Skill("Redux", _F, ("Redux Toolkit",), ("JavaScript",), first_release_year=2015),
    Skill("Tailwind CSS", _F, ("TailwindCSS",), ("CSS",), first_release_year=2017),
    Skill("Storybook", _F, ("Storybook.js",), ambiguous=True, first_release_year=2016),
    # Mobile
    Skill("Android", _M, ("Android SDK",), first_release_year=2008),
    Skill("iOS", _M, first_release_year=2007),
    Skill("React Native", _M, (), ("JavaScript",), first_release_year=2015),
    # Data engineering and analytics
    Skill("pandas", _D, (), ("Python",), first_release_year=2008),
    Skill("NumPy", _D, (), ("Python",), first_release_year=2006),
    Skill("Spark", _D, ("Apache Spark",), ambiguous=True, first_release_year=2010),
    Skill("PySpark", _D, (), ("Spark", "Python"), first_release_year=2013),
    Skill("Airflow", _D, ("Apache Airflow",), ("Python",), ambiguous=True, first_release_year=2015),
    Skill(
        "dbt",
        _D,
        ("data build tool", "dbt Core"),
        ("SQL",),
        ambiguous=True,
        first_release_year=2016,
    ),
    Skill("Snowflake", _D, (), ("SQL",), ambiguous=True, first_release_year=2014),
    Skill("BigQuery", _D, ("Google BigQuery",), ("SQL", "GCP"), first_release_year=2011),
    Skill("Tableau", _D),
    Skill("Power BI", _D, ("PowerBI",), first_release_year=2015),
    Skill("Looker", _D, first_release_year=2012),
    Skill("Jupyter", _D, ("Jupyter Notebook", "JupyterLab"), first_release_year=2011),
    Skill("tidyverse", _D, (), ("R",), first_release_year=2016),
    Skill("ggplot2", _D, (), ("R",), first_release_year=2007),
    Skill("dplyr", _D, (), ("R",), first_release_year=2014),
    Skill("A/B Testing", _P, ("A/B tests", "A/B test", "AB testing", "split testing")),
    # Machine learning
    Skill("Machine Learning", _ML, ("ML",)),
    Skill("Deep Learning", _ML, (), ("Machine Learning",)),
    Skill("Natural Language Processing", _ML, ("NLP",), ("Machine Learning",)),
    Skill("Computer Vision", _ML, (), ("Machine Learning",)),
    Skill("Large Language Models", _ML, ("LLM", "LLMs"), ("Natural Language Processing",)),
    Skill("PyTorch", _ML, (), ("Python", "Deep Learning"), first_release_year=2016),
    Skill("TensorFlow", _ML, (), ("Deep Learning",), first_release_year=2015),
    Skill(
        "scikit-learn",
        _ML,
        ("sklearn", "scikit learn"),
        ("Python", "Machine Learning"),
        first_release_year=2010,
    ),
    Skill("XGBoost", _ML, (), ("Machine Learning",), first_release_year=2014),
    Skill("statsmodels", _ML, (), ("Python",)),
    Skill(
        "Hugging Face",
        _ML,
        ("HuggingFace", "Hugging Face Transformers"),
        ("Deep Learning",),
        first_release_year=2018,
    ),
    Skill("MLflow", _ML, (), ("Machine Learning",), first_release_year=2018),
    Skill("Kubeflow", _ML, (), ("Kubernetes", "Machine Learning"), first_release_year=2018),
    Skill("ONNX", _ML, (), ("Machine Learning",), first_release_year=2017),
    Skill("CUDA", _ML, first_release_year=2007),
    # Databases
    Skill("PostgreSQL", _DB, ("Postgres", "psql"), ("SQL",)),
    Skill("MySQL", _DB, (), ("SQL",)),
    Skill("SQL Server", _DB, ("MSSQL", "Microsoft SQL Server"), ("SQL",)),
    Skill("MongoDB", _DB, ("Mongo",), first_release_year=2009),
    Skill("Redis", _DB, first_release_year=2009),
    Skill("DynamoDB", _DB, ("Amazon DynamoDB",), ("AWS",), first_release_year=2012),
    Skill("Elasticsearch", _DB, ("Elastic Search",), first_release_year=2010),
    # Cloud
    Skill("AWS", _C, ("Amazon Web Services",), first_release_year=2006),
    Skill("GCP", _C, ("Google Cloud", "Google Cloud Platform"), first_release_year=2008),
    Skill("Azure", _C, ("Microsoft Azure",), first_release_year=2010),
    Skill("AWS Lambda", _C, (), ("AWS",), first_release_year=2014),
    Skill("EKS", _C, ("Amazon EKS", "AWS EKS"), ("AWS", "Kubernetes"), first_release_year=2018),
    # DevOps, infrastructure and tooling
    Skill("Docker", _O, first_release_year=2013),
    Skill("Kubernetes", _O, ("k8s",), first_release_year=2014),
    Skill(
        "Helm",
        _O,
        ("Helm charts", "Helm chart"),
        ("Kubernetes",),
        ambiguous=True,
        first_release_year=2016,
    ),
    Skill("ArgoCD", _O, ("Argo CD",), ("Kubernetes",), first_release_year=2018),
    Skill("Terraform", _O, first_release_year=2014),
    Skill("Ansible", _O, first_release_year=2012),
    Skill("Jenkins", _O, first_release_year=2011),
    Skill("GitHub Actions", _O, first_release_year=2018),
    Skill("CI/CD", _P, ("CI CD", "continuous integration")),
    Skill("Prometheus", _O, first_release_year=2012),
    Skill("Grafana", _O, first_release_year=2014),
    Skill("Linux", _O),
    Skill("Nginx", _O),
    Skill("Git", _O),
    Skill("Maven", _O, ("Apache Maven",), ("Java",), ambiguous=True),
    Skill("Gradle", _O, first_release_year=2008),
    # Testing
    Skill("pytest", _T, (), ("Python",)),
    Skill("JUnit", _T, (), ("Java",)),
    Skill("Selenium", _T, ("Selenium WebDriver",), ambiguous=True),
    Skill("Playwright", _T, ("Playwright Test",), ambiguous=True, first_release_year=2020),
    Skill("Cypress", _T, ("Cypress.io",), ambiguous=True, first_release_year=2015),
    Skill("Jest", _T, (), ("JavaScript",), ambiguous=True, first_release_year=2014),
    Skill("Postman", _T, ("Postman API Platform",), ambiguous=True, first_release_year=2012),
    Skill("REST Assured", _T, ("RestAssured",), ("Java",), ambiguous=True, first_release_year=2010),
    # Business / non-tech tools
    Skill("Excel", _X, ("Microsoft Excel", "MS Excel"), ambiguous=True),
    Skill("Salesforce", _X, ("Salesforce CRM",), ambiguous=True),
    Skill("HubSpot", _X, first_release_year=2006),
    Skill("SEO", _X, ("Search Engine Optimization", "Search Engine Optimisation")),
    Skill("Google Analytics", _X),
    Skill("QuickBooks", _X),
    Skill("Figma", _X, first_release_year=2016),
    Skill("Jira", _X),
)

SKILLS_BY_NAME: dict[str, Skill] = {skill.name: skill for skill in SKILLS}


def _key(name: str) -> str:
    """Lookup key: case-folded with runs of whitespace collapsed to one space."""
    return " ".join(name.split()).casefold()


def _build_lookup(skills: Iterable[Skill]) -> dict[str, str]:
    """Map every canonical name and alias key to its canonical name; reject collisions."""
    lookup: dict[str, str] = {}
    for skill in skills:
        for form in (skill.name, *skill.aliases):
            key = _key(form)
            existing = lookup.setdefault(key, skill.name)
            if existing != skill.name:
                raise ValueError(f"{form!r} maps to both {existing!r} and {skill.name!r}")
    return lookup


_LOOKUP = _build_lookup(SKILLS)


def canonicalize(name: str) -> str | None:
    """Return the canonical skill name for a name or alias (case-insensitive), else None."""
    return _LOOKUP.get(_key(name))


def get_skill(name: str) -> Skill:
    """Return the ``Skill`` for a canonical name or alias. Raises ``KeyError`` if unknown."""
    canonical = canonicalize(name)
    if canonical is None:
        raise KeyError(f"Unknown skill: {name!r}")
    return SKILLS_BY_NAME[canonical]


def expand_implied(skills: Iterable[str]) -> frozenset[str]:
    """Canonicalize ``skills`` and add everything they imply, transitively.

    Raises ``KeyError`` for an unknown skill name.
    """
    result: set[str] = set()
    pending = [get_skill(name).name for name in skills]
    while pending:
        name = pending.pop()
        if name not in result:
            result.add(name)
            pending.extend(SKILLS_BY_NAME[name].implies)
    return frozenset(result)
