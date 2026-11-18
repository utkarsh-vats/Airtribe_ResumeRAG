"""Skill taxonomy: canonical skills with aliases and implications, plus lookup helpers.

This module is data plus exact-name lookups only. Finding skills inside resume or JD text
(word boundaries, ambiguous names, section context) is the metadata extractor's job.

- ``aliases`` are alternative spellings that mean the same skill (``k8s`` -> Kubernetes).
  Lookups are case-insensitive, so case variants are not listed.
- ``implies`` lists skills you necessarily use when you use this one: a framework implies its
  language (Django -> Python), a specific RDBMS implies SQL. Implications are transitive and
  must stay acyclic.
- ``ambiguous`` marks canonical names that are also common words or very short (Go, R, C,
  Excel, Spark). The extractor only matches these inside a skills section. Aliases are chosen
  to be unambiguous (``Golang``, ``Apache Spark``), so they can match anywhere.

Deliberately missing aliases: ``CV`` for Computer Vision (it also means "resume"), ``Node`` for
Node.js and ``Lambda`` for AWS Lambda (too generic).
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
    Skill("TypeScript", _L, ("TS",), ("JavaScript",)),
    Skill("Go", _L, ("Golang",), ambiguous=True),
    Skill("C", _L, ambiguous=True),
    Skill("C++", _L, ("CPP", "cplusplus")),
    Skill("C#", _L, ("CSharp", "C Sharp")),
    Skill("Kotlin", _L),
    Skill("Scala", _L),
    Skill("Rust", _L, ("Rust lang",), ambiguous=True),
    Skill("PHP", _L),
    Skill("Swift", _L, ("Swift 5",), ambiguous=True),
    Skill("R", _L, ("R language", "RStudio"), ambiguous=True),
    Skill("SQL", _L),
    Skill("Bash", _L, ("Shell scripting", "Bash scripting")),
    Skill("HTML", _L, ("HTML5",)),
    Skill("CSS", _L, ("CSS3",)),
    # Backend frameworks, APIs and messaging
    Skill("Django", _B, (), ("Python",)),
    Skill("Django REST Framework", _B, ("DRF",), ("Django",)),
    Skill("FastAPI", _B, (), ("Python",)),
    Skill("Flask", _B, (), ("Python",), ambiguous=True),
    Skill("Celery", _B, (), ("Python",)),
    Skill("Spring Boot", _B, ("SpringBoot",), ("Java",)),
    Skill("Spring Cloud", _B, (), ("Spring Boot",)),
    Skill("Hibernate", _B, (), ("Java",)),
    Skill("Node.js", _B, ("NodeJS", "Node JS"), ("JavaScript",)),
    Skill("Express.js", _B, ("ExpressJS", "Express JS"), ("Node.js",)),
    Skill("Laravel", _B, (), ("PHP",)),
    Skill(".NET", _B, ("dotnet", ".NET Core", "ASP.NET", "ASP.NET Core"), ("C#",)),
    Skill("GraphQL", _B),
    Skill("REST APIs", _P, ("REST API", "RESTful APIs", "RESTful API", "RESTful")),
    Skill("Microservices", _P, ("microservice", "microservice architecture")),
    Skill("Kafka", _B, ("Apache Kafka",)),
    Skill("RabbitMQ", _B),
    # Frontend
    Skill("React", _F, ("React.js", "ReactJS"), ("JavaScript",)),
    Skill("Next.js", _F, ("NextJS",), ("React",)),
    Skill("Vue.js", _F, ("Vue", "VueJS"), ("JavaScript",)),
    Skill("Angular", _F, ("AngularJS",), ("JavaScript",)),
    Skill("Redux", _F, ("Redux Toolkit",), ("JavaScript",)),
    Skill("Tailwind CSS", _F, ("Tailwind", "TailwindCSS"), ("CSS",)),
    Skill("Storybook", _F),
    # Mobile
    Skill("Android", _M, ("Android SDK",)),
    Skill("iOS", _M),
    Skill("React Native", _M, (), ("JavaScript",)),
    # Data engineering and analytics
    Skill("pandas", _D, (), ("Python",)),
    Skill("NumPy", _D, (), ("Python",)),
    Skill("Spark", _D, ("Apache Spark",), ambiguous=True),
    Skill("PySpark", _D, (), ("Spark", "Python")),
    Skill("Airflow", _D, ("Apache Airflow",), ("Python",), ambiguous=True),
    Skill("dbt", _D, ("data build tool",), ("SQL",)),
    Skill("Snowflake", _D, (), ("SQL",), ambiguous=True),
    Skill("BigQuery", _D, ("Google BigQuery",), ("SQL", "GCP")),
    Skill("Tableau", _D),
    Skill("Power BI", _D, ("PowerBI",)),
    Skill("Looker", _D),
    Skill("Jupyter", _D, ("Jupyter Notebook", "JupyterLab")),
    Skill("tidyverse", _D, (), ("R",)),
    Skill("ggplot2", _D, (), ("R",)),
    Skill("dplyr", _D, (), ("R",)),
    Skill("A/B Testing", _P, ("A/B tests", "A/B test", "AB testing", "split testing")),
    # Machine learning
    Skill("Machine Learning", _ML, ("ML",)),
    Skill("Deep Learning", _ML, (), ("Machine Learning",)),
    Skill("Natural Language Processing", _ML, ("NLP",), ("Machine Learning",)),
    Skill("Computer Vision", _ML, (), ("Machine Learning",)),
    Skill("Large Language Models", _ML, ("LLM", "LLMs"), ("Natural Language Processing",)),
    Skill("PyTorch", _ML, (), ("Python", "Deep Learning")),
    Skill("TensorFlow", _ML, (), ("Deep Learning",)),
    Skill("scikit-learn", _ML, ("sklearn", "scikit learn"), ("Python", "Machine Learning")),
    Skill("XGBoost", _ML, (), ("Machine Learning",)),
    Skill("statsmodels", _ML, (), ("Python",)),
    Skill("Hugging Face", _ML, ("HuggingFace", "Hugging Face Transformers"), ("Deep Learning",)),
    Skill("MLflow", _ML, (), ("Machine Learning",)),
    Skill("Kubeflow", _ML, (), ("Kubernetes", "Machine Learning")),
    Skill("ONNX", _ML, (), ("Machine Learning",)),
    Skill("CUDA", _ML),
    # Databases
    Skill("PostgreSQL", _DB, ("Postgres", "psql"), ("SQL",)),
    Skill("MySQL", _DB, (), ("SQL",)),
    Skill("SQL Server", _DB, ("MSSQL", "Microsoft SQL Server"), ("SQL",)),
    Skill("MongoDB", _DB, ("Mongo",)),
    Skill("Redis", _DB),
    Skill("DynamoDB", _DB, ("Amazon DynamoDB",), ("AWS",)),
    Skill("Elasticsearch", _DB, ("Elastic Search",)),
    # Cloud
    Skill("AWS", _C, ("Amazon Web Services",)),
    Skill("GCP", _C, ("Google Cloud", "Google Cloud Platform")),
    Skill("Azure", _C, ("Microsoft Azure",)),
    Skill("AWS Lambda", _C, (), ("AWS",)),
    Skill("EKS", _C, ("Amazon EKS", "AWS EKS"), ("AWS", "Kubernetes")),
    # DevOps, infrastructure and tooling
    Skill("Docker", _O),
    Skill("Kubernetes", _O, ("k8s",)),
    Skill("Helm", _O, (), ("Kubernetes",), ambiguous=True),
    Skill("ArgoCD", _O, ("Argo CD",), ("Kubernetes",)),
    Skill("Terraform", _O),
    Skill("Ansible", _O),
    Skill("Jenkins", _O),
    Skill("GitHub Actions", _O),
    Skill("CI/CD", _P, ("CI CD", "continuous integration")),
    Skill("Prometheus", _O),
    Skill("Grafana", _O),
    Skill("Linux", _O),
    Skill("Nginx", _O),
    Skill("Git", _O),
    Skill("Maven", _O, ("Apache Maven",), ("Java",)),
    Skill("Gradle", _O),
    # Testing
    Skill("pytest", _T, (), ("Python",)),
    Skill("JUnit", _T, (), ("Java",)),
    Skill("Selenium", _T),
    Skill("Playwright", _T),
    Skill("Cypress", _T),
    Skill("Jest", _T, (), ("JavaScript",), ambiguous=True),
    Skill("Postman", _T),
    Skill("REST Assured", _T, ("RestAssured",), ("Java",)),
    # Business / non-tech tools
    Skill("Excel", _X, ("Microsoft Excel", "MS Excel"), ambiguous=True),
    Skill("Salesforce", _X),
    Skill("HubSpot", _X),
    Skill("SEO", _X, ("Search Engine Optimization", "Search Engine Optimisation")),
    Skill("Google Analytics", _X),
    Skill("QuickBooks", _X),
    Skill("Figma", _X),
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
