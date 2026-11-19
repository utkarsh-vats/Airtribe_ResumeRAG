"""Manual embedder smoke test (``python -m scripts.embedder_smoke --embedder huggingface``).

Not run by the tests or CI: with ``huggingface`` it downloads the model on first use, and
with ``gemini`` it calls the API using ``GEMINI_API_KEY`` / ``GEMINI_EMBEDDING_MODEL`` /
``GEMINI_EMBEDDING_DIM`` from the environment or ``.env`` (``--model`` and ``--dim``
override them). It prints the embedder id and dimension, each JD's token count against
``max_tokens`` (an offline estimate for Gemini), and a cosine-similarity matrix for three
short strings.
"""

import argparse
import os
import time

from dotenv import load_dotenv

from rag.embedders import available_embedders, from_env, get_embedder
from scripts.persona_schema import ROOT

JD_DIR = ROOT / "data" / "job_descriptions"
SAMPLES = (
    "Senior Python developer building Django and FastAPI services on PostgreSQL",
    "Backend engineer: Python APIs with Postgres and Docker",
    "HVAC design engineer modelling airflow for hospital ventilation",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedder", default="huggingface", choices=available_embedders())
    parser.add_argument("--model", help="model id (huggingface or gemini)")
    parser.add_argument("--device", help="cuda, mps or cpu (huggingface only; default auto)")
    parser.add_argument("--dim", type=int, help="output dimensions (gemini only)")
    args = parser.parse_args(argv)
    if args.embedder == "gemini":
        load_dotenv(ROOT / ".env")
        env = dict(os.environ, EMBEDDER="gemini")
        if args.model:
            env["GEMINI_EMBEDDING_MODEL"] = args.model
        if args.dim:
            env["GEMINI_EMBEDDING_DIM"] = str(args.dim)
        embedder = from_env(env)
    else:
        params = {}
        if args.embedder == "huggingface":
            params = {k: v for k, v in (("model", args.model), ("device", args.device)) if v}
        embedder = get_embedder(args.embedder, **params)

    started = time.perf_counter()
    vectors = embedder.embed_documents(list(SAMPLES))
    elapsed = time.perf_counter() - started
    print(f"embedder_id: {embedder.embedder_id}")
    print(f"model:       {embedder.model}")
    print(f"dim:         {embedder.dim} (vectors: {len(vectors[0])})")
    print(f"max_tokens:  {embedder.max_tokens}")
    print(f"first embed: {elapsed:.2f}s (includes model load)")

    print("\nJD token counts:")
    for path in sorted(JD_DIR.glob("*.txt")):
        count = embedder.token_count(path.read_text(encoding="utf-8"))
        flag = "  OVER max_tokens" if count > embedder.max_tokens else ""
        print(f"  {path.name:<22} {count:>5} / {embedder.max_tokens}{flag}")

    print("\nCosine similarity (documents):")
    for i, a in enumerate(vectors):
        row = "  ".join(f"{sum(x * y for x, y in zip(a, b, strict=True)):.3f}" for b in vectors)
        print(f"  [{i}] {row}   {SAMPLES[i][:50]}")
    query = embedder.embed_query("Python backend developer")
    sims = [sum(x * y for x, y in zip(query, v, strict=True)) for v in vectors]
    print("\nQuery 'Python backend developer' vs documents: " + ", ".join(f"{s:.3f}" for s in sims))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
