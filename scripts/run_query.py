"""Ask a question against the persisted index and print a cited answer.

Examples:

    uv run python scripts/run_query.py "How did NVIDIA's data center revenue change?"
    uv run python scripts/run_query.py --top-k 8 "What risks did Apple emphasize?"
    uv run python scripts/run_query.py --no-filters "Compare R&D across all companies."
    uv run python scripts/run_query.py --agent "Compare MSFT and AAPL FY2024 risks"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_analyst.index import IndexConfig, load_index  # noqa: E402
from financial_analyst.retrieve import (  # noqa: E402
    build_retriever, infer_filters_from_question,
)
from financial_analyst.synthesize import answer_with_citations  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="Question to ask.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="Disable inferred metadata filters (use the whole corpus).",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use the AgentWorkflow (Lab 7.3) instead of the simple query engine.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    question = " ".join(args.question)

    index = load_index(IndexConfig())
    filters = None if args.no_filters else infer_filters_from_question(question).to_llama_filters()
    retriever = build_retriever(index, similarity_top_k=args.top_k, filters=filters)

    if args.agent:
        # Imported lazily — AgentWorkflow pulls in extra deps.
        from financial_analyst.workflow import run_agent_query
        result = run_agent_query(question, retriever=retriever)
    else:
        result = answer_with_citations(retriever, question)

    print("Q:", question)
    if filters is not None:
        inferred = infer_filters_from_question(question)
        print(
            f"   (filters: companies={inferred.companies or '-'}, "
            f"years={inferred.fiscal_years or '-'})"
        )
    print()
    print(result.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
