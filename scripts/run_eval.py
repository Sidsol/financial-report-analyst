"""Run the retrieval evaluation and write JSON + markdown + failure log.

Usage:

    uv run python scripts/run_eval.py
    uv run python scripts/run_eval.py --top-k 10
    uv run python scripts/run_eval.py --no-filters
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_analyst.evaluate import (  # noqa: E402
    evaluate, write_eval_json, write_eval_markdown, write_failure_log,
)
from financial_analyst.index import IndexConfig, load_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-filters", action="store_true",
                        help="Disable inferred metadata filters.")
    parser.add_argument("--config-name", type=str, default="baseline",
                        help="Tag used in output filenames.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    index = load_index(IndexConfig())
    summary = evaluate(
        index,
        top_k=args.top_k,
        use_filters=not args.no_filters,
        config_name=args.config_name,
    )

    json_path = write_eval_json(summary)
    md_path = write_eval_markdown(summary)
    fail_path = write_failure_log(summary)

    print(f"Eval `{summary.config_name}` complete:")
    print(f"  Questions:                 {summary.n_questions}")
    print(f"  Metadata Hit@{summary.top_k:<3} {summary.metadata_hit_rate:.3f}")
    print(f"  Strict   Hit@{summary.top_k:<3} {summary.strict_hit_rate:.3f}")
    print(f"  Metadata MRR              {summary.metadata_mrr:.3f}")
    print(f"  Strict   MRR              {summary.strict_mrr:.3f}")
    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {fail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
