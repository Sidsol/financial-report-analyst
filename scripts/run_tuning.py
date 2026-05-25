"""Run the Lab 7.3 tuning sweep and write the comparison artifacts.

Usage:

    uv run python scripts/run_tuning.py            # full sweep + reranker
    uv run python scripts/run_tuning.py --quick    # skip reranker
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_analyst.tune import (  # noqa: E402
    run_full_sweep_with_rerank, run_sweep, write_tuning_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip the reranker variants (faster).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    rows = run_sweep() if args.quick else run_full_sweep_with_rerank()
    json_path, md_path = write_tuning_outputs(rows)
    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
