"""Build (or rebuild) the persisted vector index.

Run from the project root:

    uv run python scripts/build_index.py             # default baseline
    uv run python scripts/build_index.py --force     # rebuild from scratch
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_analyst.index import IndexConfig, verify_reload  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the index.")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    cfg = IndexConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"Building index at: {cfg.storage_dir()}")
    if args.force:
        import shutil
        if cfg.storage_dir().exists():
            shutil.rmtree(cfg.storage_dir())
    verify_reload(cfg)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
