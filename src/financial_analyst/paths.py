"""Project paths used by every other module."""

from __future__ import annotations

from pathlib import Path

# Repository root = three levels above this file
# (src/financial_analyst/paths.py -> src/financial_analyst -> src -> root).
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
EVAL_DIR: Path = DATA_DIR / "eval"
MANIFEST_PATH: Path = DATA_DIR / "filings_manifest.yaml"
QUESTIONS_PATH: Path = EVAL_DIR / "questions.yaml"

STORAGE_DIR: Path = ROOT_DIR / "storage"
RESULTS_DIR: Path = ROOT_DIR / "results"
REPORTS_DIR: Path = ROOT_DIR / "reports"


def ensure_dirs() -> None:
    """Create artifact directories that are gitignored but expected to exist."""
    for d in (RAW_DIR, STORAGE_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
