"""Filings manifest loader.

The manifest at ``data/filings_manifest.yaml`` is the single source of truth
for which SEC filings make up the RAG corpus. Every entry contains enough
information to download the original filing deterministically.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

from .paths import MANIFEST_PATH, RAW_DIR


@dataclass(frozen=True)
class Filing:
    """One row of ``filings_manifest.yaml``."""

    company: str
    ticker: str
    cik: str
    fiscal_year: int
    filing_date: str
    period_of_report: str
    form: str
    accession: str
    primary_doc: str

    @property
    def cik_no_zero(self) -> str:
        return self.cik.lstrip("0") or "0"

    @property
    def accession_compact(self) -> str:
        """SEC archive URLs use the accession number without dashes."""
        return self.accession.replace("-", "")

    @property
    def source_url(self) -> str:
        """Canonical URL for the primary filing document on sec.gov."""
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{self.cik_no_zero}/{self.accession_compact}/{self.primary_doc}"
        )

    @property
    def local_path(self) -> Path:
        """Where to cache the downloaded HTML."""
        return RAW_DIR / self.company / f"FY{self.fiscal_year}_{self.form.replace('-', '')}.htm"


def load_manifest(path: Path | None = None) -> list[Filing]:
    """Load and validate the filings manifest."""
    path = path or MANIFEST_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "filings" not in raw:
        raise ValueError(f"Manifest {path} must have a top-level `filings:` list.")
    return [Filing(**entry) for entry in raw["filings"]]


def manifest_hash(filings: list[Filing] | None = None) -> str:
    """Stable hash of the manifest contents — used to key caches."""
    filings = filings or load_manifest()
    payload = json.dumps([asdict(f) for f in filings], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]
