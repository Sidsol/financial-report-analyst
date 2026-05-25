"""Download SEC 10-K filings listed in ``data/filings_manifest.yaml``.

Idempotent: skips files already present in ``data/raw/``.

Run from the project root:

    uv run python scripts/download_reports.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_analyst.manifest import Filing, load_manifest  # noqa: E402
from financial_analyst.paths import RAW_DIR  # noqa: E402


# SEC asks clients to keep their average request rate below 10 req/s.
# We target 8 req/s with a small floor sleep to stay safely under the cap
# even under retry/backoff bursts.
MIN_SLEEP_SECONDS = 0.15


def _require_user_agent() -> str:
    """Return SEC_USER_AGENT or exit with a clear error."""
    load_dotenv(ROOT / ".env", override=False)
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "your-email" in ua or "example.com" in ua:
        sys.stderr.write(
            "ERROR: SEC_USER_AGENT must be set to a real contact string.\n"
            "       SEC EDGAR requires this on every request.\n"
            "       Edit .env and set e.g.:\n"
            '         SEC_USER_AGENT="Jane Doe (jane@school.edu)"\n'
        )
        raise SystemExit(2)
    return ua


def _download_one(session: requests.Session, filing: Filing, force: bool = False) -> str:
    """Download a single filing if not already cached. Returns status string."""
    dest = filing.local_path
    if dest.exists() and not force:
        return f"cached  {dest.relative_to(ROOT)} ({dest.stat().st_size:,} bytes)"

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = filing.source_url

    for attempt in range(1, 5):
        try:
            resp = session.get(url, timeout=60)
        except requests.RequestException as exc:
            wait = 2 ** attempt
            sys.stderr.write(f"  network error on {url}: {exc}; retrying in {wait}s...\n")
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            wait = 2 ** attempt
            sys.stderr.write(f"  429 from SEC on {url}; backing off {wait}s...\n")
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            raise RuntimeError(
                f"SEC returned HTTP {resp.status_code} for {url}: {resp.text[:200]}"
            )

        dest.write_bytes(resp.content)
        return f"fetched {dest.relative_to(ROOT)} ({len(resp.content):,} bytes)"

    raise RuntimeError(f"Failed to fetch {url} after retries.")


def main() -> int:
    ua = _require_user_agent()
    filings = load_manifest()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }
    )

    print(f"Downloading {len(filings)} filings into {RAW_DIR.relative_to(ROOT)}/ ...")
    for filing in filings:
        print(f"  {filing.company:<10} FY{filing.fiscal_year}  {filing.accession}")
        status = _download_one(session, filing)
        print(f"    -> {status}")
        time.sleep(MIN_SLEEP_SECONDS)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
