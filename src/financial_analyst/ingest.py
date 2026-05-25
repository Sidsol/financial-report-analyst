"""Ingest SEC 10-K HTML filings into LlamaIndex ``Document`` objects.

The interesting work here is the section splitter: 10-K filings have a
table of contents (which we must skip), forward references like
"Refer to Item 1A. Risk Factors" inside the body of other sections (which
we must skip), and a strict canonical order of Item headers (1, 1A, 1B, ...)
which we use to disambiguate ambiguous matches.

The splitter is validated and **fails loud** with a parse report if it can't
find the required sections — there is intentionally no silent
"fall back to one big document" path.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from llama_index.core.schema import Document

from .manifest import Filing, load_manifest
from .paths import RESULTS_DIR

logger = logging.getLogger(__name__)

# The lxml parser sometimes flags inline-XBRL filings as XML; suppress that
# warning — we deliberately want HTML semantics here.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# --- Section catalog --------------------------------------------------------

# Canonical 10-K order of Items. The splitter enforces monotonic ordering
# against this list to disambiguate forward references.
ITEM_ORDER: list[str] = [
    "1", "1A", "1B", "1C",
    "2", "3", "4",
    "5", "6", "7", "7A", "8",
    "9", "9A", "9B", "9C",
    "10", "11", "12", "13", "14",
    "15", "16",
]

# Human-readable section names — used as metadata and for the title-match
# defensive check below.
SECTION_TITLES: dict[str, str] = {
    "1":   "Business",
    "1A":  "Risk Factors",
    "1B":  "Unresolved Staff Comments",
    "1C":  "Cybersecurity",
    "2":   "Properties",
    "3":   "Legal Proceedings",
    "4":   "Mine Safety Disclosures",
    "5":   "Market for Registrant's Common Equity",
    "6":   "Reserved",
    "7":   "Management's Discussion and Analysis",
    "7A":  "Quantitative and Qualitative Disclosures About Market Risk",
    "8":   "Financial Statements and Supplementary Data",
    "9":   "Changes in and Disagreements with Accountants",
    "9A":  "Controls and Procedures",
    "9B":  "Other Information",
    "9C":  "Disclosure Regarding Foreign Jurisdictions",
    "10":  "Directors, Executive Officers and Corporate Governance",
    "11":  "Executive Compensation",
    "12":  "Security Ownership of Certain Beneficial Owners",
    "13":  "Certain Relationships and Related Transactions",
    "14":  "Principal Accountant Fees and Services",
    "15":  "Exhibits and Financial Statement Schedules",
    "16":  "Form 10-K Summary",
}

# Title-keyword hints — used to validate that a candidate match is actually
# the named section. At least one keyword (case-insensitive, whitespace- and
# apostrophe-insensitive) must appear within ~220 chars after the "Item N"
# hit. Multi-word phrases are used where a single word is too ambiguous —
# e.g. "Management" alone would match "Statement of Management's
# Responsibility" inside Item 8.
SECTION_TITLE_KEYWORDS: dict[str, list[str]] = {
    "1":   ["business"],
    "1A":  ["risk factors"],
    "1B":  ["unresolved staff comments"],
    "1C":  ["cybersecurity"],
    "2":   ["properties"],
    "3":   ["legal proceedings"],
    "4":   ["mine safety"],
    "5":   ["market for"],
    "6":   ["reserved", "selected financial"],
    "7":   ["managements discussion and analysis"],
    "7A":  ["quantitative and qualitative"],
    "8":   ["financial statements and supplementary"],
    "9":   ["changes in and disagreements"],
    "9A":  ["controls and procedures"],
    "9B":  ["other information"],
    "9C":  ["disclosure regarding foreign"],
    "10":  ["directors", "executive officers"],
    "11":  ["executive compensation"],
    "12":  ["security ownership"],
    "13":  ["certain relationships"],
    "14":  ["principal accountant"],
    "15":  ["exhibits"],
    "16":  ["form 10-k summary"],
}

# Sections that are *generally* required for ingestion to succeed. Item 1C
# (Cybersecurity) is conditionally required — see ``required_sections_for``.
REQUIRED_SECTIONS: list[str] = ["1", "1A", "1C", "7", "7A", "8"]

# Item 1C (Cybersecurity Disclosure) was added by SEC final rule 33-11216,
# effective for annual reports covering fiscal years ending on or after
# 2023-12-15. Filings for earlier fiscal-year-ends are not required to have
# Item 1C and we must not penalize them.
ITEM_1C_EFFECTIVE_DATE = "2023-12-15"


def required_sections_for(period_of_report: str) -> list[str]:
    """Required Items list for a filing with the given ``YYYY-MM-DD`` period.

    Item 1C is only required for fiscal-year-ends on/after 2023-12-15.
    """
    base = [iid for iid in REQUIRED_SECTIONS if iid != "1C"]
    if period_of_report >= ITEM_1C_EFFECTIVE_DATE:
        base.append("1C")
    return base


# Phrases that mark an Item reference as a forward/back reference rather
# than a section header. Two flavours:
#
# * LONG_RANGE: substring match in the 80 chars preceding the hit.
#   These phrases are unambiguous references at any short distance.
# * SHORT_RANGE: only count within the immediately-preceding 25 chars.
#   "Part II, Item 7" forward references attach "Part II," directly to
#   the Item N. We must NOT treat unrelated "Part X," text earlier in the
#   sentence as a reference (otherwise e.g. "...Part IV, Item 15 ... for
#   additional information. Item 8. Financial Statements..." gets
#   misclassified — the "Part IV," is referring to Item 15, not Item 8).
LONG_RANGE_REF_PHRASES: list[str] = [
    "refer to ", "see item ", "see part ", "see the ",
    "described in ", "discussed in ", "set forth in ",
    "located in ", "found in ", "found under ", "contained in ",
    "incorporated by reference", "as described under ",
    "as discussed under ", "as discussed in ", "as described in ",
    "we incorporate", "previously filed",
]

SHORT_RANGE_REF_PHRASES: list[str] = [
    "in part i ", "in part ii ", "in part iii ", "in part iv ",
    "part i, ", "part ii, ", "part iii, ", "part iv, ",
]

# Backwards-compat alias kept so external callers don't break.
REFERENCE_PHRASES: list[str] = LONG_RANGE_REF_PHRASES + SHORT_RANGE_REF_PHRASES

# Quote-like and bracket characters that, when immediately preceding a hit,
# indicate a textual reference.
REF_OPENING_CHARS: set[str] = {'"', "\u201c", "\u201d", "\u2018", "\u2019", "(", "["}


# --- Data classes -----------------------------------------------------------

@dataclass
class ParsedSection:
    """One section extracted from a filing."""

    item_id: str
    title: str
    start: int
    end: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class IngestionReport:
    """Per-filing diagnostic information from the section splitter."""

    filing: Filing
    plain_text_chars: int
    toc_end: int
    sections: list[ParsedSection]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def section_by_id(self, item_id: str) -> ParsedSection | None:
        return next((s for s in self.sections if s.item_id == item_id), None)


# --- HTML -> plain text -----------------------------------------------------

def html_to_text(html: str) -> str:
    """Render filing HTML into a single space-normalized plain-text string."""
    soup = BeautifulSoup(html, "lxml")
    # Strip noise we never want indexed.
    for tag in soup(["script", "style", "noscript", "head"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # Normalize whitespace and collapse non-breaking spaces.
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text


# --- Section splitter -------------------------------------------------------

def _build_item_pattern(item_id: str) -> re.Pattern[str]:
    """Pattern matching `Item N[A-Z]?` with tolerant whitespace / case."""
    # Negative-lookahead `(?![A-Za-z0-9])` ensures `Item 7` does NOT match
    # `Item 7A`. `(?<!\w)` ensures we don't match mid-word.
    return re.compile(
        rf"(?<!\w)(?:Item|ITEM)\s+{re.escape(item_id)}(?![A-Za-z0-9])\.?",
        re.UNICODE,
    )


def _is_forward_reference(text: str, pos: int) -> bool:
    """Heuristic: True if the Item N hit at ``pos`` is a textual reference.

    See ``LONG_RANGE_REF_PHRASES`` / ``SHORT_RANGE_REF_PHRASES`` for the two
    distance scopes.
    """
    if pos == 0:
        return False
    if text[pos - 1] in REF_OPENING_CHARS:
        return True
    long_window = text[max(0, pos - 80):pos].lower()
    if any(phrase in long_window for phrase in LONG_RANGE_REF_PHRASES):
        return True
    short_window = text[max(0, pos - 25):pos].lower()
    if any(phrase in short_window for phrase in SHORT_RANGE_REF_PHRASES):
        return True
    return False


def _matches_title_keyword(text: str, item_id: str, end_pos: int) -> bool:
    """Does the post-match window contain an expected section-title keyword?

    Compact-normalizes whitespace, apostrophes, and curly quotes before
    substring matching so we tolerate the "B USINESS" / "Management's"
    letter-spacing and smart-quote artifacts that some 10-K HTML renderers
    (notably Microsoft's) produce.
    """
    snippet = text[end_pos:end_pos + 220].lower()
    snippet_compact = re.sub(r"[\s'\u2018\u2019\u2014\u2013\-.]+", "", snippet)
    keywords = SECTION_TITLE_KEYWORDS.get(item_id, [])
    if not keywords:
        return True
    for kw in keywords:
        if re.sub(r"[\s'\u2018\u2019\u2014\u2013\-.]+", "", kw.lower()) in snippet_compact:
            return True
    return False


def _find_toc_end(text: str, all_hits: list[tuple[int, str]]) -> int:
    """Locate the end of the table of contents.

    The TOC shows up as a tightly packed run of consecutive Item hits with
    short gaps between them. We end the TOC at the position just past the
    last hit in that run.

    Returns 0 if no TOC-like cluster is detected. Note: TOC removal is now
    only used for diagnostics — the real anchor selection happens via
    body-length scoring in :func:`split_into_sections` and does not depend
    on this estimate being precise.
    """
    if not all_hits:
        return 0
    GAP_THRESHOLD = 250
    MIN_RUN = 8
    run_start_idx = 0
    for i in range(1, len(all_hits)):
        gap = all_hits[i][0] - all_hits[i - 1][0]
        if gap > GAP_THRESHOLD:
            if i - run_start_idx >= MIN_RUN:
                return all_hits[i - 1][0] + 200
            run_start_idx = i
    # Whole document is one tight run (unlikely) — treat as no TOC.
    if len(all_hits) - run_start_idx >= MIN_RUN and run_start_idx == 0:
        return 0
    return 0


def split_into_sections(text: str) -> tuple[int, list[ParsedSection]]:
    """Split the plain-text 10-K into sections by Item header.

    Approach: for each candidate `Item N` match in the document, filter out
    forward references (preceded by "refer to"/"see"/quote) and matches not
    immediately followed by the expected section title. Then score each
    remaining candidate by the distance to the *next* candidate of any kind
    — real section headers have thousands of chars of body before the next
    Item; TOC entries have only tens of chars. Per item ID, pick the highest
    scoring candidate. Finally enforce that anchors appear in the canonical
    Item order; out-of-order anchors are dropped (which produces a
    validation error rather than a silent miss).

    Returns ``(toc_end, sections)`` where ``toc_end`` is a diagnostic
    estimate and ``sections`` are document-ordered.
    """
    # Collect every Item N candidate position, item-id-keyed and sorted.
    hits_by_item: dict[str, list[int]] = {}
    for item_id in ITEM_ORDER:
        pat = _build_item_pattern(item_id)
        hits_by_item[item_id] = [m.start() for m in pat.finditer(text)]

    all_hits = sorted((p, iid) for iid, ps in hits_by_item.items() for p in ps)
    toc_end = _find_toc_end(text, all_hits)

    # Filter candidates per item_id: not a forward reference AND title matches.
    filtered_by_item: dict[str, list[int]] = {}
    for item_id, positions in hits_by_item.items():
        pat = _build_item_pattern(item_id)
        kept = []
        for p in positions:
            if _is_forward_reference(text, p):
                continue
            m = pat.match(text, p)
            assert m is not None
            if not _matches_title_keyword(text, item_id, m.end()):
                continue
            kept.append(p)
        filtered_by_item[item_id] = kept

    # Flat sorted list of all kept candidates — used to compute body-length
    # scores (distance to the next kept Item header of any kind).
    kept_flat = sorted(p for ps in filtered_by_item.values() for p in ps)

    def body_length(p: int) -> int:
        # Find the next kept position strictly greater than p.
        for q in kept_flat:
            if q > p:
                return q - p
        return len(text) - p

    # For each item_id, pick the candidate with the maximum body length.
    chosen: dict[str, int] = {}
    for item_id, positions in filtered_by_item.items():
        if not positions:
            continue
        best = max(positions, key=body_length)
        chosen[item_id] = best

    # Enforce monotonic canonical order. If an item's chosen anchor is not
    # strictly greater than the previous one, drop it — that signals the
    # algorithm picked something wrong and we want validation to fail loudly
    # rather than silently produce overlapping sections.
    #
    # MIN_SECTION_LEN is intentionally small: Items 4 ("Mine Safety
    # Disclosures") and 6 ("Reserved") are commonly just one or two
    # sentences ("Not applicable.", "[Reserved]"), so legitimate sections
    # can sit within ~50 chars of each other.
    anchors: list[tuple[int, str]] = []
    cursor = 0
    MIN_SECTION_LEN = 30
    for item_id in ITEM_ORDER:
        if item_id not in chosen:
            continue
        pos = chosen[item_id]
        if pos < cursor + MIN_SECTION_LEN:
            continue
        anchors.append((pos, item_id))
        cursor = pos

    # Build sections by slicing between anchors.
    sections: list[ParsedSection] = []
    for idx, (start, item_id) in enumerate(anchors):
        end = anchors[idx + 1][0] if idx + 1 < len(anchors) else len(text)
        body = text[start:end].strip()
        sections.append(
            ParsedSection(
                item_id=item_id,
                title=SECTION_TITLES.get(item_id, f"Item {item_id}"),
                start=start,
                end=end,
                text=body,
            )
        )
    return toc_end, sections


# --- Validation -------------------------------------------------------------

def _validate(filing: Filing, plain_text: str, sections: list[ParsedSection]) -> tuple[list[str], list[str]]:
    """Return (warnings, errors) lists. Errors block ingestion."""
    warnings_: list[str] = []
    errors: list[str] = []
    found_ids = {s.item_id for s in sections}

    required = required_sections_for(filing.period_of_report)
    missing = [iid for iid in required if iid not in found_ids]
    if missing:
        errors.append(
            f"Missing required sections: {', '.join('Item ' + i for i in missing)}"
        )

    # If Item 1C is in the generic required list but excluded for this filing,
    # warn (but don't error) when it's absent — useful diagnostic.
    if "1C" in REQUIRED_SECTIONS and "1C" not in required and "1C" not in found_ids:
        warnings_.append(
            "Item 1C (Cybersecurity) absent — expected for fiscal years "
            f"ending before {ITEM_1C_EFFECTIVE_DATE}."
        )

    # Item 7 (MD&A) is content-rich and short Item 7 strongly indicates a
    # parse failure.
    mdna = next((s for s in sections if s.item_id == "7"), None)
    if mdna and mdna.char_count < 5_000:
        errors.append(
            f"Item 7 (MD&A) is only {mdna.char_count} chars; expected >5,000."
        )

    # No single non-Item-8 section should consume more than 70% of the doc;
    # that would mean we collapsed multiple sections together.
    total = max(len(plain_text), 1)
    for s in sections:
        if s.item_id == "8":
            continue  # Item 8 (Financial Statements) is legitimately huge
        if s.char_count / total > 0.70:
            errors.append(
                f"Section Item {s.item_id} is {s.char_count / total:.0%} of "
                f"the document; splitter likely collapsed multiple sections."
            )

    # Monotonic order is guaranteed by the algorithm; warn anyway as a safety net.
    last_pos = -1
    for s in sections:
        if s.start <= last_pos:
            warnings_.append(f"Item {s.item_id} appears out of order at offset {s.start}.")
        last_pos = s.start

    return warnings_, errors


# --- Public API -------------------------------------------------------------

def ingest_filing(filing: Filing) -> IngestionReport:
    """Read a filing from disk and split it into validated sections."""
    if not filing.local_path.exists():
        raise FileNotFoundError(
            f"Filing not downloaded: {filing.local_path}. "
            f"Run `uv run python scripts/download_reports.py` first."
        )

    html = filing.local_path.read_text(encoding="utf-8", errors="replace")
    plain_text = html_to_text(html)
    toc_end, sections = split_into_sections(plain_text)
    warnings_, errors = _validate(filing, plain_text, sections)
    return IngestionReport(
        filing=filing,
        plain_text_chars=len(plain_text),
        toc_end=toc_end,
        sections=sections,
        warnings=warnings_,
        errors=errors,
    )


def report_to_documents(report: IngestionReport) -> list[Document]:
    """Convert a successful ingestion report into LlamaIndex Documents."""
    if not report.ok:
        raise ValueError(
            f"Cannot convert failed ingestion for "
            f"{report.filing.company} FY{report.filing.fiscal_year}: "
            f"{report.errors}"
        )
    filing = report.filing
    docs: list[Document] = []
    for section in report.sections:
        metadata = {
            "company": filing.company,
            "ticker": filing.ticker,
            "fiscal_year": filing.fiscal_year,
            "filing_date": filing.filing_date,
            "period_of_report": filing.period_of_report,
            "report_type": filing.form,
            "accession": filing.accession,
            "source_url": filing.source_url,
            "section_id": "item_" + section.item_id.lower(),
            "section_title": section.title,
            "section_char_count": section.char_count,
        }
        docs.append(
            Document(
                text=section.text,
                metadata=metadata,
                # Exclude bulky/uninformative metadata from both embedding and
                # LLM prompts to keep prompts focused.
                excluded_embed_metadata_keys=[
                    "accession", "source_url", "filing_date",
                    "period_of_report", "section_char_count",
                ],
                excluded_llm_metadata_keys=[
                    "accession", "section_char_count",
                ],
                doc_id=(
                    f"{filing.company}_FY{filing.fiscal_year}"
                    f"_item_{section.item_id.lower()}"
                ),
            )
        )
    return docs


def ingest_all(filings: list[Filing] | None = None) -> tuple[list[Document], list[IngestionReport]]:
    """Ingest every filing in the manifest.

    Returns the flattened list of Documents *and* the per-filing reports
    (used by ``write_parse_report``).

    Raises ``RuntimeError`` if any filing fails validation. The parse report
    is written first so the caller can inspect what failed.
    """
    filings = filings or load_manifest()
    reports: list[IngestionReport] = [ingest_filing(f) for f in filings]
    write_parse_report(reports)
    failed = [r for r in reports if not r.ok]
    if failed:
        msg_lines = ["Ingestion validation failed for one or more filings:"]
        for r in failed:
            msg_lines.append(
                f"  - {r.filing.company} FY{r.filing.fiscal_year}: " + "; ".join(r.errors)
            )
        msg_lines.append(
            f"See {(RESULTS_DIR / 'parse_report.md').as_posix()} for details."
        )
        raise RuntimeError("\n".join(msg_lines))
    docs: list[Document] = []
    for r in reports:
        docs.extend(report_to_documents(r))
    return docs, reports


def write_parse_report(reports: list[IngestionReport], out_path: Path | None = None) -> Path:
    """Write a markdown summary of section splitting per filing."""
    out_path = out_path or (RESULTS_DIR / "parse_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# 10-K Parse Report",
        "",
        "Per-filing summary of section detection.",
        "",
        "| Company | FY | Plain chars | TOC end | Sections | Required OK | Errors |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for r in reports:
        required = required_sections_for(r.filing.period_of_report)
        required_ok = "yes" if all(r.section_by_id(s) for s in required) else "**no**"
        errors_cell = "; ".join(r.errors) if r.errors else "-"
        lines.append(
            f"| {r.filing.company} | {r.filing.fiscal_year} "
            f"| {r.plain_text_chars:,} | {r.toc_end:,} | {len(r.sections)} "
            f"| {required_ok} | {errors_cell} |"
        )
    lines.append("")
    for r in reports:
        lines.append(f"## {r.filing.company.upper()} FY{r.filing.fiscal_year}")
        lines.append("")
        lines.append("| Item | Title | Start | Chars |")
        lines.append("|---|---|---:|---:|")
        for s in r.sections:
            lines.append(
                f"| {s.item_id} | {s.title} | {s.start:,} | {s.char_count:,} |"
            )
        if r.warnings:
            lines.append("")
            lines.append("**Warnings:** " + "; ".join(r.warnings))
        if r.errors:
            lines.append("")
            lines.append("**Errors:** " + "; ".join(r.errors))
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
