"""Retrieval evaluation: Hit-Rate and MRR over a fixed question set.

Reports BOTH metrics in two flavors:

* **Metadata** match — the retrieved chunk's (company, fiscal_year,
  section_id) tuple matches one of the question's expected tuples.
* **Strict** match — metadata match AND the chunk text contains at least
  one of the ``must_contain_any`` keyword anchors for that expected entry.

The strict metric is the more honest measure of "actually retrieved the
answer-bearing chunk".
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .index import IndexConfig, load_index
from .paths import QUESTIONS_PATH, RESULTS_DIR
from .retrieve import (
    QuestionFilters, build_retriever, build_reranking_retriever,
    format_source_citation, infer_filters_from_question,
)


# --- Data shapes ------------------------------------------------------------

@dataclass
class ExpectedSource:
    company: str
    fiscal_year: int
    section_id: str
    must_contain_any: list[str] = field(default_factory=list)


@dataclass
class EvalQuestion:
    id: str
    category: str
    question: str
    expected: list[ExpectedSource]


@dataclass
class HitResult:
    """One chunk's match against the expected ground-truth list."""

    rank: int
    metadata_hit: bool
    strict_hit: bool
    matched_expected_index: int | None
    citation: str
    snippet: str


@dataclass
class QuestionResult:
    question_id: str
    question: str
    category: str
    filter_companies: list[str]
    filter_years: list[int]
    metadata_first_rank: int | None  # 1-based; None if not found in top-k
    strict_first_rank: int | None
    hits: list[HitResult]

    @property
    def metadata_mrr(self) -> float:
        return 1.0 / self.metadata_first_rank if self.metadata_first_rank else 0.0

    @property
    def strict_mrr(self) -> float:
        return 1.0 / self.strict_first_rank if self.strict_first_rank else 0.0


@dataclass
class EvalSummary:
    """Aggregate report for a single evaluation run."""

    config_name: str
    top_k: int
    use_filters: bool
    n_questions: int
    metadata_hit_rate: float       # share of questions with metadata match in top-k
    strict_hit_rate: float
    metadata_mrr: float
    strict_mrr: float
    per_question: list[QuestionResult]


# --- Question loading -------------------------------------------------------

def load_questions(path: Path | None = None) -> list[EvalQuestion]:
    """Load and validate the evaluation question set."""
    path = path or QUESTIONS_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "questions" not in raw:
        raise ValueError(f"{path} must have a top-level `questions:` list.")
    out: list[EvalQuestion] = []
    for q in raw["questions"]:
        expected = [
            ExpectedSource(
                company=e["company"],
                fiscal_year=int(e["fiscal_year"]),
                section_id=e["section_id"],
                must_contain_any=list(e.get("must_contain_any") or []),
            )
            for e in q["expected"]
        ]
        out.append(EvalQuestion(
            id=q["id"],
            category=q.get("category", "uncategorized"),
            question=q["question"],
            expected=expected,
        ))
    return out


# --- Matching helpers -------------------------------------------------------

def _match_node_to_expected(node: NodeWithScore, expected: list[ExpectedSource]) -> tuple[bool, bool, int | None]:
    """Return (metadata_hit, strict_hit, matched_expected_index)."""
    md = node.node.metadata
    text_lower = node.node.get_content().lower()
    for i, e in enumerate(expected):
        if (
            md.get("company") == e.company
            and md.get("fiscal_year") == e.fiscal_year
            and md.get("section_id") == e.section_id
        ):
            keywords = e.must_contain_any
            if not keywords:
                return (True, True, i)  # no strict requirement → metadata match counts as strict
            for kw in keywords:
                if kw.lower() in text_lower:
                    return (True, True, i)
            return (True, False, i)
    return (False, False, None)


# --- Evaluator --------------------------------------------------------------

def evaluate_question(
    question: EvalQuestion,
    index: VectorStoreIndex,
    top_k: int = 5,
    use_filters: bool = True,
    use_reranker: bool = False,
    candidate_top_k: int = 20,
) -> QuestionResult:
    inferred = infer_filters_from_question(question.question)
    filters = inferred.to_llama_filters() if use_filters else None
    if use_reranker:
        retriever = build_reranking_retriever(
            index,
            candidate_top_k=candidate_top_k,
            final_top_k=top_k,
            filters=filters,
        )
    else:
        retriever = build_retriever(index, similarity_top_k=top_k, filters=filters)
    nodes = retriever.retrieve(question.question)

    hits: list[HitResult] = []
    md_first: int | None = None
    strict_first: int | None = None
    for rank, n in enumerate(nodes, start=1):
        md_hit, strict_hit, expected_idx = _match_node_to_expected(n, question.expected)
        if md_hit and md_first is None:
            md_first = rank
        if strict_hit and strict_first is None:
            strict_first = rank
        snippet = n.node.get_content().strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        hits.append(HitResult(
            rank=rank,
            metadata_hit=md_hit,
            strict_hit=strict_hit,
            matched_expected_index=expected_idx,
            citation=format_source_citation(n.node.metadata),
            snippet=snippet,
        ))

    return QuestionResult(
        question_id=question.id,
        question=question.question,
        category=question.category,
        filter_companies=inferred.companies if use_filters else [],
        filter_years=inferred.fiscal_years if use_filters else [],
        metadata_first_rank=md_first,
        strict_first_rank=strict_first,
        hits=hits,
    )


def evaluate(
    index: VectorStoreIndex,
    top_k: int = 5,
    use_filters: bool = True,
    use_reranker: bool = False,
    candidate_top_k: int = 20,
    questions: list[EvalQuestion] | None = None,
    config_name: str = "baseline",
) -> EvalSummary:
    questions = questions or load_questions()
    per_q = [
        evaluate_question(
            q, index,
            top_k=top_k, use_filters=use_filters,
            use_reranker=use_reranker, candidate_top_k=candidate_top_k,
        )
        for q in questions
    ]
    n = len(per_q)
    md_hits = sum(1 for r in per_q if r.metadata_first_rank is not None)
    strict_hits = sum(1 for r in per_q if r.strict_first_rank is not None)
    return EvalSummary(
        config_name=config_name,
        top_k=top_k,
        use_filters=use_filters,
        n_questions=n,
        metadata_hit_rate=md_hits / n if n else 0.0,
        strict_hit_rate=strict_hits / n if n else 0.0,
        metadata_mrr=statistics.fmean(r.metadata_mrr for r in per_q) if n else 0.0,
        strict_mrr=statistics.fmean(r.strict_mrr for r in per_q) if n else 0.0,
        per_question=per_q,
    )


# --- Output writers ---------------------------------------------------------

def write_eval_json(summary: EvalSummary, out_path: Path | None = None) -> Path:
    out_path = out_path or (RESULTS_DIR / f"eval_{summary.config_name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_name": summary.config_name,
        "top_k": summary.top_k,
        "use_filters": summary.use_filters,
        "n_questions": summary.n_questions,
        "metrics": {
            "metadata_hit_rate": summary.metadata_hit_rate,
            "strict_hit_rate": summary.strict_hit_rate,
            "metadata_mrr": summary.metadata_mrr,
            "strict_mrr": summary.strict_mrr,
        },
        "per_question": [asdict(r) for r in summary.per_question],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def write_eval_markdown(summary: EvalSummary, out_path: Path | None = None) -> Path:
    out_path = out_path or (RESULTS_DIR / f"eval_{summary.config_name}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"# Retrieval Evaluation — `{summary.config_name}`",
        "",
        f"- Top-k: **{summary.top_k}**",
        f"- Metadata filters: **{'on' if summary.use_filters else 'off'}**",
        f"- Questions: **{summary.n_questions}**",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Metadata Hit@{summary.top_k} | {summary.metadata_hit_rate:.3f} |",
        f"| Strict Hit@{summary.top_k}   | {summary.strict_hit_rate:.3f} |",
        f"| Metadata MRR                  | {summary.metadata_mrr:.3f} |",
        f"| Strict MRR                    | {summary.strict_mrr:.3f} |",
        "",
        "## Per-question",
        "",
        f"| id | category | Md-rank | Strict-rank | top-1 chunk |",
        f"|---|---|---:|---:|---|",
    ]
    for r in summary.per_question:
        top1 = r.hits[0].citation if r.hits else "-"
        md_r = r.metadata_first_rank or "—"
        st_r = r.strict_first_rank or "—"
        lines.append(f"| {r.question_id} | {r.category} | {md_r} | {st_r} | {top1} |")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_failure_log(
    summary: EvalSummary, k_worst: int = 3, out_path: Path | None = None
) -> Path:
    """Write a markdown-formatted log of the ``k_worst`` worst questions."""
    out_path = out_path or (RESULTS_DIR / f"failures_{summary.config_name}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        summary.per_question,
        key=lambda r: (r.strict_mrr, r.metadata_mrr),  # ascending: worst first
    )
    worst = ranked[:k_worst]
    lines: list[str] = [
        f"# Retrieval Failure Log — `{summary.config_name}`",
        "",
        f"The {k_worst} worst-performing questions by strict MRR (then metadata MRR).",
        "",
    ]
    for r in worst:
        lines.append(f"## {r.question_id} — `{r.category}`")
        lines.append("")
        lines.append(f"**Question:** {r.question}")
        lines.append("")
        lines.append(f"- Metadata first-rank: **{r.metadata_first_rank or 'not in top-k'}**")
        lines.append(f"- Strict first-rank:   **{r.strict_first_rank or 'not in top-k'}**")
        lines.append(f"- Inferred filters: companies={r.filter_companies or '-'}, years={r.filter_years or '-'}")
        lines.append("")
        lines.append("### Top retrievals")
        lines.append("")
        for h in r.hits:
            flags = []
            if h.metadata_hit:
                flags.append("md")
            if h.strict_hit:
                flags.append("strict")
            tag = ",".join(flags) or "miss"
            lines.append(f"- **[{h.rank}] ({tag})** {h.citation}")
            lines.append(f"  > {h.snippet}")
        lines.append("")
        lines.append("### Diagnosis (author-judged)")
        lines.append("")
        diagnosis = _diagnose(r)
        lines.append(f"- {diagnosis}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _diagnose(r: QuestionResult) -> str:
    if r.metadata_first_rank is None:
        return (
            "**Missed company/year/section** — none of the top-k chunks came "
            "from the expected (company, fiscal_year, section_id). Likely "
            "embedding similarity favored unrelated chunks; consider a "
            "metadata filter or rerank."
        )
    if r.strict_first_rank is None:
        return (
            "**Vague chunk** — metadata correct, but no chunk in the right "
            "section contained the expected keyword anchor. The relevant "
            "passage was outranked by other chunks of the same section. "
            "A larger top-k or rerank may help."
        )
    if r.strict_first_rank > 3:
        return (
            f"**Weak rank ({r.strict_first_rank})** — the answer-bearing "
            "chunk was retrieved but not near the top. A reranker would help."
        )
    return "Answer-bearing chunk was retrieved at rank ≤3 — minor weakness only."


# --- One-shot convenience ---------------------------------------------------

def run_baseline(top_k: int = 5, use_filters: bool = True) -> EvalSummary:
    """Run the baseline (default IndexConfig) eval and persist artifacts."""
    index = load_index(IndexConfig())
    summary = evaluate(index, top_k=top_k, use_filters=use_filters, config_name="baseline")
    write_eval_json(summary)
    write_eval_markdown(summary)
    write_failure_log(summary)
    return summary
