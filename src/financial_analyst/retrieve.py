"""Build retrievers from the persisted index, with optional metadata filters.

The retriever layer is shared by the simple query engine (Lab 7.1), the
evaluation harness (Lab 7.2), and the tuning sweep (Lab 7.3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from .ingest import SECTION_TITLES

# Known company aliases — used by the question parser to construct metadata
# filters for queries like "How did Apple's iPhone revenue change?"
COMPANY_ALIASES: dict[str, list[str]] = {
    "microsoft": ["microsoft", "msft", "azure", "windows", "office"],
    "apple":     ["apple", "aapl", "iphone", "ipad", "mac "],
    "nvidia":    ["nvidia", "nvda", "geforce", "cuda"],
}
TICKER_TO_COMPANY: dict[str, str] = {
    "MSFT": "microsoft", "AAPL": "apple", "NVDA": "nvidia",
}


@dataclass
class QuestionFilters:
    """Metadata filters inferred from a natural-language question."""

    companies: list[str] = field(default_factory=list)
    fiscal_years: list[int] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.companies and not self.fiscal_years

    def to_llama_filters(self) -> MetadataFilters | None:
        """Convert into a LlamaIndex ``MetadataFilters`` (or None)."""
        if self.is_empty():
            return None
        filters: list[MetadataFilter] = []
        if self.companies:
            filters.append(MetadataFilter(
                key="company", value=self.companies, operator=FilterOperator.IN
            ))
        if self.fiscal_years:
            filters.append(MetadataFilter(
                key="fiscal_year", value=self.fiscal_years, operator=FilterOperator.IN
            ))
        return MetadataFilters(filters=filters, condition=FilterCondition.AND)


def infer_filters_from_question(question: str) -> QuestionFilters:
    """Extract company and fiscal-year filters from a question.

    Multi-value friendly: returns all mentioned companies / years rather than
    forcing a single choice. Vague questions yield an empty filter set so the
    retriever sees the whole corpus.
    """
    q = question.lower()
    companies: list[str] = []
    for canonical, aliases in COMPANY_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias).strip()}\b", q) for alias in aliases):
            companies.append(canonical)
    # Tickers (uppercase boundary check against the original)
    for ticker, canon in TICKER_TO_COMPANY.items():
        if re.search(rf"\b{ticker}\b", question):
            if canon not in companies:
                companies.append(canon)

    fiscal_years: list[int] = []
    # Match FY2024 / FY 2024 / fiscal year 2024 / 2024
    for m in re.finditer(r"(?:fy|fiscal year)\s*(20\d{2})", q):
        y = int(m.group(1))
        if y not in fiscal_years:
            fiscal_years.append(y)
    for m in re.finditer(r"\b(20\d{2})\b", q):
        y = int(m.group(1))
        if 2020 <= y <= 2030 and y not in fiscal_years:
            fiscal_years.append(y)
    # "last three years", "past N years" → no filter (let retrieval handle it)
    return QuestionFilters(companies=companies, fiscal_years=fiscal_years)


def build_retriever(
    index: VectorStoreIndex,
    similarity_top_k: int = 5,
    filters: MetadataFilters | None = None,
) -> VectorIndexRetriever:
    """Construct a configurable retriever from a persisted index."""
    return VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
        filters=filters,
    )


class PostprocessingRetriever:
    """A retriever wrapper that applies node post-processors (e.g. a reranker)
    after similarity search. Reuses the same ``retrieve(query)`` contract as
    a plain LlamaIndex retriever so it drops into the evaluator unchanged.
    """

    def __init__(
        self,
        base: VectorIndexRetriever,
        postprocessors: list[BaseNodePostprocessor],
        final_top_n: int | None = None,
    ) -> None:
        self._base = base
        self._postprocessors = postprocessors
        self._final_top_n = final_top_n

    def retrieve(self, query: str) -> list[NodeWithScore]:
        nodes = self._base.retrieve(query)
        bundle = QueryBundle(query_str=query)
        for pp in self._postprocessors:
            nodes = pp.postprocess_nodes(nodes, query_bundle=bundle)
        if self._final_top_n is not None:
            nodes = nodes[: self._final_top_n]
        return nodes


def build_reranking_retriever(
    index: VectorStoreIndex,
    candidate_top_k: int = 20,
    final_top_k: int = 5,
    filters: MetadataFilters | None = None,
    reranker_model: str = "BAAI/bge-reranker-base",
) -> PostprocessingRetriever:
    """Retriever that fetches ``candidate_top_k`` chunks, then reranks them
    with a cross-encoder down to ``final_top_k``.
    """
    from llama_index.core.postprocessor import SentenceTransformerRerank
    base = build_retriever(
        index, similarity_top_k=candidate_top_k, filters=filters
    )
    reranker = SentenceTransformerRerank(model=reranker_model, top_n=final_top_k)
    return PostprocessingRetriever(base, [reranker], final_top_n=final_top_k)


def format_source_citation(metadata: dict) -> str:
    """Render a source-node metadata dict as a human-readable citation."""
    company = metadata.get("company", "?").upper()
    fy = metadata.get("fiscal_year", "?")
    section_id = metadata.get("section_id", "")
    section_title = metadata.get("section_title") or SECTION_TITLES.get(
        section_id.replace("item_", "").upper(), section_id
    )
    return f"{company} FY{fy} — Item {section_id.replace('item_', '').upper()} ({section_title})"
