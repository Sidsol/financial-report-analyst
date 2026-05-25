"""AgentWorkflow orchestration for the cited Q&A pipeline.

Implements the six-step workflow from the project spec:

    classify_query
        ↓
    retrieve_documents
        ↓
    validate_sources
        ↓
    synthesize_answer
        ↓
    produce_citations
        ↓
    log_failure_if_low_confidence

Built on top of LlamaIndex's ``Workflow`` primitive. Used by Lab 7.3 and
exposed via ``scripts/run_query.py --agent``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
    Context, Event, StartEvent, StopEvent, Workflow, step,
)

from .config import build_llm
from .index import IndexConfig, load_index
from .paths import RESULTS_DIR
from .retrieve import (
    PostprocessingRetriever, build_retriever, build_reranking_retriever,
    format_source_citation, infer_filters_from_question,
)
from .synthesize import CITED_QA_TEMPLATE, CitedAnswer, CitedSource, _format_sources_for_prompt

logger = logging.getLogger(__name__)

# Lowest similarity score below which we log the answer as a likely failure.
DEFAULT_LOW_CONFIDENCE_SCORE = 0.55


# --- Workflow events --------------------------------------------------------

class QueryClassified(Event):
    question: str
    category: str
    requires_filter: bool


class CandidatesRetrieved(Event):
    question: str
    category: str
    nodes: list[NodeWithScore]


class SourcesValidated(Event):
    question: str
    category: str
    nodes: list[NodeWithScore]
    confidence: float


class AnswerSynthesized(Event):
    question: str
    category: str
    nodes: list[NodeWithScore]
    confidence: float
    answer: str


# --- Workflow ---------------------------------------------------------------

class FinancialAnalystAgent(Workflow):
    """Six-step retrieve→validate→synthesize→cite agent."""

    def __init__(
        self,
        retriever: BaseRetriever | PostprocessingRetriever,
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_SCORE,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(timeout=timeout)
        self._retriever = retriever
        self._low_conf = low_confidence_threshold
        self._llm = build_llm()

    @step
    async def classify_query(self, ev: StartEvent) -> QueryClassified:
        question = ev.question
        inferred = infer_filters_from_question(question)
        # Heuristic classification:
        # * comparison if multiple companies or years mentioned
        # * single otherwise
        category = (
            "comparison"
            if len(inferred.companies) > 1 or len(inferred.fiscal_years) > 1
            else "single"
        )
        requires_filter = not inferred.is_empty()
        logger.info(
            "classify: q=%r category=%s requires_filter=%s",
            question[:60], category, requires_filter,
        )
        return QueryClassified(
            question=question, category=category, requires_filter=requires_filter
        )

    @step
    async def retrieve_documents(self, ev: QueryClassified) -> CandidatesRetrieved:
        nodes = self._retriever.retrieve(ev.question)
        logger.info("retrieve: got %d candidates", len(nodes))
        return CandidatesRetrieved(
            question=ev.question, category=ev.category, nodes=nodes
        )

    @step
    async def validate_sources(self, ev: CandidatesRetrieved) -> SourcesValidated:
        # Confidence = top node similarity score.
        scores = [n.score or 0.0 for n in ev.nodes]
        confidence = max(scores) if scores else 0.0
        logger.info("validate: top score=%.3f, n_nodes=%d", confidence, len(ev.nodes))
        return SourcesValidated(
            question=ev.question, category=ev.category,
            nodes=ev.nodes, confidence=confidence,
        )

    @step
    async def synthesize_answer(self, ev: SourcesValidated) -> AnswerSynthesized:
        if not ev.nodes:
            answer = "No relevant sources were retrieved for this question."
        else:
            context = _format_sources_for_prompt(ev.nodes)
            prompt = CITED_QA_TEMPLATE.format(
                context_str=context, query_str=ev.question
            )
            response = self._llm.complete(prompt)
            answer = str(response)
        return AnswerSynthesized(
            question=ev.question, category=ev.category, nodes=ev.nodes,
            confidence=ev.confidence, answer=answer,
        )

    @step
    async def produce_citations_and_finish(self, ev: AnswerSynthesized) -> StopEvent:
        sources = [
            CitedSource(
                citation_id=i,
                citation=format_source_citation(n.node.metadata),
                score=n.score or 0.0,
                snippet=n.node.get_content(),
            )
            for i, n in enumerate(ev.nodes, start=1)
        ]
        if ev.confidence < self._low_conf:
            self._log_low_confidence(ev, sources)
        return StopEvent(result=CitedAnswer(
            question=ev.question, answer=ev.answer, sources=sources,
        ))

    def _log_low_confidence(
        self, ev: AnswerSynthesized, sources: list[CitedSource]
    ) -> None:
        path = RESULTS_DIR / "agent_failures.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "question": ev.question,
            "category": ev.category,
            "top_score": ev.confidence,
            "threshold": self._low_conf,
            "n_sources": len(sources),
            "top_citations": [s.citation for s in sources[:3]],
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.warning(
            "low confidence: score=%.3f < %.3f, logged to %s",
            ev.confidence, self._low_conf, path,
        )


# --- Sync wrapper for CLI use ----------------------------------------------

def run_agent_query(
    question: str,
    retriever: BaseRetriever | PostprocessingRetriever | None = None,
) -> CitedAnswer:
    """Run one query through the agent workflow. Synchronous wrapper."""
    import asyncio

    if retriever is None:
        index = load_index(IndexConfig())
        filters = infer_filters_from_question(question).to_llama_filters()
        retriever = build_retriever(index, similarity_top_k=5, filters=filters)
    agent = FinancialAnalystAgent(retriever=retriever)

    async def _go() -> CitedAnswer:
        handler = agent.run(question=question)
        return await handler

    return asyncio.run(_go())
