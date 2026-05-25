"""Grounded answer synthesis with explicit citations."""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore

from .config import build_llm
from .retrieve import format_source_citation


CITED_QA_TEMPLATE = PromptTemplate(
    "You are a financial analyst assistant. Answer the user's question using ONLY "
    "the numbered sources below. Each factual claim MUST end with the source ids "
    "in square brackets, e.g. `[1]` or `[2, 3]`. If the sources don't cover the "
    "question, say so plainly.\n"
    "\n"
    "Sources:\n"
    "{context_str}\n"
    "\n"
    "Question: {query_str}\n"
    "\n"
    "Answer:"
)


@dataclass
class CitedSource:
    """One source row presented under an answer."""

    citation_id: int
    citation: str
    score: float
    snippet: str


@dataclass
class CitedAnswer:
    """An LLM answer plus the supporting source rows."""

    question: str
    answer: str
    sources: list[CitedSource]

    def render(self) -> str:
        lines = [self.answer.strip(), "", "Sources:"]
        for s in self.sources:
            lines.append(f"  [{s.citation_id}] {s.citation}  (score={s.score:.3f})")
            snippet = s.snippet.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            lines.append(f"      {snippet}")
        return "\n".join(lines)


def _format_sources_for_prompt(nodes: list[NodeWithScore]) -> str:
    """Render the retrieved nodes into the numbered block fed to the LLM."""
    chunks = []
    for i, n in enumerate(nodes, start=1):
        citation = format_source_citation(n.node.metadata)
        text = n.node.get_content().strip().replace("\n\n", "\n")
        chunks.append(f"[{i}] {citation}\n{text}")
    return "\n\n".join(chunks)


def build_query_engine(retriever: BaseRetriever) -> RetrieverQueryEngine:
    """Construct the standard cited query engine used in Lab 7.1."""
    llm = build_llm()
    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=ResponseMode.COMPACT,
        text_qa_template=CITED_QA_TEMPLATE,
    )
    return RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)


def answer_with_citations(
    retriever: BaseRetriever, question: str
) -> CitedAnswer:
    """Retrieve, synthesize a grounded answer, and package source citations."""
    nodes = retriever.retrieve(question)
    if not nodes:
        return CitedAnswer(question=question, answer="No relevant sources found.", sources=[])

    llm = build_llm()
    context = _format_sources_for_prompt(nodes)
    prompt = CITED_QA_TEMPLATE.format(context_str=context, query_str=question)
    response = llm.complete(prompt)

    sources = [
        CitedSource(
            citation_id=i,
            citation=format_source_citation(n.node.metadata),
            score=n.score or 0.0,
            snippet=n.node.get_content(),
        )
        for i, n in enumerate(nodes, start=1)
    ]
    return CitedAnswer(question=question, answer=str(response), sources=sources)
