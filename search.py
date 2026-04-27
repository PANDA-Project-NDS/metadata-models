import logging
from dataclasses import dataclass, field
from typing import List

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from pydantic_ai import RunContext

logger = logging.getLogger(__name__)


@dataclass
class JournalSearchDeps:
    """Runtime dependencies for the journal search tool."""

    index: BaseIndex
    journal_id: str
    existing_node_ids: set[str] = field(default_factory=set)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation with metadata about its success."""

    nodes: list[NodeWithScore]
    empty_count: int
    failure_count: int

    @property
    def is_empty(self) -> bool:
        return not self.nodes

    @property
    def is_complete(self) -> bool:
        return self.failure_count == 0


def retrieve_for_pass(
        index: object,
        queries: List[str],
        journal_id: str,
        top_k: int = 2,
) -> RetrievalResult:
    """Performs targeted queries filtered by journal_id, with per-query resilience."""
    retriever = build_retriever(index, journal_id, top_k)

    unique_nodes: dict = {}
    empty_count = 0
    failure_count = 0

    for query in queries:
        try:
            nodes = retriever.retrieve(query)
            if not nodes:
                empty_count += 1
            for node in nodes:
                unique_nodes[node.node.node_id] = node
        except Exception as e:
            failure_count += 1
            logger.warning(f"retrieve_for_pass: query threw — {e}")

    return RetrievalResult(
        nodes=list(unique_nodes.values()),
        empty_count=empty_count,
        failure_count=failure_count,
    )


def assemble_context(nodes: list[NodeWithScore]) -> str:
    """Formats the retrieved nodes into a single context string with source citations."""
    context_parts = []

    for node in nodes:
        source = node.node.metadata.get("source_uri", "Unknown Source")
        formatted_chunk = f"--- [Source: {source}] ---\n{node.node.text}\n"
        context_parts.append(formatted_chunk)

    return "\n".join(context_parts)


def build_retriever(index: BaseIndex, journal_id: str, top_k: int = 2) -> BaseRetriever:
    """Build a LlamaIndex retriever scoped to journal_id using MetadataFilters."""
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="journal_id", value=journal_id)]
    )
    return index.as_retriever(similarity_top_k=top_k, filters=filters)


async def journal_search(ctx: RunContext[JournalSearchDeps], query: str) -> str:
    """Search index for additional context scoped to the current journal.
    Ignores results that were already in context.

    Args:
        query: The search query.
    """
    index = ctx.deps.index
    journal_id = ctx.deps.journal_id

    logger.info(
        f"Agent invoked journal_search with query: '{query}' for journal_id: '{journal_id}'"
    )
    try:
        results = build_retriever(index, journal_id).retrieve(query)
        if not results:
            return "No results found for this query."
        filtered_nodes = [n for n in results if n.node.node_id not in ctx.deps.existing_node_ids]
        if not filtered_nodes:
            return "Query gave no new results."
        return assemble_context(filtered_nodes)
    except Exception:
        logger.exception("retrieval tool failed for query: %s", query)
        return "Search failed internally due to an error."
