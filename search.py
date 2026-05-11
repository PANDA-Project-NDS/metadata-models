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
class JournalSourcesDeps:
    """Agent runtime dependencies for the journal context"""

    index: BaseIndex
    journal_id: str
    context_nodes: list[NodeWithScore] = field(default_factory=list)

    def context_instructions(self) -> str:
        return f"Extract metadata from Journal '{self.journal_id}' using the following retrieved context:\n\n{self.format_nodes()}"

    def format_nodes(self, nodes: list[NodeWithScore] | None = None) -> str:
        """Formats the retrieved nodes into a single context string with source citations.

        Args:
            nodes: Nodes to format. Formats all nodes present in `context_nodes` by default.
        """
        context_parts = []

        if nodes is None:
            # nodes not passed, fallback to all
            nodes = self.context_nodes

        for node in nodes:
            source = node.node.metadata.get("source_uri", "Unknown Source")
            formatted_chunk = f"--- [Source: {source}] ---\n{node.node.text}\n"
            context_parts.append(formatted_chunk)

        return "\n".join(context_parts)

    def node_ids(self) -> set[str]:
        """Returns the set of node IDs currently in context."""
        return set(n.node.node_id for n in self.context_nodes)

    def extend_nodes(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """Extends the provided nodes to the context, ignoring any that are already present. Returns the list of new nodes added."""
        new_nodes = list(filter(lambda n: n.node.node_id not in self.node_ids(), nodes))
        self.context_nodes.extend(new_nodes)
        return new_nodes


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


def build_retriever(index: BaseIndex, journal_id: str, top_k: int = 2) -> BaseRetriever:
    """Build a LlamaIndex retriever scoped to journal_id using MetadataFilters."""
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="journal_id", value=journal_id)]
    )
    return index.as_retriever(similarity_top_k=top_k, filters=filters)


async def journal_search(ctx: RunContext[JournalSourcesDeps], query: str) -> str:
    """Search index for additional context scoped to the current journal.
    Ignores results that were already in context.

    Args:
        ctx: tool context
        query: The search query.
    """
    index = ctx.deps.index
    journal_id = ctx.deps.journal_id

    logger.info(
        f"Agent invoked journal_search with query: '{query}' for journal_id: '{journal_id}'"
    )
    if ctx.usage.tool_calls >= 2:
        return "Number of allowed searches exceeded. Please form the output."
    try:
        results = build_retriever(index, journal_id).retrieve(query)
        if not results:
            return "No results found for this query."
        new_nodes = ctx.deps.extend_nodes(results)
        if not new_nodes:
            return "Query gave no new results."
        return ctx.deps.format_nodes(new_nodes)
    except Exception:
        logger.exception("retrieval tool failed for query: %s", query)
        return "Search failed internally due to an error."
