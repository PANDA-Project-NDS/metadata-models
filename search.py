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
    seen_node_ids: set[str] = field(default_factory=set)

    def __post_init__(self, *args, **kwargs):
        # add context nodes into seen nodes set
        self.seen_node_ids = set(node.node.node_id for node in self.context_nodes)

    def context_instructions(self) -> str:
        return f"Extract metadata from a Journal using the following retrieved context:\n\n{self.format_nodes()}"

    def format_nodes(self, nodes: list[NodeWithScore] | None = None) -> str:
        """Formats the retrieved nodes into a single context string with source citations.

        Args:
            nodes: Nodes to format. Formats all nodes present in `context_nodes` by default.
        """
        target = nodes if nodes is not None else self.context_nodes
        parts = []
        for node in target:
            source = node.node.metadata.get("source_uri", "Unknown Source")
            parts.append(f"--- [Source: {source}] ---\n{node.node.text}\n")
        return "\n".join(parts)

    def extend_nodes(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """Tracks new nodes for deduplication. Does NOT mutate context_nodes.
        Returns the list of new nodes added to tracking."""
        added = [n for n in nodes if n.node.node_id not in self.seen_node_ids]
        self.seen_node_ids.update(n.node.node_id for n in added)
        return added


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


def build_retriever(index: BaseIndex, journal_id: str, top_k: int = 3) -> BaseRetriever:
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
            return "Search completed: No new information found to add to context."

        return f"Found {len(new_nodes)} results:\n\n{ctx.deps.format_nodes(new_nodes)}"

    except Exception:
        logger.exception("retrieval tool failed for query: %s", query)
        return "Search failed internally due to an error."
