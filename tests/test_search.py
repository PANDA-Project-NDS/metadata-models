"""Tests for search.py module - pure Python, no agent invocation."""

import pytest
from unittest.mock import MagicMock, patch

from pydantic_ai import RunContext

from search import (
    JournalSourcesDeps,
    retrieve_for_pass,
    journal_search,
)
from fixtures.journal_docs import make_node


# --- JournalSourcesDeps tests ---


def test_context_instructions_format(sample_nodes):
    """context_instructions() includes all node text in the returned prompt string."""
    deps = JournalSourcesDeps(
        index=MagicMock(), journal_id="test-journal", context_nodes=sample_nodes
    )
    result = deps.context_instructions()
    assert "$2000 USD" in result
    assert "test-journal" not in result


def test_format_nodes_includes_sources(sample_nodes):
    """format_nodes() wraps each node's text with a --- [Source: <uri>] --- header from the node's metadata."""
    deps = JournalSourcesDeps(
        index=MagicMock(), journal_id="test-journal", context_nodes=sample_nodes
    )
    result = deps.format_nodes()
    assert "--- [Source: fees.html] ---" in result
    assert "--- [Source: about.html] ---" in result
    assert "--- [Source: editorial.html] ---" in result


def test_node_ids_returns_ids(sample_nodes):
    """node_ids() returns a set of all node IDs currently stored in context_nodes."""
    deps = JournalSourcesDeps(
        index=MagicMock(), journal_id="test-journal", context_nodes=sample_nodes
    )
    ids = deps.node_ids()
    assert ids == {"node-1", "node-2", "node-3"}


def test_extend_nodes_deduplicates(sample_nodes):
    """extend_nodes() skips nodes whose ID already exists in context_nodes or seen_node_ids, preventing duplicates."""
    node1 = sample_nodes[0]
    deps = JournalSourcesDeps(
        index=MagicMock(), journal_id="test-journal", context_nodes=[node1]
    )
    added = deps.extend_nodes(sample_nodes[:2])
    assert len(added) == 1
    assert "node-2" in deps.seen_node_ids
    assert deps.node_ids() == {"node-1", "node-2"}


def test_extend_nodes_returns_new_only(sample_nodes):
    """extend_nodes() returns only the nodes that were newly added, excluding those already present."""
    node1 = sample_nodes[0]
    deps = JournalSourcesDeps(
        index=MagicMock(), journal_id="test-journal", context_nodes=[node1]
    )
    added = deps.extend_nodes(sample_nodes[:2])
    assert len(added) == 1
    assert added[0].node.node_id == "node-2"


def test_context_instructions_stays_static(sample_nodes):
    """context_instructions() does not change after extend_nodes — search results only appear in tool responses, not re-injected into context."""
    node1 = sample_nodes[0]
    deps = JournalSourcesDeps(
        index=MagicMock(), journal_id="test-journal", context_nodes=[node1]
    )
    before = deps.context_instructions()
    deps.extend_nodes([sample_nodes[1]])
    after = deps.context_instructions()
    assert before == after
    assert "node-2" not in after


# --- journal_search tool tests ---


@pytest.mark.asyncio
async def test_journal_search_tool_success(mock_retriever, mock_index):
    """Successful search returns formatted node text and tracks new node IDs for deduplication."""
    nodes = [make_node("APC fee is $2500.", node_id="n1", source_uri="fees.html")]
    retriever = mock_retriever({"APC": nodes})
    idx = mock_index(retriever)
    deps = JournalSourcesDeps(index=idx, journal_id="test-journal")

    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    ctx.usage = MagicMock()
    ctx.usage.tool_calls = 0

    result = await journal_search(ctx, "APC fees")
    assert "Found" in result
    assert "$2500" in result
    assert "n1" in deps.seen_node_ids
    assert len(deps.context_nodes) == 0


@pytest.mark.asyncio
async def test_journal_search_tool_no_results(mock_retriever, mock_index):
    """When the retriever returns an empty list, the tool returns a 'No results found' message."""
    retriever = mock_retriever({})
    idx = mock_index(retriever)
    deps = JournalSourcesDeps(index=idx, journal_id="test-journal")

    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    ctx.usage = MagicMock()
    ctx.usage.tool_calls = 0

    result = await journal_search(ctx, "nonexistent query")
    assert "No results found" in result


@pytest.mark.asyncio
async def test_journal_search_tool_max_calls(mock_retriever, mock_index):
    """When usage.tool_calls >= 3, the tool refuses further searches and returns an 'exceeded' message."""
    nodes = [make_node("some result", node_id="n1")]
    retriever = mock_retriever({"anything": nodes})
    idx = mock_index(retriever)
    deps = JournalSourcesDeps(index=idx, journal_id="test-journal")

    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    ctx.usage = MagicMock()
    ctx.usage.tool_calls = 3

    result = await journal_search(ctx, "anything")
    assert "exceeded" in result


@pytest.mark.asyncio
async def test_journal_search_tool_error(mock_retriever, mock_index):
    """When the retriever raises an exception, the tool catches it and returns a 'Search failed' message."""
    retriever = mock_retriever({})
    retriever.retrieve = lambda q: (_ for _ in ()).throw(RuntimeError("db down"))
    idx = mock_index(retriever)
    deps = JournalSourcesDeps(index=idx, journal_id="test-journal")

    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    ctx.usage = MagicMock()
    ctx.usage.tool_calls = 0

    result = await journal_search(ctx, "fail query")
    assert "Search failed" in result


# --- retrieve_for_pass tests ---


def test_retrieve_for_pass_empty():
    """When every query returns no nodes, result.is_empty is True and empty_count equals the number of queries."""
    retriever = MagicMock()
    retriever.retrieve.return_value = []

    with patch("search.build_retriever", return_value=retriever):
        result = retrieve_for_pass(MagicMock(), ["q1", "q2", "q3"], "test-journal")

    assert result.is_empty is True
    assert result.empty_count == 3


def test_retrieve_for_pass_partial_failure():
    """When some queries raise exceptions but others succeed, failure_count is tracked and successful nodes are still returned."""
    nodes_q1 = [make_node("result for q1", node_id="n1")]
    retriever = MagicMock()
    retriever.retrieve.side_effect = [nodes_q1, RuntimeError("fail")]

    with patch("search.build_retriever", return_value=retriever):
        result = retrieve_for_pass(MagicMock(), ["q1", "q2"], "test-journal")

    assert result.failure_count == 1
    assert len(result.nodes) == 1
    assert result.nodes[0].node.node_id == "n1"


def test_retrieve_for_pass_deduplicates():
    """When multiple queries return nodes with the same ID, only one copy appears in the final result."""
    shared = [make_node("shared result", node_id="n-shared")]
    retriever = MagicMock()
    retriever.retrieve.side_effect = [shared, shared]

    with patch("search.build_retriever", return_value=retriever):
        result = retrieve_for_pass(MagicMock(), ["q1", "q2"], "test-journal")

    assert len(result.nodes) == 1
