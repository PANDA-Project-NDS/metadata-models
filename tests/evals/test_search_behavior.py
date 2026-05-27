from unittest.mock import MagicMock

import pytest
from fixtures.journal_docs import make_node
from llama_index.core.base.base_retriever import BaseRetriever
from pydantic_ai import RunContext, Tool, capture_run_messages
from pydantic_ai.messages import ToolCallPart

from agents import PASSES, make_agent
from search import JournalSourcesDeps, journal_search
from tests.evals.fixtures import (
    APC_DOUBLE_CONTEXT,
    EDITORIAL_CONTEXT,
    IRRELEVANT_CONTEXT,
    ISSN_CONTEXT,
    is_empty,
)


def build_deps(index, journal_id="eval-journal", context_nodes=None):
    return JournalSourcesDeps(
        index=index, journal_id=journal_id, context_nodes=context_nodes or []
    )


def count_search_calls(messages):
    return len(
        [
            part
            for msg in messages
            for part in (msg.parts if hasattr(msg, "parts") else [])
            if isinstance(part, ToolCallPart) and part.tool_name == "Journal Search"
        ]
    )


def make_limited_search_tool(max_calls=3):
    """Wrap journal_search to fail fast on exceeding max_calls."""
    call_count = [0]

    async def limited_search(ctx: RunContext[JournalSourcesDeps], query: str) -> str:
        call_count[0] += 1
        if call_count[0] > max_calls:
            raise AssertionError(
                f"Agent called Journal Search {call_count[0]} times (max {max_calls}). "
                "Agent should stop after being told searches are exceeded."
            )
        return await journal_search(ctx, query)

    return Tool(limited_search, name="Journal Search", max_retries=1)


async def test_search_retries_on_irrelevant_result(
    subtests: pytest.Subtests, eval_model, mock_index, mock_retriever
):
    """When first search returns irrelevant data, agent retries and extracts ISSN on second call."""
    nodes = [
        make_node(IRRELEVANT_CONTEXT, node_id="irrelevant-1", source_uri="home.html")
    ]
    irrelevant_search_node = make_node(
        "This is a general page about publishing.",
        node_id="irrelevant-search",
        source_uri="general.html",
    )
    issn_search_node = make_node(
        ISSN_CONTEXT, node_id="issn-search", source_uri="about.html"
    )

    call_count = [0]

    def retrieve(query: str):
        call_count[0] += 1
        if call_count[0] == 1:
            return [irrelevant_search_node]
        return [issn_search_node]

    retriever = MagicMock(spec=BaseRetriever)
    retriever.retrieve = retrieve
    idx = mock_index(retriever)

    limited_tool = make_limited_search_tool(max_calls=3)
    agent = make_agent(PASSES[0], tools=[limited_tool])
    deps = build_deps(idx, context_nodes=nodes)
    with capture_run_messages() as messages:
        with agent.override(model=eval_model):
            result = await agent.run(deps=deps)

    tool_calls = count_search_calls(messages)
    assert tool_calls >= 1, "Agent did not call Journal Search tool"

    with subtests.test("search retry count"):
        assert 1 < tool_calls <= 3

    with subtests.test("issn extracted"):
        assert result.output.issn is not None
        assert result.output.issn.print.value == "1234-5678"


async def test_search_returns_nothing_returns_empty(
    eval_model, mock_index, mock_retriever
):
    """When both context and search return nothing relevant, agent returns empty fields — no hallucination."""
    nodes = [
        make_node(IRRELEVANT_CONTEXT, node_id="irrelevant-1", source_uri="home.html")
    ]
    retriever = mock_retriever({})
    idx = mock_index(retriever)

    limited_tool = make_limited_search_tool(max_calls=3)
    agent = make_agent(PASSES[0], tools=[limited_tool])
    deps = build_deps(idx, context_nodes=nodes)
    with capture_run_messages() as messages:
        with agent.override(model=eval_model):
            result = await agent.run(deps=deps)

    tool_calls = count_search_calls(messages)
    assert tool_calls >= 1, "Agent did not call Journal Search tool"

    assert result.output.title is None
    assert is_empty(result.output.issn)
    assert result.output.scope is None
    assert is_empty(result.output.facts)
    assert is_empty(result.output.metrics)


async def test_search_respects_max_calls(
    subtests: pytest.Subtests, eval_model, mock_index, mock_retriever
):
    """Agent stops searching after hitting the call limit, does not hallucinate data."""
    nodes = [
        make_node(IRRELEVANT_CONTEXT, node_id="irrelevant-1", source_uri="home.html")
    ]
    always_irrelevant = make_node(
        "This is a general page about publishing.",
        node_id="always-irrelevant",
        source_uri="general.html",
    )

    call_count = [0]

    def retrieve(query: str):
        call_count[0] += 1
        return [always_irrelevant]

    retriever = MagicMock(spec=BaseRetriever)
    retriever.retrieve = retrieve
    idx = mock_index(retriever)

    limited_tool = make_limited_search_tool(max_calls=3)
    agent = make_agent(PASSES[0], tools=[limited_tool])
    deps = build_deps(idx, context_nodes=nodes)
    with capture_run_messages() as messages:
        with agent.override(model=eval_model):
            result = await agent.run(deps=deps)

    tool_calls = count_search_calls(messages)
    assert tool_calls >= 1, "Agent did not call Journal Search tool"

    with subtests.test("search within limit"):
        assert tool_calls <= 5

    with subtests.test("no hallucination"):
        assert is_empty(result.output.issn)
        assert result.output.title is None


async def test_search_finds_issn_from_irrelevant_context(
    eval_model, mock_index, mock_retriever
):
    """When initial context is irrelevant, Info Agent calls journal_search and extracts ISSN from search results."""
    nodes = [
        make_node(IRRELEVANT_CONTEXT, node_id="irrelevant-1", source_uri="home.html")
    ]
    issn_nodes = [
        make_node(ISSN_CONTEXT, node_id="issn-search-1", source_uri="about.html")
    ]
    retriever = mock_retriever(
        {"ISSN": issn_nodes, "issn": issn_nodes, "publisher": issn_nodes}
    )
    idx = mock_index(retriever)

    limited_tool = make_limited_search_tool(max_calls=3)
    agent = make_agent(PASSES[0], tools=[limited_tool])
    deps = build_deps(idx, context_nodes=nodes)
    with capture_run_messages() as messages:
        with agent.override(model=eval_model):
            result = await agent.run(deps=deps)

    tool_calls = count_search_calls(messages)
    assert tool_calls >= 1, "Agent did not call Journal Search tool"

    assert result.output.issn is not None
    assert result.output.issn.print.value == "1234-5678"


async def test_search_finds_apc_from_irrelevant_context(
    subtests: pytest.Subtests, eval_model, mock_index, mock_retriever
):
    """Fees Agent finds APC data via search when initial context is irrelevant, then extracts it."""
    nodes = [
        make_node(EDITORIAL_CONTEXT, node_id="editorial-1", source_uri="editorial.html")
    ]
    apc_nodes = [
        make_node(APC_DOUBLE_CONTEXT, node_id="apc-search-1", source_uri="fees.html")
    ]
    retriever = mock_retriever(
        {
            "APC": apc_nodes,
            "apc": apc_nodes,
            "fee": apc_nodes,
            "cost": apc_nodes,
            "charge": apc_nodes,
        }
    )
    idx = mock_index(retriever)

    limited_tool = make_limited_search_tool(max_calls=3)
    agent = make_agent(PASSES[2], tools=[limited_tool])
    deps = build_deps(idx, context_nodes=nodes)
    with capture_run_messages() as messages:
        with agent.override(model=eval_model):
            result = await agent.run(deps=deps)

    tool_calls = count_search_calls(messages)
    assert tool_calls >= 1, "Agent did not call Journal Search tool"

    assert result.output.pricing is not None

    with subtests.test("has APCs"):
        assert len(result.output.pricing.article_processing_charges) >= 1

    with subtests.test("contains $2000 USD"):
        apcs = result.output.pricing.article_processing_charges
        apc = next(
            (a for a in apcs if a.fee.value == 2000 and a.fee.currency == "USD"), None
        )
        assert apc is not None
