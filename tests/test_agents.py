"""Tests for agents.py - agent construction and TestModel smoke tests."""

from unittest.mock import MagicMock

from agents import PASSES, make_agent
from models.journal import (
    BasicInfoExtraction,
    EditorialExtraction,
    FeesExtraction,
    PoliciesExtraction,
)
from pydantic_ai.models.test import TestModel
from search import JournalSourcesDeps


def test_pass_configs_count():
    """There are extraction passes defined."""
    assert len(PASSES) > 0


def test_pass_configs_output_types():
    """Each pass has the expected output type (BasicInfoExtraction, PoliciesExtraction, FeesExtraction, EditorialExtraction)."""
    assert PASSES[0].output_type == BasicInfoExtraction
    assert PASSES[1].output_type == PoliciesExtraction
    assert PASSES[2].output_type == FeesExtraction
    assert PASSES[3].output_type == EditorialExtraction


def test_pass_configs_have_queries():
    """Each pass has a non-empty list of search queries."""
    for pass_config in PASSES:
        assert len(pass_config.queries) > 0


async def test_make_agent_has_search_tool():
    """Agent built via make_agent has the 'Journal Search' tool registered."""
    agent = make_agent(PASSES[0])
    model = TestModel()
    with agent.override(model=model):
        deps = JournalSourcesDeps(index=MagicMock(), journal_id="test")
        await agent.run(deps=deps)
    tools = model.last_model_request_parameters.function_tools
    tool_names = [t.name for t in tools]
    assert "Journal Search" in tool_names


def test_make_agent_output_type():
    """Agent output_type matches the corresponding pass config output_type."""
    for pass_config in PASSES:
        agent = make_agent(pass_config)
        assert agent.output_type == pass_config.output_type


def test_make_agent_deps_type():
    """Agent deps_type is JournalSourcesDeps."""
    agent = make_agent(PASSES[0])
    assert agent.deps_type is JournalSourcesDeps


def test_make_agent_system_prompt_contains_fallback():
    """Agent system prompt includes search fallback instructions and no-hallucination rule."""
    agent = make_agent(PASSES[0])
    system_prompt = "".join(agent._system_prompts)
    assert "Journal Search" in system_prompt
    assert "NO HALLUCINATION" in system_prompt


async def test_testmodel_smoke_all_agents(mock_index, mock_retriever):
    """Each agent runs end-to-end with TestModel, calling tools and producing schema-valid output."""
    retriever = mock_retriever({})
    idx = mock_index(retriever)

    for pass_config in PASSES:
        agent = make_agent(pass_config)
        model = TestModel()
        deps = JournalSourcesDeps(index=idx, journal_id="test-journal")
        with agent.override(model=model):
            result = await agent.run(deps=deps)
        assert isinstance(result.output, pass_config.output_type)
