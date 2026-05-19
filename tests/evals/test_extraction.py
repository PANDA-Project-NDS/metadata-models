from unittest.mock import MagicMock

import pytest
from fixtures.journal_docs import make_node

from agents import PASSES, make_agent
from search import JournalSourcesDeps
from tests.evals.fixtures import (
    APC_DOUBLE_CONTEXT,
    APC_SINGLE_CONTEXT,
    EDITORIAL_CONTEXT,
    ISSN_BOTH_CONTEXT,
    ISSN_CONTEXT,
    ISSN_PRINT_ONLY_CONTEXT,
    POLICIES_CONTEXT,
    is_empty,
    IRRELEVANT_CONTEXT,
)


def build_deps(index, journal_id="eval-journal", context_nodes=None):
    return JournalSourcesDeps(
        index=index, journal_id=journal_id, context_nodes=context_nodes or []
    )


async def run_pass(pass_index, eval_model, idx, context_nodes):
    """Helper: build agent for a pass (no search), run with eval_model, return result.output."""
    agent = make_agent(PASSES[pass_index], include_search=False)
    deps = build_deps(idx, context_nodes=context_nodes)
    with agent.override(model=eval_model):
        result = await agent.run(deps=deps)
    return result.output


async def test_extract_print_issn(eval_model, mock_index, mock_retriever):
    """Info Agent extracts print ISSN value from context containing only print ISSN."""
    nodes = [
        make_node(ISSN_PRINT_ONLY_CONTEXT, node_id="issn-1", source_uri="about.html")
    ]
    retriever = mock_retriever({})
    idx = mock_index(retriever)
    output = await run_pass(0, eval_model, idx, nodes)

    assert output.issn is not None
    assert output.issn.print.value == "1234-5678"


async def test_extract_issn_evidence(eval_model, mock_index, mock_retriever):
    """Info Agent provides verbatim quote and source for extracted ISSN."""
    nodes = [
        make_node(ISSN_PRINT_ONLY_CONTEXT, node_id="issn-1", source_uri="about.html")
    ]
    retriever = mock_retriever({})
    idx = mock_index(retriever)
    output = await run_pass(0, eval_model, idx, nodes)

    assert output.issn.print.evidence is not None
    assert output.issn.print.evidence.quote
    assert output.issn.print.evidence.source == "about.html"


async def test_extract_both_issns(eval_model, mock_index, mock_retriever):
    """Info Agent extracts both print and online ISSN when both are present in context."""
    nodes = [make_node(ISSN_BOTH_CONTEXT, node_id="issn-1", source_uri="about.html")]
    retriever = mock_retriever({})
    idx = mock_index(retriever)
    output = await run_pass(0, eval_model, idx, nodes)

    assert output.issn.print.value == "1234-5678"
    assert output.issn.online.value == "9876-5432"


async def test_extract_single_apc(eval_model, mock_index, mock_retriever):
    """Fees Agent extracts APC fee value and currency from context with a single APC."""
    nodes = [make_node(APC_SINGLE_CONTEXT, node_id="apc-1", source_uri="fees.html")]
    retriever = mock_retriever({})
    idx = mock_index(retriever)
    output = await run_pass(2, eval_model, idx, nodes)

    assert output.pricing is not None
    assert len(output.pricing.article_processing_charges) >= 1
    assert output.pricing.article_processing_charges[0].fee.value == 2000
    assert output.pricing.article_processing_charges[0].fee.currency == "USD"


async def test_extract_apc_evidence(eval_model, mock_index, mock_retriever):
    """Fees Agent provides verbatim quote and source for extracted APC."""
    nodes = [make_node(APC_SINGLE_CONTEXT, node_id="apc-1", source_uri="fees.html")]
    retriever = mock_retriever({})
    idx = mock_index(retriever)
    output = await run_pass(2, eval_model, idx, nodes)

    apc = output.pricing.article_processing_charges[0]
    assert apc.evidence is not None
    assert apc.evidence.quote
    assert apc.evidence.source == "fees.html"


async def test_extract_multiple_apcs(eval_model, mock_index, mock_retriever):
    """Fees Agent extracts both APC values when context lists multiple fees."""
    nodes = [make_node(APC_DOUBLE_CONTEXT, node_id="apc-1", source_uri="fees.html")]
    retriever = mock_retriever({})
    idx = mock_index(retriever)
    output = await run_pass(2, eval_model, idx, nodes)

    assert len(output.pricing.article_processing_charges) >= 2
    values = [a.fee.value for a in output.pricing.article_processing_charges]
    assert 2000 in values
    assert 1500 in values


async def test_schema_valid_all_passes(eval_model, mock_index, mock_retriever):
    """All 4 passes produce schema-valid output when context contains data for every field type."""
    nodes = [
        make_node(ISSN_CONTEXT, node_id="n1", source_uri="about.html"),
        make_node(APC_DOUBLE_CONTEXT, node_id="n2", source_uri="fees.html"),
        make_node(EDITORIAL_CONTEXT, node_id="n3", source_uri="editorial.html"),
        make_node(POLICIES_CONTEXT, node_id="n4", source_uri="policies.html"),
    ]
    retriever = mock_retriever({})
    idx = mock_index(retriever)

    from models.journal import (
        BasicInfoExtraction,
        EditorialExtraction,
        FeesExtraction,
        PoliciesExtraction,
    )

    expected_types = [
        BasicInfoExtraction,
        PoliciesExtraction,
        FeesExtraction,
        EditorialExtraction,
    ]

    for i, pass_config in enumerate(PASSES):
        output = await run_pass(i, eval_model, idx, nodes)
        assert isinstance(output, expected_types[i]), (
            f"Pass {i} ({pass_config.name}) output type mismatch"
        )


async def test_empty_context_returns_empty(eval_model, mock_index, mock_retriever):
    """Info Agent returns all fields null/empty when context is empty — no hallucination."""
    retriever = mock_retriever({})
    idx = mock_index(retriever)
    output = await run_pass(0, eval_model, idx, [])

    assert output.title is None
    assert is_empty(output.issn)
    assert output.scope is None
    assert is_empty(output.facts)
    assert is_empty(output.metrics)

async def test_irrelevant_context_returns_empty(
        eval_model, mock_index, mock_retriever
    ):
        """Info Agent returns all fields null/empty when context is irrelevant"""
        retriever = mock_retriever({})
        idx = mock_index(retriever)
        nodes = [
            make_node(IRRELEVANT_CONTEXT, node_id="irrelevant-1", source_uri="home.html")
        ]
        output = await run_pass(
            0,
            eval_model,
            idx,
            nodes
        )

        assert output.title is None
        assert is_empty(output.issn)
        assert output.scope is None
        assert is_empty(output.facts)
        assert is_empty(output.metrics)


async def test_only_editors_extracted(eval_model, mock_index, mock_retriever):
    """When only editorial context is provided, Editors pass populates data while other passes return empty fields."""
    nodes = [
        make_node(
            EDITORIAL_CONTEXT, node_id="editorial-only", source_uri="editorial.html"
        )
    ]
    retriever = mock_retriever({})
    idx = mock_index(retriever)

    from models.journal import (
        BasicInfoExtraction,
        EditorialExtraction,
        FeesExtraction,
        PoliciesExtraction,
    )

    info = await run_pass(0, eval_model, idx, nodes)
    assert isinstance(info, BasicInfoExtraction)
    assert info.title is None
    assert is_empty(info.issn)

    policy = await run_pass(1, eval_model, idx, nodes)
    assert isinstance(policy, PoliciesExtraction)
    assert policy.publication_frequency is None

    fees = await run_pass(2, eval_model, idx, nodes)
    assert isinstance(fees, FeesExtraction)
    assert is_empty(fees.pricing)

    editors = await run_pass(3, eval_model, idx, nodes)
    assert isinstance(editors, EditorialExtraction)
    assert len(editors.editors) >= 1
    editor_names = [e.name for e in editors.editors if e.name]
    assert len(editor_names) >= 1
