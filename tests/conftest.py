from unittest.mock import MagicMock

import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore

from pydantic_ai import models

models.ALLOW_MODEL_REQUESTS = False

from fixtures.journal_docs import make_node


@pytest.fixture
def mock_retriever():
    def _make_retriever(nodes_map: dict[str, list[NodeWithScore]]):
        retriever = MagicMock(spec=BaseRetriever)

        def retrieve(query: str) -> list[NodeWithScore]:
            for prefix, nodes in nodes_map.items():
                if prefix.lower() in query.lower():
                    return nodes
            return []

        retriever.retrieve = retrieve
        return retriever

    return _make_retriever


@pytest.fixture
def mock_index():
    def _make_index(retriever):
        idx = MagicMock(spec=VectorStoreIndex)
        idx.as_retriever.return_value = retriever
        return idx

    return _make_index


@pytest.fixture
def sample_nodes():
    return [
        make_node(
            "The APC is $2000 USD for all article types.",
            node_id="node-1",
            source_uri="fees.html",
        ),
        make_node(
            "Journal of Testing publishes monthly with 12 issues per year.",
            node_id="node-2",
            source_uri="about.html",
        ),
        make_node(
            "Editor-in-Chief: Dr. Jane Smith, MIT.",
            node_id="node-3",
            source_uri="editorial.html",
        ),
    ]
