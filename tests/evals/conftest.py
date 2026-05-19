import os
from unittest.mock import MagicMock

import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore

from pydantic_ai import models
from pydantic_ai.models import create_async_http_client
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

models.ALLOW_MODEL_REQUESTS = True


@pytest.fixture(autouse=True)
def skip_if_no_eval_run():
    """Skip all eval tests unless EVAL_RUN env var is set."""
    if not os.getenv("EVAL_RUN"):
        pytest.skip("Set EVAL_RUN=1 to run eval tests")


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
def eval_model():
    """OpenAI-compatible model from EVAL_MODEL or falls back to production OPENAI_MODEL."""
    model_name = os.getenv("EVAL_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_url = os.getenv("OPENAI_API_URL")
    timeout = int(os.getenv("OPENAI_HTTP_TIMEOUT", "120"))

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            api_url,
            http_client=create_async_http_client(timeout=timeout),
        ),
    )
