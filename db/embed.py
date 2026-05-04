import os

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()


class OpenAIEmbeddingQueryPrefix(OpenAIEmbedding):
    """OpenAIEmbedding that prefixes queries with query instructions.

    BGE Models need the following prefix for example:
    "Represent this sentence for searching relevant passages: "
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._query_instruction = self._resolve_instruction()

    def _resolve_instruction(self) -> str:
        from llama_index.embeddings.huggingface.utils import (
            get_query_instruct_for_model_name,
        )

        for name in (self.model_name, f"BAAI/{self.model_name}"):
            inst = get_query_instruct_for_model_name(name)
            if inst:
                return inst
        return ""

    def _get_query_embedding(self, query: str) -> list[float]:
        prefixed = f"{self._query_instruction} {query}".strip()
        return super()._get_query_embedding(prefixed)


def get_embed_model():
    """Return an embedding model — OpenAI API if OPENAI_EMBED_MODEL is set, else local HuggingFace."""
    if os.getenv("OPENAI_EMBED_MODEL"):
        return OpenAIEmbeddingQueryPrefix(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_URL"),
            model_name=os.getenv("OPENAI_EMBED_MODEL"),
        )
    return HuggingFaceEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
