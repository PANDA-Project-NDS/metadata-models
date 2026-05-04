import itertools
import logging
import os

import tqdm
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from .embed import get_embed_model
from .manager import MongoDBManager

logger = logging.getLogger(__name__)


def _make_ingestion_pipeline(vector_store: MongoDBAtlasVectorSearch):
    """Create an IngestionPipeline with chunking and embedding transformations."""
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SentenceSplitter
    from transformers import AutoTokenizer

    # set tokenizer to get better approximation of token counts for chunking, based on the embedding model
    embed_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    tokenizer = AutoTokenizer.from_pretrained(embed_model_name)

    return IngestionPipeline(
        transformations=[
            SentenceSplitter(
                # lower chunk_size to account for metadata
                chunk_size=450,
                chunk_overlap=50,
                tokenizer=tokenizer.encode,
            ),
            get_embed_model(),
        ],
        vector_store=vector_store,
    )


class Indexer:
    """Orchestrates vector indexing of documents stored in MongoDB."""

    def __init__(self, db_manager: MongoDBManager) -> None:
        self._db_manager = db_manager

    def index_documents(
        self,
        collection: str,
        limit: int = 0,
        batch_size: int = 10,
    ) -> VectorStoreIndex:
        """Stream raw documents, chunk, embed, and persist into the search_index collection.
        Processes in batches via IngestionPipeline."""
        vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=self._db_manager.client,
            db_name=self._db_manager.db_name,
            collection_name=self._db_manager.index_collection_name,
            vector_index_name="vector_index",
        )
        vector_store.create_vector_search_index(
            dimensions=int(os.getenv("EMBEDDING_DIM", "384")),
            path="embedding",
            similarity="cosine",
            filters=["metadata.journal_id", "metadata.publisher", "metadata.scope"],
        )
        ingestion = _make_ingestion_pipeline(vector_store)

        src_coll = self._db_manager.get_collection(collection)
        total_docs = src_coll.count_documents(
            {"metadata.html": {"$exists": True, "$ne": None}}
        ) + src_coll.count_documents({"metadata.html": {"$exists": False}})
        if limit:
            total_docs = min(total_docs, limit * 2)

        doc_iter = itertools.chain(
            self._db_manager.stream_source_documents(collection, limit),
            self._db_manager.stream_excel_documents(collection, limit),
        )

        with tqdm.tqdm(
            total=total_docs,
            desc=f"Indexing to '{self._db_manager.index_collection_name}'",
        ) as pbar:
            for batch in itertools.batched(doc_iter, batch_size):
                ingestion.run(documents=list(batch))
                pbar.update(len(batch))
            logger.info(
                f"Indexing complete. {pbar.n} documents from '{collection}' processed into '{self._db_manager.index_collection_name}'."
            )
        return VectorStoreIndex.from_vector_store(vector_store)

    def load_vector_index(self) -> VectorStoreIndex:
        """Load a pre-existing vector index from MongoDB. No document embedding."""
        vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=self._db_manager.client,
            db_name=self._db_manager.db_name,
            collection_name=self._db_manager.index_collection_name,
            vector_index_name="vector_index",
        )
        return VectorStoreIndex.from_vector_store(vector_store)
