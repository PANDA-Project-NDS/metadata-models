import itertools
import json
import logging
import os
from typing import Iterator
from typing import List

import trafilatura
import tqdm
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

EXCEL_METADATA_FIELDS = [
    ("Journal", "title"),
    ("Subject", "subject_area"),
    ("ISSN", "online_issn"),
    ("Open access type", "open_access_type"),
    ("License", "license_types_offered"),
    ("APC", "full_price"),
    ("Blocked", "blocked"),
]


def _serialize_excel_doc(db_doc: dict, collection_name: str) -> Document:
    """Serialize a non-HTML (Excel/APC) document into a Document for embedding."""
    m = db_doc.get("metadata", {})
    parts = []
    for label, key in EXCEL_METADATA_FIELDS:
        val = m.get(key)
        if val is not None and val != "":
            parts.append(f"{label}: {json.dumps(val)}")
    header = m.get("header_footer")
    if header:
        parts.append(header)
    return Document(
        text="\n".join(parts),
        metadata={
            "source_uri": m.get("url", "unknown"),
            "journal_id": m.get("title", "unknown"),
            "publisher": collection_name,
            "scope": "excel",
        },
        excluded_embed_metadata_keys=["source_uri", "journal_id", "publisher", "scope"],
    )


def _make_ingestion_pipeline(vector_store: MongoDBAtlasVectorSearch):
    """Create an IngestionPipeline with chunking and embedding transformations."""
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SentenceSplitter
    from transformers import AutoTokenizer

    from embed import get_embed_model

    # set tokenizer to get better approximation of token counts for chunking, based on the embedding model
    embed_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    tokenizer = AutoTokenizer.from_pretrained(embed_model_name)

    return IngestionPipeline(
        transformations=[
            SentenceSplitter(
                # lower chunk_size to account for metadata
                chunk_size=450, chunk_overlap=50, tokenizer=tokenizer.encode
            ),
            get_embed_model(),
        ],
        vector_store=vector_store,
    )


class MongoDBManager:
    """Manages all MongoDB operations for the pipeline."""

    def __init__(self, uri: str) -> None:
        self._uri = uri
        self._client: MongoClient | None = None

    @property
    def client(self) -> MongoClient:
        if self._client is None:
            self._client = MongoClient(self._uri, serverSelectionTimeoutMS=5000)
        return self._client

    def get_collection(self, name: str) -> Collection:
        return self.client.get_database()[name]

    @property
    def index_collection_name(self) -> str:
        return os.getenv("MONGO_INDEX_COLLECTION", "search_index")

    @property
    def db_name(self) -> str:
        return os.getenv("MONGO_DB", "retrieve")

    def load_source_documents(
        self, collection_name: str, limit: int = 0
    ) -> List[Document]:
        """Load source HTML documents from a MongoDB collection."""
        collection = self.get_collection(collection_name)
        query = {"metadata.html": {"$exists": True, "$ne": None}}
        documents = []
        for db_doc in collection.find(query).limit(limit):
            html_content = db_doc.get("metadata", {}).get("html", "")
            if not html_content:
                continue
            extracted_text = trafilatura.extract(html_content)
            if not extracted_text:
                extracted_text = html_content
            metadata = db_doc.get("metadata", {})
            source_url = metadata.get("url", "unknown")
            journal_id = metadata.get("title", "unknown")
            doc = Document(
                text=extracted_text,
                metadata={
                    "source_uri": source_url,
                    "journal_id": journal_id,
                },
            )
            documents.append(doc)
        logger.info(
            f"Loaded {len(documents)} documents from MongoDB collection '{collection_name}'"
        )
        return documents

    def stream_source_documents(
        self, collection_name: str, limit: int = 0
    ) -> Iterator[Document]:
        """Yield Document objects from MongoDB without holding them all in memory.
        Only yields documents that have metadata.html."""
        collection = self.get_collection(collection_name)
        cursor = collection.find(
            {"metadata.html": {"$exists": True, "$ne": None}}
        ).limit(limit)
        for db_doc in cursor:
            html_content = db_doc.get("metadata", {}).get("html", "")
            if not html_content:
                continue
            extracted_text = trafilatura.extract(html_content)
            if not extracted_text:
                extracted_text = html_content
            metadata = db_doc.get("metadata", {})
            source_url = metadata.get("url", "unknown")
            journal_id = metadata.get("title", "unknown")
            yield Document(
                text=extracted_text,
                metadata={
                    "source_uri": source_url,
                    "journal_id": journal_id,
                    "publisher": collection_name,
                    "scope": "html",
                },
                excluded_embed_metadata_keys=[
                    "source_uri",
                    "journal_id",
                    "publisher",
                    "scope",
                ],
            )

    def stream_excel_documents(
        self, collection_name: str, limit: int = 0
    ) -> Iterator[Document]:
        """Yield Document objects from non-HTML (Excel/APC) documents in MongoDB."""
        collection = self.get_collection(collection_name)
        cursor = collection.find({"metadata.html": {"$exists": False}}).limit(limit)
        for db_doc in cursor:
            yield _serialize_excel_doc(db_doc, collection_name)

    def index_documents(
        self,
        collection: str,
        limit: int = 0,
        batch_size: int = 10,
    ) -> VectorStoreIndex:
        """Stream raw documents, chunk, embed, and persist into the search_index collection.
        Processes in batches via IngestionPipeline."""
        vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=self.client,
            db_name=self.db_name,
            collection_name=self.index_collection_name,
            vector_index_name="vector_index",
        )
        vector_store.create_vector_search_index(
            dimensions=int(os.getenv("EMBEDDING_DIM", "384")),
            path="embedding",
            similarity="cosine",
            filters=["metadata.journal_id", "metadata.publisher", "metadata.scope"],
        )
        ingestion = _make_ingestion_pipeline(vector_store)

        src_coll = self.get_collection(collection)
        total_docs = src_coll.count_documents(
            {"metadata.html": {"$exists": True, "$ne": None}}
        ) + src_coll.count_documents({"metadata.html": {"$exists": False}})
        if limit:
            total_docs = min(total_docs, limit * 2)

        doc_iter = itertools.chain(
            self.stream_source_documents(collection, limit),
            self.stream_excel_documents(collection, limit),
        )
        batch: List[Document] = []
        indexed = 0
        with tqdm.tqdm(
            total=total_docs,
            desc=f"Indexing to '{self.index_collection_name}'",
        ) as pbar:
            for doc in doc_iter:
                batch.append(doc)
                if len(batch) >= batch_size:
                    ingestion.run(documents=batch)
                    indexed += len(batch)
                    pbar.update(len(batch))
                    batch = []
            if batch:
                ingestion.run(documents=batch)
                indexed += len(batch)
                pbar.update(len(batch))

        logger.info(
            f"Indexing complete. {indexed} documents from '{collection}' processed into '{self.index_collection_name}'."
        )
        return VectorStoreIndex.from_vector_store(vector_store)

    def load_vector_index(self) -> VectorStoreIndex:
        """Load a pre-existing vector index from MongoDB. No document embedding."""
        vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=self.client,
            db_name=self.db_name,
            collection_name=self.index_collection_name,
            vector_index_name="vector_index",
        )
        return VectorStoreIndex.from_vector_store(vector_store)

    def get_journal_ids(self) -> List[str]:
        """Get distinct journal IDs from the indexed collection via MongoDB query.
        Queries metadata.journal_id from the vector store nodes."""
        collection = self.get_collection(self.index_collection_name)
        ids = collection.distinct("metadata.journal_id")
        ids = [jid for jid in ids if jid and jid != "unknown"]
        logger.info(
            f"Found {len(ids)} journal IDs in collection '{self.index_collection_name}': {ids}"
        )
        return ids

    def save_metadata(self, collection_name: str, results: dict) -> int:
        """Save extracted journal metadata to MongoDB. Returns count saved."""
        collection = self.get_collection(collection_name)
        collection.create_index("journal_id", unique=True)
        saved = 0
        for journal_id, metadata in results.items():
            collection.replace_one(
                {"journal_id": journal_id},
                {**metadata, "journal_id": journal_id},
                upsert=True,
            )
            saved += 1
        logger.info(
            f"Saved {saved} journal metadata documents to MongoDB collection '{collection_name}'"
        )
        return saved

    def save_metadata_one(self, collection_name: str, journal_id: str, metadata: dict):
        """Save a single journal's metadata for streaming pipeline output."""
        collection = self.get_collection(collection_name)
        collection.create_index("journal_id", unique=True)
        collection.replace_one(
            {"journal_id": journal_id},
            {**metadata, "journal_id": journal_id},
            upsert=True,
        )

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
