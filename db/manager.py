import logging
import os
from typing import Iterator
from typing import List

from llama_index.core import Document
from pymongo import MongoClient
from pymongo.collection import Collection

from .parsers import _serialize_excel_doc
from .parsers import _serialize_html_doc

logger = logging.getLogger(__name__)


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
            doc = _serialize_html_doc(db_doc, collection_name)
            if doc:
                yield doc

    def stream_excel_documents(
        self, collection_name: str, limit: int = 0
    ) -> Iterator[Document]:
        """Yield Document objects from non-HTML (Excel/APC) documents in MongoDB."""
        collection = self.get_collection(collection_name)
        cursor = collection.find({"metadata.html": {"$exists": False}}).limit(limit)
        for db_doc in cursor:
            yield _serialize_excel_doc(db_doc, collection_name)

    def get_journal_ids(self, publisher: str | None = None) -> List[str]:
        """Get distinct journal IDs from the indexed collection via MongoDB query.
        Queries metadata.journal_id from the vector store nodes. Optionally filters by publisher."""
        collection = self.get_collection(self.index_collection_name)
        filter_query = {"metadata.publisher": publisher} if publisher else {}
        ids = collection.distinct("metadata.journal_id", filter_query)
        ids = [jid for jid in ids if jid and jid != "unknown"]
        logger.info(
            f"Found {len(ids)} journal IDs in collection '{self.index_collection_name}': {ids}"
        )
        return ids

    def init_metadata_index(self, collection_name: str) -> None:
        """Initialize the metadata collection has the necessary indexes."""
        collection = self.get_collection(collection_name)
        collection.create_index("journal_id", unique=True)
        logger.info(
            f"Initialized MongoDB collection '{collection_name}' with unique index on 'journal_id'"
        )

    def save_metadata_one(
        self, collection_name: str, publisher_id: str, journal_id: str, metadata: dict
    ):
        """Save a single journal's metadata for streaming pipeline output."""
        collection = self.get_collection(collection_name)
        collection.replace_one(
            {"journal_id": journal_id},
            {**metadata, "publisher_id": publisher_id, "journal_id": journal_id},
            upsert=True,
        )

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
