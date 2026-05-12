import logging
import os
from typing import Iterator

from llama_index.core import Document
from pymongo import MongoClient
from pymongo.collection import Collection

from .parsers import _serialize_excel_doc
from .parsers import _serialize_html_doc

logger = logging.getLogger(__name__)


class DocumentStore:
    """Manages document streaming operations against MongoDB."""

    def __init__(self, client: MongoClient) -> None:
        self.client = client

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

    def get_journal_ids(self, publisher: str | None = None) -> list[str]:
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
