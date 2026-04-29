import logging
from typing import List

import trafilatura
from llama_index.core import Document
from pymongo import MongoClient
from pymongo.collection import Collection

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

    def load_source_documents(self, collection_name: str, limit: int = 0) -> List[Document]:
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
                    "file_path": source_url,
                    "source_uri": source_url,
                    "journal_id": journal_id,
                },
            )
            documents.append(doc)
        logger.info(
            f"Loaded {len(documents)} documents from MongoDB collection '{collection_name}'"
        )
        return documents

    def save_metadata(self, collection_name: str, results: dict) -> int:
        """Save extracted journal metadata to MongoDB. Returns count saved."""
        collection = self.get_collection(collection_name)
        collection.create_index("journal_id", unique=True)
        saved = 0
        for journal_id, metadata in results.items():
            collection.replace_one({"journal_id": journal_id}, {"journal_id": journal_id, **metadata}, upsert=True)
            saved += 1
        logger.info(
            f"Saved {saved} journal metadata documents to MongoDB collection '{collection_name}'"
        )
        return saved

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
