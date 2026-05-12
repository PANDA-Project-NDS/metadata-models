import logging
import os

from pymongo import MongoClient
from pymongo.collection import Collection

logger = logging.getLogger(__name__)


class MetadataStore:
    """Manages metadata operations against MongoDB."""

    def __init__(self, client: MongoClient) -> None:
        self.client = client

    def get_collection(self, name: str) -> Collection:
        return self.client.get_database()[name]

    @property
    def metadata_collection(self) -> str:
        return os.getenv("MONGO_METADATA_COLLECTION", "journal_metadata")

    def init_metadata_index(self) -> None:
        """Initialize the metadata collection with the necessary indexes."""
        collection = self.get_collection(self.metadata_collection)
        collection.create_index("journal_id", unique=True)
        logger.info(
            f"Initialized MongoDB collection '{self.metadata_collection}' with unique index on 'journal_id'"
        )

    def save_metadata_one(
        self, publisher_id: str, journal_id: str, metadata: dict
    ) -> None:
        """Save a single journal's metadata for streaming pipeline output."""
        collection = self.get_collection(self.metadata_collection)
        collection.replace_one(
            {"journal_id": journal_id},
            {**metadata, "publisher_id": publisher_id, "journal_id": journal_id},
            upsert=True,
        )
