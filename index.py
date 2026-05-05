#!/usr/bin/env python3
"""Index raw documents into MongoDB vector store."""

import argparse
import logging
import os

from db import Indexer
from db import MongoDBManager

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Index documents into MongoDB vector store"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of documents to index"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for ingestion pipeline"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove existing documents for this publisher from the index before re-indexing",
    )
    parser.add_argument(
        "--clear-embeddings",
        action="store_true",
        help="Remove embedding field from source collection after indexing",
    )
    args = parser.parse_args()

    collection_name = os.environ.get("MONGO_COLLECTION", "wiley_full")

    db = MongoDBManager(os.environ["MONGODB_URI"])
    try:
        if args.clear_embeddings:
            result = db.get_collection(collection_name).update_many(
                {}, {"$unset": {"embedding": ""}}
            )
            logger.info(
                f"Cleared embeddings from {result.modified_count} documents in '{collection_name}'"
            )
        else:
            if args.clear:
                result = db.get_collection(db.index_collection_name).delete_many(
                    {"metadata.publisher": collection_name}
                )
                logger.info(
                    f"Removed {result.deleted_count} existing documents for publisher '{collection_name}' from '{db.index_collection_name}'"
                )

            indexer = Indexer(db)
            indexer.index_documents(
                collection_name,
                limit=args.limit,
                batch_size=args.batch_size,
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
