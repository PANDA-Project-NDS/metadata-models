#!/usr/bin/env python3
"""Index raw documents into MongoDB vector store."""

import argparse
import logging
import os

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
        "--force-rebuild",
        action="store_true",
        help="Drop existing index collection before re-indexing",
    )
    parser.add_argument(
        "--clear-embeddings",
        action="store_true",
        help="Remove embedding field from source collection after indexing",
    )
    args = parser.parse_args()

    collection_name = os.environ.get("MONGO_COLLECTION", "wiley_full")

    client = MongoDBManager(os.environ["MONGODB_URI"])
    try:
        if args.clear_embeddings:
            result = client.get_collection(collection_name).update_many(
                {}, {"$unset": {"embedding": ""}}
            )
            logger.info(
                f"Cleared embeddings from {result.modified_count} documents in '{collection_name}'"
            )
        else:
            if args.force_rebuild:
                client.get_collection(client.index_collection_name).drop()
                logger.info(
                    f"Dropped existing collection '{client.index_collection_name}'"
                )

            client.index_documents(
                collection_name,
                limit=args.limit,
                batch_size=args.batch_size,
            )

            journal_ids = client.get_journal_ids()
            logger.info(f"Indexed {len(journal_ids)} journals: {journal_ids}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
