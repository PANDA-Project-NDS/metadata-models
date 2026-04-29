#!/usr/bin/env python3
"""Index raw documents into MongoDB vector store."""

import argparse
import logging
import os

from db import MongoDBManager

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
    args = parser.parse_args()

    input_collection = os.environ.get("MONGO_COLLECTION", "wiley_full")
    output_collection = os.environ.get(
        "MONGO_INDEX_COLLECTION", f"{input_collection}_index"
    )

    client = MongoDBManager(os.environ["MONGO_URI"])
    try:
        if args.force_rebuild:
            client.get_collection(output_collection).drop()
            logger.info(f"Dropped existing collection '{output_collection}'")

        client.index_documents(
            input_collection,
            output_collection,
            limit=args.limit,
            batch_size=args.batch_size,
        )

        journal_ids = client.get_journal_ids(output_collection)
        logger.info(f"Indexed {len(journal_ids)} journals: {journal_ids}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
