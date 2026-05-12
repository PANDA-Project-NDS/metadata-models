#!/usr/bin/env python3
"""Index raw documents into MongoDB vector store."""

import argparse
import logging

from dotenv import load_dotenv

from db import DocumentStore, Indexer, mongo_connection

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
    parser.add_argument(
        "--collection",
        required=True,
        help="Source collection name",
    )
    args = parser.parse_args()

    collection_name = args.collection

    with mongo_connection() as client:
        store = DocumentStore(client)
        if args.clear_embeddings:
            result = store.get_collection(collection_name).update_many(
                {}, {"$unset": {"embedding": ""}}
            )
            logger.info(
                f"Cleared embeddings from {result.modified_count} documents in '{collection_name}'"
            )
        else:
            if args.clear:
                result = store.get_collection(store.index_collection_name).delete_many(
                    {"metadata.publisher": collection_name}
                )
                logger.info(
                    f"Removed {result.deleted_count} existing documents for publisher '{collection_name}' from '{store.index_collection_name}'"
                )

            indexer = Indexer(store)
            indexer.index_documents(
                collection_name,
                limit=args.limit,
                batch_size=args.batch_size,
            )


if __name__ == "__main__":
    main()
