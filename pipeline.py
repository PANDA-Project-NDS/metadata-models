#!/usr/bin/env python3
import argparse
import logging
import os
from typing import List

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from pydantic import ValidationError, BaseModel
from pydantic_ai.exceptions import UnexpectedModelBehavior
from tqdm import tqdm

from agents import PASSES, make_agent
from db import get_embed_model
from models.journal import JournalMetadata
from search import retrieve_for_pass, JournalSourcesDeps

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LlamaIndex Configuration ---
# Query-time embedding model (required for retrieval)
Settings.embed_model = get_embed_model()
Settings.llm = None  # We handle the LLM via Pydantic AI


async def run_extraction_pass(
    index: VectorStoreIndex, agent, queries: List[str], journal_id: str
) -> BaseModel:
    """Executes a single extraction pass using specific queries, filtered by journal_id."""
    logger.info(f"[{journal_id}] Running extraction pass for queries: {queries[:1]}...")
    retrieval = retrieve_for_pass(index, queries, journal_id)

    if retrieval.is_empty:
        if retrieval.failure_count == len(queries):
            logger.error(f"[{journal_id}] All queries threw before returning results")
        elif retrieval.empty_count == len(queries):
            logger.warning(
                f"[{journal_id}] All queries returned empty — no matching docs"
            )
        else:
            logger.warning(
                f"[{journal_id}] Retrieval partially failed — {retrieval.empty_count} empty, {retrieval.failure_count} errors"
            )
    elif retrieval.failure_count > 0:
        logger.warning(
            f"[{journal_id}] Retrieval partially failed — {retrieval.failure_count}/{len(queries)} queries errored"
        )

    deps = JournalSourcesDeps(
        index=index, journal_id=journal_id, context_nodes=retrieval.nodes
    )
    try:
        result = await agent.run(deps=deps)
        return result.output
    except ValidationError as e:
        logger.error(f"[{journal_id}] Validation error during extraction: {e}")
    except UnexpectedModelBehavior as e:
        logger.error(f"[{journal_id}] Unexpected model behavior during extraction: {e}")
    except Exception as e:
        logger.error(f"[{journal_id}] Unexpected error during extraction: {e}")

    # Fallback to an empty instance of the expected schema
    logger.warning(f"[{journal_id}] Returning empty fallback for failed extraction.")
    # pydantic_ai stores the expected output schema in agent.result_type
    return agent.output_type()


async def process_journal(index: VectorStoreIndex, journal_id: str) -> JournalMetadata:
    """End-to-end multi-pass pipeline for a single journal ID using the global index."""
    import asyncio

    logger.info(f"Starting multi-pass extraction for {journal_id}...")

    agents = [make_agent(p) for p in PASSES]
    tasks = [
        run_extraction_pass(index, agent, p.queries, journal_id)
        for agent, p in zip(agents, PASSES)
    ]
    results = await asyncio.gather(*tasks)

    # Merge into Final Schema
    final_metadata = JournalMetadata()
    for result in results:
        final_metadata.__dict__.update(result.model_dump())

    return final_metadata


if __name__ == "__main__":
    import asyncio
    import json

    from db import Indexer
    from db import MongoDBManager

    async def main():
        parser = argparse.ArgumentParser(
            description="Run extraction pipeline for a publisher"
        )
        parser.add_argument("--publisher", required=True, help="Publisher to process")
        args = parser.parse_args()

        try:
            db = MongoDBManager(os.environ["MONGODB_URI"])
            try:
                # Load pre-indexed vector store (no document embedding)
                indexer = Indexer(db)
                global_index = indexer.load_vector_index()

                # Query journal IDs from indexed collection, filtered by publisher
                all_journal_ids = db.get_journal_ids(publisher=args.publisher)
                logger.info(f"Processing {len(all_journal_ids)} journals")

                # Stream: process and save each journal immediately
                metadata_collection = os.environ.get(
                    "MONGO_METADATA_COLLECTION", "journal_metadata"
                )
                db.init_metadata_index(metadata_collection)
                for j_id in tqdm(all_journal_ids, desc="Processing journals"):
                    metadata = await process_journal(global_index, j_id)
                    db.save_metadata_one(
                        collection_name=metadata_collection,
                        journal_id=j_id,
                        publisher_id=args.publisher,
                        metadata=json.loads(metadata.model_dump_json()),
                    )
                    logger.info(f"Saved metadata for {j_id}")

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)

    asyncio.run(main())
