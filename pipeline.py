#!/usr/bin/env python3
import logging
import os
from typing import List

import trafilatura
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import ValidationError, BaseModel
from pydantic_ai import ModelSettings
from pydantic_ai.exceptions import UnexpectedModelBehavior

from agents import (
    basic_info_agent,
    policies_agent,
    fees_agent,
    people_agent,
    EXTRACTION_QUERIES,
)
from models.journal import JournalMetadata
from search import assemble_context, retrieve_for_pass, JournalSearchDeps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LlamaIndex Configuration ---
Settings.embed_model = HuggingFaceEmbedding(
    model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
)
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
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

    context_str = assemble_context(retrieval.nodes)
    existing_ids = set(n.node.node_id for n in retrieval.nodes)

    prompt = f"Extract metadata from Journal '{journal_id}' using the following retrieved context:\n\n{context_str}"

    try:
        deps = JournalSearchDeps(
            index=index, journal_id=journal_id, existing_node_ids=existing_ids
        )
        result = await agent.run(
            prompt, deps=deps, model_settings=ModelSettings(timeout=300)
        )
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
    return agent.output_type


async def process_journal(index: VectorStoreIndex, journal_id: str) -> JournalMetadata:
    """End-to-end multi-pass pipeline for a single journal ID using the global index."""
    import asyncio

    logger.info(f"Starting multi-pass extraction for {journal_id}...")

    basic_task = run_extraction_pass(
        index, basic_info_agent, EXTRACTION_QUERIES["basic_info"], journal_id
    )
    policy_task = run_extraction_pass(
        index, policies_agent, EXTRACTION_QUERIES["policies_submissions"], journal_id
    )
    fees_task = run_extraction_pass(
        index, fees_agent, EXTRACTION_QUERIES["fees_membership"], journal_id
    )
    people_task = run_extraction_pass(
        index, people_agent, EXTRACTION_QUERIES["people_metrics"], journal_id
    )

    basic_data, policy_data, fees_data, people_data = await asyncio.gather(
        basic_task, policy_task, fees_task, people_task
    )

    # Merge into Final Schema
    final_metadata = JournalMetadata(
        **basic_data.model_dump(),
        **policy_data.model_dump(),
        **fees_data.model_dump(),
        **people_data.model_dump(),
    )

    return final_metadata


if __name__ == "__main__":
    import asyncio
    import json

    from db import MongoDBManager

    data_source = os.environ.get("DATA_SOURCE", "mongodb")

    async def main():
        try:
            client = MongoDBManager(os.environ["MONGO_URI"])
            try:
                docs = client.load_source_documents(
                    os.environ.get("MONGO_COLLECTION", "wiley_full"), limit=10
                )
                global_index = VectorStoreIndex.from_documents(docs)

                # Extract unique journal IDs from the documents
                all_journal_ids = set(
                    doc.metadata.get("journal_id")
                    for doc in global_index.docstore.docs.values()
                )
                all_journal_ids.discard("unknown")

                logger.info(f"Found journal IDs to process: {all_journal_ids}")

                # Process each journal isolated by metadata filter
                results = {}
                for j_id in all_journal_ids:
                    metadata = await process_journal(global_index, j_id)
                    results[j_id] = json.loads(metadata.model_dump_json())

                # Save all results to MongoDB
                client.save_metadata(
                    os.environ.get("MONGO_METADATA_COLLECTION", "journal_metadata"),
                    results,
                )
            finally:
                client.close()

        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)

    asyncio.run(main())
