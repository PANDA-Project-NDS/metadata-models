#!/usr/bin/env python3
import logging
import os
from typing import List

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic_ai import ModelSettings

from agents import (
    basic_info_agent, policies_agent, fees_agent, people_metrics_agent,
    EXTRACTION_QUERIES
)
from models.journal import JournalMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LlamaIndex Configuration ---
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
Settings.llm = None  # We handle the LLM via Pydantic AI


def build_global_index(directory_path: str) -> VectorStoreIndex:
    """Loads all documents, injects journal_id metadata, and builds a global vector index."""
    logger.info(f"Building global index from: {directory_path}")
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    documents = SimpleDirectoryReader(directory_path).load_data()

    # Inject journal_id metadata for filtering
    for doc in documents:
        filename = os.path.basename(doc.metadata.get("file_path", ""))
        # Heuristic: journal_alpha_apc.html -> journal_alpha
        parts = filename.split('_')
        if len(parts) >= 2:
            journal_id = f"{parts[0]}_{parts[1]}"
        else:
            journal_id = "unknown"
        doc.metadata["journal_id"] = journal_id
        logger.info(f"Assigned journal_id '{journal_id}' to {filename}")

    index = VectorStoreIndex.from_documents(documents)
    return index


def retrieve_for_pass(index: VectorStoreIndex, queries: List[str], journal_id: str, top_k: int = 2) -> List:
    """Performs targeted queries filtered by a specific journal_id."""
    # Apply metadata filter to restrict search to the target journal
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="journal_id", value=journal_id)]
    )
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)

    unique_nodes = {}
    for query in queries:
        nodes = retriever.retrieve(query)
        for node in nodes:
            unique_nodes[node.node.node_id] = node

    return list(unique_nodes.values())


def assemble_context(nodes: List) -> str:
    """Formats the retrieved nodes into a single context string with source citations."""
    context_parts = []

    for node in nodes:
        source_file = node.node.metadata.get("file_name", "Unknown Source")
        formatted_chunk = f"--- [Source: {source_file}] ---\n{node.node.text}\n"
        context_parts.append(formatted_chunk)

    return "\n".join(context_parts)


async def run_extraction_pass(index: VectorStoreIndex, agent, queries: List[str], journal_id: str):
    """Executes a single extraction pass using specific queries, filtered by journal_id."""
    logger.info(f"[{journal_id}] Running extraction pass for queries: {queries[:1]}...")
    nodes = retrieve_for_pass(index, queries, journal_id)

    if not nodes:
        logger.warning(f"[{journal_id}] No nodes retrieved for queries: {queries[:1]}")

    context_str = assemble_context(nodes)

    prompt = f"Extract metadata using the following retrieved context:\n\n{context_str}"

    result = await agent.run(prompt, model_settings=ModelSettings(timeout=600))
    return result.data


async def process_journal(index: VectorStoreIndex, journal_id: str) -> JournalMetadata:
    """End-to-end multi-pass pipeline for a single journal ID using the global index."""
    import asyncio

    logger.info(f"Starting multi-pass extraction for {journal_id}...")

    basic_task = run_extraction_pass(index, basic_info_agent, EXTRACTION_QUERIES["basic_info"], journal_id)
    policy_task = run_extraction_pass(index, policies_agent, EXTRACTION_QUERIES["policies_submissions"], journal_id)
    fees_task = run_extraction_pass(index, fees_agent, EXTRACTION_QUERIES["fees_membership"], journal_id)
    people_task = run_extraction_pass(index, people_metrics_agent, EXTRACTION_QUERIES["people_metrics"], journal_id)

    basic_data, policy_data, fees_data, people_data = await asyncio.gather(
        basic_task, policy_task, fees_task, people_task
    )

    # Merge into Final Schema
    final_metadata = JournalMetadata(
        basic_info=basic_data,
        policies=policy_data,
        fees=fees_data,
        people_metrics=people_data
    )

    return final_metadata


if __name__ == "__main__":
    import asyncio
    import json

    journal_path = os.path.abspath("example_docs")


    async def main():
        try:
            # 1. Build global index with metadata
            global_index = build_global_index(journal_path)

            # Extract unique journal IDs from the documents
            all_journal_ids = set(
                doc.metadata.get("journal_id")
                for doc in global_index.docstore.docs.values()
            )
            all_journal_ids.discard("unknown")

            logger.info(f"Found journal IDs to process: {all_journal_ids}")

            # 2. Process each journal isolated by metadata filter
            results = {}
            for j_id in all_journal_ids:
                metadata = await process_journal(global_index, j_id)
                results[j_id] = json.loads(metadata.model_dump_json())

            # Save all results
            with open("extracted_metadata.json", "w") as f:
                json.dump(results, f, indent=2)

            logger.info("Extraction complete. Results saved to extracted_metadata.json")

        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)


    asyncio.run(main())
