#!/usr/bin/env python3
import logging
import os
from typing import List

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic_ai import ModelSettings
import trafilatura
from llama_index.core import Document
from pydantic import ValidationError
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
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
Settings.llm = None  # We handle the LLM via Pydantic AI


def load_with_trafilatura(directory_path: str) -> List[Document]:
    """Loads HTML documents from a directory and extracts core text using Trafilatura."""
    documents = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith((".html", ".htm")):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_html = f.read()

                # Extract main content, fallback to raw HTML if trafilatura fails
                extracted_text = trafilatura.extract(raw_html)
                if not extracted_text:
                    logger.warning(
                        f"Trafilatura failed to extract text from {file_path}, using raw text."
                    )
                    extracted_text = raw_html

                doc = Document(
                    text=extracted_text,
                    metadata={"file_path": file_path, "file_name": file},
                )
                documents.append(doc)
    return documents


def build_global_index(directory_path: str) -> VectorStoreIndex:
    """Loads all documents, injects journal_id metadata, and builds a global vector index."""
    logger.info(f"Building global index from: {directory_path}")
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Use Trafilatura for better content extraction
    documents = load_with_trafilatura(directory_path)

    # Inject journal_id metadata for filtering
    for doc in documents:
        filename = os.path.basename(doc.metadata.get("file_path", ""))
        # Heuristic: journal_alpha_apc.html -> journal_alpha
        parts = filename.split("_")
        if len(parts) >= 2:
            journal_id = f"{parts[0]}_{parts[1]}"
        else:
            journal_id = "unknown"
        doc.metadata["journal_id"] = journal_id
        logger.info(f"Assigned journal_id '{journal_id}' to {filename}")

    index = VectorStoreIndex.from_documents(documents)
    return index


async def run_extraction_pass(
    index: VectorStoreIndex, agent, queries: List[str], journal_id: str
):
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

    prompt = f"Extract metadata using the following retrieved context:\n\n{context_str}"

    try:
        deps = JournalSearchDeps(index=index, journal_id=journal_id)
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
    return agent.result_type()


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
            for j_id in list(all_journal_ids)[:5]:
                metadata = await process_journal(global_index, j_id)
                results[j_id] = json.loads(metadata.model_dump_json())

            # Save all results
            with open("extracted_metadata.json", "w") as f:
                json.dump(results, f, indent=2)

            logger.info("Extraction complete. Results saved to extracted_metadata.json")

        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)

    asyncio.run(main())
