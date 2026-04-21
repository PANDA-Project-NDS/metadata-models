import os
import logging
from typing import List, Dict
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

from models.journal import JournalMetadata
from agents import (
    basic_info_agent, policies_agent, fees_agent, people_metrics_agent,
    EXTRACTION_QUERIES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LlamaIndex Configuration ---
# Use the recommended local embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
# We handle the LLM via Pydantic AI
Settings.llm = None

def build_journal_index(directory_path: str) -> VectorStoreIndex:
    """Loads documents for a specific journal and builds a vector index."""
    logger.info(f"Building index for: {directory_path}")
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index

def retrieve_for_pass(index: VectorStoreIndex, queries: List[str]) -> List:
    """Performs targeted queries to gather nodes for a specific extraction pass."""
    retriever = index.as_retriever(similarity_top_k=3)
    unique_nodes = {}
    
    for query in queries:
        nodes = retriever.retrieve(query)
        for node in nodes:
            # node is a NodeWithScore object
            unique_nodes[node.node.node_id] = node
            
    return list(unique_nodes.values())

def assemble_context(nodes: List) -> str:
    """Formats the retrieved nodes into a single context string with source citations."""
    context_parts = []
    
    for node in nodes:
        # metadata is in node.node.metadata
        source_file = node.node.metadata.get("file_name", "Unknown Source")
        formatted_chunk = f"--- [Source: {source_file}] ---\n{node.node.text}\n"
        context_parts.append(formatted_chunk)
        
    return "\n".join(context_parts)

async def run_extraction_pass(index: VectorStoreIndex, agent, queries: List[str]):
    """Executes a single extraction pass using specific queries and an agent."""
    logger.info(f"Running extraction pass for queries: {queries[:1]}...")
    nodes = retrieve_for_pass(index, queries)
    context_str = assemble_context(nodes)
    
    prompt = f"Extract metadata using the following retrieved context:\n\n{context_str}"
    
    # Run the Pydantic AI agent
    result = await agent.run(prompt)
    return result.data

async def process_journal(directory_path: str) -> JournalMetadata:
    """End-to-end multi-pass pipeline for a single journal."""
    
    # 1. Build Index
    index = build_journal_index(directory_path)
    
    # 2. Execute Passes concurrently
    import asyncio
    
    logger.info("Starting multi-pass extraction...")
    
    # Run passes
    # Note: We can run these in parallel because they are independent agent calls
    basic_task = run_extraction_pass(index, basic_info_agent, EXTRACTION_QUERIES["basic_info"])
    policy_task = run_extraction_pass(index, policies_agent, EXTRACTION_QUERIES["policies_submissions"])
    fees_task = run_extraction_pass(index, fees_agent, EXTRACTION_QUERIES["fees_membership"])
    people_task = run_extraction_pass(index, people_metrics_agent, EXTRACTION_QUERIES["people_metrics"])
    
    basic_data, policy_data, fees_data, people_data = await asyncio.gather(
        basic_task, policy_task, fees_task, people_task
    )
    
    # 3. Merge into Final Schema (Nested Composition)
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
    from pydantic import RootModel

    # Example: Process Journal Alpha
    journal_path = os.path.abspath("example_docs") # For testing, we'll use all docs
    
    async def main():
        try:
            metadata = await process_journal(journal_path)
            print("\n--- Extracted Metadata ---\n")
            # Use model_dump_json for clean output
            print(metadata.model_dump_json(indent=2))
            
            # Save to file
            with open("extracted_metadata.json", "w") as f:
                f.write(metadata.model_dump_json(indent=2))
                
        except Exception as e:
            logger.error(f"Extraction failed: {e}")

    asyncio.run(main())
