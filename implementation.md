# Journal Metadata Extraction System - Implementation Plan

## General Overview

This system extracts structured metadata (APC costs, Editors, Impact Factors, etc.) from heterogeneous journal documents (HTML, PDF, Excel) using a Retrieval-Augmented Generation (RAG) pipeline. 

The architecture uses a **Multi-Pass Retrieve-then-Extract** pattern to handle long documents and small context windows (e.g., local 1.7B models):
1. **LlamaIndex** handles ingestion, chunking, local embedding, and targeted retrieval.
2. **Pydantic AI** handles the strict, type-safe extraction of metadata using a local LLM in four distinct passes (Basic Info, Policies, Fees, People). This ensures each extracted data point includes verbatim evidence and source file citations without overwhelming the model's context window.

### Tech Stack
*   **Frameworks:** LlamaIndex (RAG), Pydantic AI (Agentic Extraction)
*   **Embeddings:** `BAAI/bge-small-en-v1.5` (Local HuggingFace model, optimized for semantic retrieval)
*   **LLM:** `qwen/qwen3-1.7b` (Local execution via LM-Studio OpenAI-compatible endpoint)

---

## Design Decisions

### 1. Handling Heterogeneous Input Documents
The system anticipates varied input formats (HTML, PDF, Excel) containing the journal data. Instead of writing custom parsers for each format, we leverage **LlamaIndex's `SimpleDirectoryReader`** and its ecosystem of data connectors. These connectors abstract away the parsing complexity, reducing all documents to raw text chunks that are easily searchable via the RAG pipeline.

### 2. Multi-Pass Retrieve-then-Extract RAG Architecture
*   **Retrieve-then-Extract vs Agentic Search Tool:** We chose not to give the LLM a "search tool" because smaller models (like Qwen 1.7B) struggle with complex reasoning loops and tool calling.
*   **Multi-Pass:** Instead of a single massive search that might overwhelm the context window (causing "lost in the middle" hallucinations), we run four separate extraction passes (Basic Info, Policies, Fees, People). Each pass performs highly targeted semantic queries to retrieve only the most relevant chunks for that specific schema subset.

### 3. Local-First with Easy Cloud Swap
The system is built entirely on open-source, local-first tools (HuggingFace embeddings, LM-Studio for the LLM). This ensures zero data privacy risks and no API costs during development. Because Pydantic AI natively uses the OpenAI schema, swapping from local `qwen3-1.7b` to a cloud model like `gpt-4o` in production requires changing just one string in `agents.py`.

### 4. Strict Evidence Tracking (Provenance)
To combat LLM hallucination and build human trust in the system, every extracted value is wrapped in a generic `Sourced[T]` Pydantic model. 
*   **`quote`:** The LLM is forced to extract the verbatim text proving its claim.
*   **`source`:** The context chunks are artificially injected with `[Source: filename]` headers, and the LLM must cite which chunk provided the evidence.

### 5. Pydantic Model Composition
To adhere to DRY (Don't Repeat Yourself) principles, we defined granular, modular domain blocks (e.g., `JournalIdentity`, `Pricing`) in `models/journal.py`. These blocks are composed into the four target extraction schemas used by the agents, and finally nested into the root `JournalMetadata` schema. This prevents duplicate field definitions while keeping concerns strictly separated.

---

## Step 1: Data Ingestion and Indexing

External systems provide heterogeneous documents. We use LlamaIndex to load these documents, inject a `journal_id` into the metadata, split them into manageable chunks (e.g., 512 tokens), and embed them into a global VectorStore using a strong local embedding model.

```python
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Configure local embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
Settings.llm = None # We use Pydantic AI for the LLM part

def build_global_index(directory_path: str) -> VectorStoreIndex:
    """Loads all documents, injects journal_id metadata, and builds a global vector index."""
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    # Inject journal_id metadata for filtering
    for doc in documents:
        filename = os.path.basename(doc.metadata.get("file_path", ""))
        parts = filename.split('_')
        journal_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "unknown"
        doc.metadata["journal_id"] = journal_id

    return VectorStoreIndex.from_documents(documents)
```

## Step 2: Targeted Multi-Pass Retrieval

Instead of relying on a single broad search, we perform targeted semantic searches for specific groups of metadata fields across four passes. We also apply a `MetadataFilter` to restrict the search strictly to documents matching the current `journal_id`.

```python
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# Defined in agents.py
EXTRACTION_QUERIES = {
    "basic_info": [
        "Journal title, publisher, about this journal, mission, scope, sections",
        "ISSN, print ISSN, online ISSN, indexed in, abstracting and indexing databases"
    ],
    # ... more queries for policies, fees and people
}

def retrieve_for_pass(index: VectorStoreIndex, queries: list[str], journal_id: str) -> list:
    """Performs targeted queries filtered by a specific journal_id."""
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="journal_id", value=journal_id)]
    )
    retriever = index.as_retriever(similarity_top_k=3, filters=filters)
    
    unique_nodes = {}
    for query in queries:
        nodes = retriever.retrieve(query)
        for node in nodes:
            unique_nodes[node.node.node_id] = node
            
    return list(unique_nodes.values())
```

## Step 3: Context Assembly with Source Metadata

To enable evidence tracking, we must inject the source file name/URI into the text context so the LLM can reference it during extraction.

```python
def assemble_context(nodes: list) -> str:
    """Formats the retrieved nodes into a single context string with source citations."""
    context_parts = []
    
    for node in nodes:
        source_file = node.metadata.get("file_name", "Unknown Source")
        formatted_chunk = f"--- [Source: {source_file}] ---\n{node.text}\n"
        context_parts.append(formatted_chunk)
        
    return "\n".join(context_parts)
```

## Step 4: Multi-Pass Structured Extraction (Pydantic AI)

We define multiple partial Pydantic models (e.g., `BasicInfoExtraction`, `FeesAndMembershipExtraction`) that represent subsets of the full schema. Pydantic AI connects directly to the local LM-Studio endpoint, and separate agents handle each pass, scoped by `journal_id`.

```python
from agents import (
    basic_info_agent, policies_agent, fees_agent, people_metrics_agent,
    EXTRACTION_QUERIES
)
from models.journal import JournalMetadata

async def run_extraction_pass(index: VectorStoreIndex, agent, queries: list[str], journal_id: str):
    """Executes a single extraction pass using specific queries, filtered by journal_id."""
    nodes = retrieve_for_pass(index, queries, journal_id)
    context_str = assemble_context(nodes)
    
    prompt = f"Extract metadata using the following retrieved context:\n\n{context_str}"
    result = await agent.run(prompt)
    return result.data
```

## Step 5: Final Execution Pipeline

The final step ties the multi-pass retrieval and extraction together. It finds all unique `journal_id`s in the index, executes the extraction passes for each independently, and merges the partial results into a full `JournalMetadata` object.

```python
import asyncio

async def process_journal(index: VectorStoreIndex, journal_id: str) -> JournalMetadata:
    """End-to-end multi-pass pipeline for a single journal ID using the global index."""
    
    # 1. Execute Passes Concurrently
    basic_task = run_extraction_pass(index, basic_info_agent, EXTRACTION_QUERIES["basic_info"], journal_id)
    policy_task = run_extraction_pass(index, policies_agent, EXTRACTION_QUERIES["policies_submissions"], journal_id)
    fees_task = run_extraction_pass(index, fees_agent, EXTRACTION_QUERIES["fees_membership"], journal_id)
    people_task = run_extraction_pass(index, people_metrics_agent, EXTRACTION_QUERIES["people_metrics"], journal_id)
    
    basic_data, policy_data, fees_data, people_data = await asyncio.gather(
        basic_task, policy_task, fees_task, people_task
    )
    
    # 2. Merge into Final Schema (Nested Composition)
    final_metadata = JournalMetadata(
        basic_info=basic_data,
        policies=policy_data,
        fees=fees_data,
        people_metrics=people_data
    )
    
    return final_metadata

# Main Loop over all journals
# global_index = build_global_index("example_docs")
# for j_id in unique_journal_ids:
#     metadata = await process_journal(global_index, j_id)
```
