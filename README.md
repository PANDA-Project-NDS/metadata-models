# APC Pipeline

RAG-based metadata extraction pipeline for academic journals. Extracts structured metadata (APC costs, editors, impact factors, ISSN, policies, etc.) from heterogeneous journal documents (HTML, Excel) using a multi-pass, agentic architecture.

## Architecture

Four specialized Pydantic AI agents run concurrently, each handling a disjoint subset of the `JournalMetadata` schema. Each agent performs targeted semantic retrieval from a MongoDB Atlas vector store, then extracts structured data from the retrieved context. An agentic search fallback allows agents to re-query the index when initial retrieval misses relevant chunks.

```
Raw Documents (MongoDB)
    │
    ▼
┌─────────────┐
│  index.py   │  Chunk, embed, persist to MongoDB Atlas vector store
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  pipeline.py  —  process_journal(journal_id)            │
│                                                         │
│  ┌──────────────┐ ┌──────────────┐                      │
│  │ Info Agent   │ │ Policies     │   (asyncio.gather)   │
│  │ (BasicInfo)  │ │ Agent        │                      │
│  └──────┬───────┘ └──────┬───────┘                      │
│         │                │                              │
│  ┌──────┴───────┐ ┌──────┴───────┐                      │
│  │ Fees Agent   │ │ Editors Agent│                      │
│  │ (Fees+Member)│ │ (Editorial)  │                      │
│  └──────┬───────┘ └──────┬───────┘                      │
│         └──────┬─────────┘                              │
│                ▼                                        │
│         JournalMetadata (merged)                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
MongoDB (journal_metadata collection)
```

### Multi-Pass Design

| Pass | Agent | Schema | Queries |
|------|-------|--------|---------|
| 1 | Info Agent | `BasicInfoExtraction` | Title, ISSN, scope, metrics |
| 2 | Policies Agent | `PoliciesExtraction` | Frequency, submission, review, languages, diamond OA |
| 3 | Fees Agent | `FeesExtraction` | APC, waivers, discounts, membership |
| 4 | Editors Agent | `EditorialExtraction` | Editorial board |

Multi-pass prevents context overloading and avoids "lost in the middle" degradation that occurs with single-pass broad retrieval across many chunked documents.

### Agentic Search Fallback

Each agent carries a `journal_search` tool. When the initial retrieved context lacks evidence for a required field, the agent formulates its own search query to find additional chunks. The tool is scoped to the current journal via `journal_id` metadata filters, with deduplication and a 2-call limit per agent.

### Evidence Tracking

Every extracted value is wrapped in `SourcedValue[T]`, which pairs the value with verbatim `quote` and `source` fields. This provides full provenance for audit and quality assurance.

## Tech Stack

- **RAG**: LlamaIndex (`VectorStoreIndex`, `MongoDBAtlasVectorSearch`, `IngestionPipeline`)
- **Agents**: Pydantic AI (tool calling, structured output, dependency injection)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (HuggingFace, local) or OpenAI-compatible API
- **LLM**: Any OpenAI-compatible model with tool calling (OpenAI, Ollama, OpenRouter, etc.)
- **Vector Store**: MongoDB Atlas with persistent vector search index
- **HTML Parsing**: Trafilatura (core content extraction)
- **Observability**: Logfire (Pydantic AI instrumentation)

## Project Structure

```
├── pipeline.py          # Main extraction orchestrator
├── agents.py            # Agent factory, pass configs, system prompts
├── search.py            # Retrieval, JournalSourcesDeps, journal_search tool
├── index.py             # Standalone indexing script
├── db/                  # MongoDB operations package
│   ├── __init__.py      # mongo_connection, exports
│   ├── documents.py     # DocumentStore (streaming, journal IDs)
│   ├── metadata.py      # MetadataStore (save, index init)
│   ├── indexer.py       # Indexer (ingestion pipeline, vector index)
│   ├── embed.py         # Embedding model factory
│   └── parsers.py       # Pure document serialization (HTML, Excel)
├── models/              # Pydantic schema definitions
│   ├── journal.py       # JournalMetadata, extraction schemas, SourcedValue
│   └── vocab.py         # Literal type definitions (currencies, review types, etc.)
├── tests/               # pytest test suite
│   ├── test_agents.py   # Agent construction, TestModel smoke tests
│   ├── test_search.py   # Retrieval, JournalSourcesDeps, journal_search tool
│   └── evals/           # Evaluation harness
├── scripts/             # Utility scripts
│   ├── field_coverage.py
│   ├── extract_coars.py
├   └── compare_fees.py  # APC fee comparison utility against panter data
├── docker/              # MongoDB Atlas Local setup
└── implementation/      # Design documents and decision records
```

## Setup

### Prerequisites

- Python >= 3.13
- uv (package manager)
- MongoDB Atlas (or MongoDB Atlas Local via `docker/`)

### Install

```bash
uv sync
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MONGODB_URI` | MongoDB connection string | (required) |
| `MONGO_DB` | Database name | `retrieve` |
| `MONGO_INDEX_COLLECTION` | Vector index collection | `search_index` |
| `MONGO_METADATA_COLLECTION` | Extraction results collection | `journal_metadata` |
| `EMBEDDING_MODEL` | Embedding model name | `BAAI/bge-small-en-v1.5` |
| `EMBEDDING_DIM` | Embedding vector dimensions | `384` |
| `OPENAI_MODEL` | LLM model name | (required) |
| `OPENAI_API_URL` | LLM API base URL | (required) |
| `OPENAI_API_KEY` | LLM API key | — |
| `OPENAI_EMBED_MODEL` | If set, use OpenAI-compatible embeddings | — |
| `OPENAI_TEMPERATURE` | LLM temperature | `0.0` |
| `OPENAI_HTTP_TIMEOUT` | LLM HTTP timeout (seconds) | `60` |

### MongoDB Atlas Local

```bash
docker compose -f docker/docker-compose.atlas.yaml up -d
```

## Usage

The pipeline assumes raw journal documents are already stored in MongoDB collections (one per publisher).

### 1. Index Documents

Embed raw documents into the MongoDB vector store. Run once when documents are added or updated.

```bash
# Index all documents from a publisher collection
python index.py --collection wiley

# Index a subset for testing
python index.py --collection wiley --limit 5

# Clear existing index before re-indexing
python index.py --collection wiley --clear

# Clear embeddings from source collection
python index.py --collection wiley --clear-embeddings
```

### 2. Run Extraction Pipeline

Process journals and extract structured metadata.

```bash
python pipeline.py --publisher wiley
```

The pipeline loads the pre-indexed vector store, discovers journal IDs, runs all four extraction passes concurrently per journal, and saves results to the `journal_metadata` collection.

## Testing

```bash
# Run basic tests without agent invocation (e.g., agent construction, retrieval, tool calls)
uv run pytest

# Run evaluation tests with LLM models
EVAL_RUN=1 uv run pytest tests/evals/
```


## Adding a New Extraction Pass

Add a `PassConfig` entry to `PASSES` in `agents.py`:

```python
PassConfig(
    "My Agent",
    MyExtractionSchema,
    [
        "query one targeting the information I need",
        "query two for additional coverage",
    ],
    domain_guidelines="## DOMAIN RULES\n- My specific rules",
)
```

The pipeline discovers passes automatically — no other code changes needed.
