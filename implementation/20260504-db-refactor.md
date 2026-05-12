# DB Refactor Plan

## Motivation
Currently, `db.py` is taking on three heavily disparate responsibilities:
1. **Database I/O:** Connecting to MongoDB, managing collections, and reading/writing raw dictionaries.
2. **Data Parsing:** Using `trafilatura` to extract HTML, mapping Excel fields, and constructing LlamaIndex `Document` objects.
3. **Vector/Embedding Orchestration:** Building the LlamaIndex `IngestionPipeline`, configuring the HuggingFace tokenizer, and running the batch ingestion loop.

Furthermore, `embed.py` sits in the project root but contains tightly coupled ingestion and embedding configurations. Consolidating all of this into a cohesive `db` package cleans up the root directory and establishes a clear boundary for data storage, retrieval, and vectorization concerns.

## Benefits
* **Separation of Concerns:** Isolates PyMongo connection logic from text extraction and vector chunking logic.
* **Maintainability:** Modifying chunking strategies or swapping out the HTML parser (e.g., replacing `trafilatura`) only requires touching isolated modules, reducing the risk of breaking database interactions.
* **Testability:** Pure data transformations (`parsers.py`) become trivial to unit test without needing to mock MongoDB clients.
* **Simpler Imports:** External scripts (`index.py`, `pipeline.py`) can rely on a clean facade via `db/__init__.py`.

## Package Structure
```text
db/
├── __init__.py      # Exposes MongoDBManager, Indexer, and get_embed_model
├── parsers.py       # Pure data transformation (_serialize_html_doc, _serialize_excel_doc, EXCEL_METADATA_FIELDS)
├── embed.py         # (Moved from root) get_embed_model, OpenAIEmbeddingQueryPrefix
├── manager.py       # MongoDBManager — connection, collection access, document streaming, metadata persistence
└── indexer.py       # Indexer — vector indexing orchestration (index_documents, load_vector_index, _make_ingestion_pipeline)
```

## Module Boundaries
* **`parsers.py`** — Pure transformations. No DB, no LlamaIndex pipeline. Depends on `trafilatura` and `llama_index.core.Document`.
* **`embed.py`** — Embedding model factory. Used by both `indexer.py` and `pipeline.py`.
* **`manager.py`** — Raw DB access. Holds `MongoClient`, collection helpers, env config (`db_name`, `index_collection_name`), document streaming (`stream_source_documents`, `stream_excel_documents`), metadata persistence (`save_metadata*`, `init_metadata_index`), and `get_journal_ids`. Uses `.parsers` internally for document serialization.
* **`indexer.py`** — Indexing orchestration. Takes `MongoDBManager` in `__init__`. Contains `index_documents`, `load_vector_index`, and the private `_make_ingestion_pipeline` helper. Imports `get_embed_model` from `.embed`.

## Steps
1. Create `db/` directory.
2. Move `embed.py` to `db/embed.py`.
3. Create `db/parsers.py` — move `_serialize_html_doc`, `_serialize_excel_doc`, and `EXCEL_METADATA_FIELDS` from `db.py`.
4. Create `db/manager.py` — move `MongoDBManager` from `db.py`. Remove `index_documents` and `load_vector_index`. Update internal imports to use `.parsers`.
5. Create `db/indexer.py` — new `Indexer` class taking `MongoDBManager`. Move `index_documents` and `load_vector_index` here. Move `_make_ingestion_pipeline` here as a private helper. Import `get_embed_model` from `.embed`.
6. Create `db/__init__.py` — expose `MongoDBManager`, `Indexer`, and `get_embed_model`.
7. Delete root `db.py`.
8. Update `index.py` — `from db import MongoDBManager, Indexer`. Wrap manager: `indexer = Indexer(client)`, then call `indexer.index_documents(...)`.
9. Update `pipeline.py` — `from embed import get_embed_model` becomes `from db import get_embed_model`. `MongoDBManager` import remains `from db import MongoDBManager`.
