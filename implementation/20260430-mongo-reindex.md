# MongoDB Persistent Vector Store — Implementation Plan

## Motivation

The pipeline currently builds an in-memory `VectorStoreIndex` on every run:

```
pipeline.py __main__
  → MongoDBManager.load_source_documents()  ← raw HTML from MongoDB
  → VectorStoreIndex.from_documents(docs)   ← embeds in-memory every time
  → process_journal(index, ...)
```

Embedding is the most expensive step (HuggingFace model, runs once per document per pipeline invocation). With a persistent vector store, embedding happens once at index time. The pipeline then loads pre-embedded nodes from MongoDB and skips re-embedding entirely.

### Goals

1. **Index once, reuse many times** — Embed documents into MongoDB via `MongoDBAtlasVectorSearch`, store in `{input_collection}_index`.
2. **Pipeline loads pre-indexed data** — No embedding at pipeline runtime.
3. **Indexing logic lives in `db.py`** — `MongoDBManager` handles both indexing and loading.
4. **Standalone `index.py` script** — Thin entry point to trigger indexing.
5. **Streaming throughout** — Never hold complete document/node lists in memory.
6. **`IngestionPipeline` for indexing** — Replace global `Settings` with a self-contained `IngestionPipeline`.

---

## Architecture

### Before (current)

```
MongoDB (wiley_full)
  └── load_source_documents() → List[Document]       (all in memory)
        └── VectorStoreIndex.from_documents() → in-memory index
              └── pipeline: retrieve, extract, save
```

### After

```
MongoDB (wiley_full)
  └── MongoDBManager.stream_source_documents() → iterator[Document]
        └── IngestionPipeline.run(batch) → chunks, embeds, persists to MongoDB
              (MongoDBAtlasVectorSearch → wiley_full_index)

MongoDB (wiley_full_index)
  └── MongoDBManager.load_vector_index() → VectorStoreIndex
        └── MongoDBManager.get_journal_ids() → distinct journal_ids
              └── pipeline: stream process_journal(), save per journal
```

---

## Implementation Details

### 1. LlamaIndex Settings — Moved to `IngestionPipeline`

**Current state:** `pipeline.py` top-level defines global `Settings`:

```python
# pipeline.py (current, will be removed)
Settings.embed_model = HuggingFaceEmbedding(model_name=...)
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
```

**New approach:** The `IngestionPipeline` encapsulates all indexing-time transformations. No global `Settings` needed for indexing. `pipeline.py` only sets `Settings.embed_model` for **query-time** embedding (retrieval queries must be embedded to compare against stored vectors).

```python
# db.py — factory for the ingestion pipeline
def _make_ingestion_pipeline(vector_store, embed_model_name: str):
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=50),
            HuggingFaceEmbedding(model_name=embed_model_name),
        ],
        vector_store=vector_store,
    )
```

`IngestionPipeline.run()` accepts `documents=` (raw `Document` objects that still need chunking/embedding) or `nodes=` (pre-parsed nodes). We use `documents=` so the pipeline applies both `SentenceSplitter` and embedding transformations.

### 2. `db.py` — New/Modified Methods on `MongoDBManager`

#### `stream_source_documents(collection_name: str, limit: int = 0) -> Iterator[Document]`

**Purpose:** Yield `Document` objects one at a time from MongoDB. Replaces the batch-oriented `load_source_documents()` for indexing use. Only yields documents that have `metadata.html`.

```python
def stream_source_documents(self, collection_name: str, limit: int = 0) -> Iterator[Document]:
    """Yield Document objects from MongoDB without holding them all in memory."""
    collection = self.get_collection(collection_name)
    cursor = collection.find({"metadata.html": {"$exists": True, "$ne": None}}).limit(limit)
    for db_doc in cursor:
        html_content = db_doc.get("metadata", {}).get("html", "")
        if not html_content:
            continue
        extracted_text = trafilatura.extract(html_content)
        if not extracted_text:
            extracted_text = html_content
        metadata = db_doc.get("metadata", {})
        yield Document(
            text=extracted_text,
            metadata={
                "file_path": metadata.get("url", "unknown"),
                "source_uri": metadata.get("url", "unknown"),
                "journal_id": metadata.get("title", "unknown"),
            },
        )
```

Existing `load_source_documents()` is retained for backward compatibility but deprecated internally.

#### `index_documents(input_collection: str, output_collection: str | None = None, limit: int = 0, batch_size: int = 10) -> VectorStoreIndex`

**Purpose:** Stream raw documents from `input_collection`, chunk, embed, and persist into `output_collection` (defaults to `{input_collection}_index`). Processes in batches via `IngestionPipeline`.

**Steps:**
1. Create `MongoDBAtlasVectorSearch` for `output_collection`
2. Create Atlas search index if it doesn't exist: `vector_store.create_index_if_not_exists()`
3. Create `IngestionPipeline` with `SentenceSplitter` + `HuggingFaceEmbedding` transformations
4. Stream documents from `stream_source_documents()`, collect into batches of `batch_size`, run through the pipeline
5. Return `VectorStoreIndex.from_vector_store(vector_store)` for ad-hoc use

```python
def index_documents(self, input_collection, output_collection=None, limit=0, batch_size=10):
    if output_collection is None:
        output_collection = f"{input_collection}_index"

    embed_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    db = self.client.get_database()
    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=self.client,
        db_name=db.name,
        collection_name=output_collection,
        vector_index_name="vector_index",
    )
    vector_store.create_index_if_not_exists()

    ingestion = _make_ingestion_pipeline(vector_store, embed_model_name)

    # Count total documents for progress bar
    total_docs = self.get_collection(input_collection).count_documents(
        {"metadata.html": {"$exists": True, "$ne": None}}
    )
    if limit:
        total_docs = min(total_docs, limit)

    doc_iter = self.stream_source_documents(input_collection, limit)
    batch: List[Document] = []
    indexed = 0
    with tqdm(total=total_docs, desc=f"Indexing to '{output_collection}'") as pbar:
        for doc in doc_iter:
            batch.append(doc)
            if len(batch) >= batch_size:
                ingestion.run(documents=batch)
                indexed += len(batch)
                pbar.update(len(batch))
                batch = []
        if batch:
            ingestion.run(documents=batch)
            indexed += len(batch)
            pbar.update(len(batch))

    logger.info(f"Indexing complete. {indexed} documents processed into '{output_collection}'.")
    return VectorStoreIndex.from_vector_store(vector_store)
```

**Memory profile:** At any point, only `batch_size` documents (default 10) are held in memory. The MongoDB cursor streams from disk.

#### `load_vector_index(collection_name: str) -> VectorStoreIndex`

**Purpose:** Load a pre-existing vector index from MongoDB. No document embedding.

```python
def load_vector_index(self, collection_name: str) -> VectorStoreIndex:
    db = self.client.get_database()
    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=self.client,
        db_name=db.name,
        collection_name=collection_name,
        vector_index_name="vector_index",
    )
    return VectorStoreIndex.from_vector_store(vector_store)
```

**Note:** `Settings.embed_model` must be set by the caller (i.e., `pipeline.py`) for query-time embedding. This method only wires up the persistent vector store.

#### `get_journal_ids(collection_name: str) -> List[str]`

**Purpose:** Query distinct `journal_id` values from the **indexed collection**. The `journal_id` field is stored in each node's metadata (set when creating `Document` objects during indexing). This is the authoritative source of what's actually been indexed, avoiding coupling to the raw input collection schema.

```python
def get_journal_ids(self, collection_name: str) -> List[str]:
    """Get distinct journal IDs from the indexed collection via MongoDB query.
    Queries metadata.journal_id from the vector store nodes."""
    collection = self.get_collection(collection_name)
    ids = collection.distinct("metadata.journal_id")
    ids = [jid for jid in ids if jid and jid != "unknown"]
    logger.info(f"Found {len(ids)} journal IDs in collection '{collection_name}': {ids}")
    return ids
```

**Note:** `MongoDBAtlasVectorSearch` stores node metadata under the key `metadata` (configurable via `metadata_key` constructor param, defaults to `"metadata"`). So `journal_id` is at path `metadata.journal_id`.

#### `save_metadata_one(collection_name: str, journal_id: str, metadata: dict)`

**Purpose:** Save a single journal's metadata, for streaming pipeline output.

```python
def save_metadata_one(self, collection_name: str, journal_id: str, metadata: dict):
    collection = self.get_collection(collection_name)
    collection.create_index("journal_id", unique=True)
    collection.replace_one(
        {"journal_id": journal_id},
        {"journal_id": journal_id, **metadata},
        upsert=True,
    )
```

Existing `save_metadata()` (batch) is retained for backward compatibility.

---

### 3. New file: `index.py`

**Purpose:** Standalone script to trigger indexing.

```python
#!/usr/bin/env python3
"""Index raw documents into MongoDB vector store."""
import argparse
import logging
import os

from db import MongoDBManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Index documents into MongoDB vector store")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of documents to index")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for ingestion pipeline")
    parser.add_argument("--force-rebuild", action="store_true", help="Drop existing index collection before re-indexing")
    args = parser.parse_args()

    input_collection = os.environ.get("MONGO_COLLECTION", "wiley_full")
    output_collection = os.environ.get("MONGO_INDEX_COLLECTION", f"{input_collection}_index")

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
```

**Usage:**
```bash
# Full index
python index.py

# Test with 5 documents
python index.py --limit 5

# Rebuild from scratch
python index.py --force-rebuild
```

---

### 4. `pipeline.py` — Modified `__main__`

**Changes:**
1. **Remove** global `Settings.text_splitter` — no longer needed (only relevant at index time)
2. **Keep** `Settings.embed_model` — required for query-time embedding during retrieval
3. Replace `VectorStoreIndex.from_documents()` with `client.load_vector_index()`
4. Replace `index.docstore.docs.values()` iteration with `client.get_journal_ids()` MongoDB query
5. Stream results: save each journal's metadata immediately instead of accumulating in `results` dict

**Modified `__main__`:**
```python
if __name__ == "__main__":
    import asyncio
    import json

    from db import MongoDBManager

    # Query-time embedding model (required for retrieval)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    async def main():
        try:
            client = MongoDBManager(os.environ["MONGO_URI"])
            try:
                index_collection = os.environ.get(
                    "MONGO_INDEX_COLLECTION",
                    f"{os.environ.get('MONGO_COLLECTION', 'wiley_full')}_index",
                )

                # Load pre-indexed vector store (no document embedding)
                global_index = client.load_vector_index(index_collection)

                # Query journal IDs from indexed collection
                all_journal_ids = client.get_journal_ids(index_collection)
                logger.info(f"Processing {len(all_journal_ids)} journals")

                # Stream: process and save each journal immediately
                metadata_collection = os.environ.get("MONGO_METADATA_COLLECTION", "journal_metadata")
                for j_id in all_journal_ids:
                    metadata = await process_journal(global_index, j_id)
                    client.save_metadata_one(metadata_collection, j_id, json.loads(metadata.model_dump_json()))
                    logger.info(f"Saved metadata for {j_id}")

            finally:
                client.close()

        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)

    asyncio.run(main())
```

**Removed from module top-level:**
```python
# REMOVED — was here, no longer needed
# Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
```

---

### 5. `search.py` — No changes

The retrieval layer (`retrieve_for_pass`, `build_retriever`, `journal_search`) operates on `BaseIndex`/`BaseRetriever` abstractions and is agnostic to the underlying vector store backend. No modifications needed.

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MONGO_URI` | MongoDB connection string | (required) |
| `MONGO_COLLECTION` | Raw input collection name | `wiley_full` |
| `MONGO_INDEX_COLLECTION` | Indexed output collection name | `{MONGO_COLLECTION}_index` |
| `MONGO_METADATA_COLLECTION` | Extraction results collection | `journal_metadata` |
| `EMBEDDING_MODEL` | Embedding model name | `BAAI/bge-small-en-v1.5` |

Removed `EMBEDDING_DIM` — dimension is derived from the model at runtime.

---

## File Changes Summary

| File | Change |
|------|--------|
| `db.py` | Add `stream_source_documents()`, `index_documents()`, `load_vector_index()`, `get_journal_ids()`, `save_metadata_one()` to `MongoDBManager`. Add `_make_ingestion_pipeline()` helper. |
| `pipeline.py` | Remove `Settings.text_splitter`. Keep `Settings.embed_model` (query-time only). Replace `__main__` to load vector index, query journal IDs from MongoDB, stream results. |
| `index.py` | **New file** — standalone indexing script with `--limit`, `--batch-size`, `--force-rebuild` |
| `search.py` | No changes |
| `agents.py` | No changes |
| `models/` | No changes |

---

## Migration Path

1. **Run `index.py`** once to populate `{collection}_index` from existing raw data
2. **Run `pipeline.py`** — loads pre-indexed collection, queries journal IDs from MongoDB, streams extraction results
3. **Re-run `index.py`** whenever new raw documents are added to the input collection
4. Old `load_source_documents()` and `save_metadata()` in `db.py` are retained for backward compatibility

---

## Risks & Considerations

### `get_journal_ids()` metadata field path
`get_journal_ids()` queries `metadata.journal_id` from the indexed collection. The `MongoDBAtlasVectorSearch` stores node metadata under the `metadata_key` (defaults to `"metadata"`), so the path is `metadata.journal_id`. This is under our control since we set the metadata when creating `Document` objects.

### Embedding dimension
`MongoDBAtlasVectorSearch` needs to know the embedding dimension for the Atlas search index. The dimension is derived from the model. If the embedding model changes, the index must be rebuilt with `--force-rebuild`.

### Atlas search index creation
`MongoDBAtlasVectorSearch.create_index_if_not_exists()` handles index creation automatically. Requires MongoDB Atlas (or Atlas Local via `db/docker-compose.atlas.yaml`).

### Query-time embedding still required
`Settings.embed_model` is still set in `pipeline.py` because retrieval queries must be embedded to compute similarity against stored vectors. Only **document** embedding is moved offline.

### `IngestionPipeline` vs `VectorStoreIndex.from_documents()`
`IngestionPipeline` gives us explicit control over batching and streaming. `VectorStoreIndex.from_documents()` loads all documents at once. The pipeline approach is strictly better for memory efficiency.

---

## Testing

1. **Index script (small batch):**
   ```bash
   MONGO_COLLECTION=wiley_full python index.py --limit 5
   ```
   Verify nodes exist in `wiley_full_index` with `embedding` field populated.

2. **Journal ID query:**
   ```python
   client = MongoDBManager(uri)
   print(client.get_journal_ids("wiley_full_index"))
   ```
   Verify output matches expected journal IDs (from indexed collection).

3. **Pipeline with pre-indexed data:**
   ```bash
   MONGO_COLLECTION=wiley_full python pipeline.py
   ```
   Verify extraction runs without calling the embedding model on documents.

4. **Re-indexing:**
   Add new documents to `wiley_full`, re-run `python index.py --force-rebuild`, verify new journal IDs appear in pipeline output.
