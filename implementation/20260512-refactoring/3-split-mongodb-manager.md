# Split MongoDBManager into Focused Modules

## Status

Done

## Problem

`MongoDBManager` (`db/manager.py`, 97 lines) had 9 public methods spanning three unrelated responsibilities:

| Concern | Members | Used by |
|---|---|---|
| Connection | `client`, `get_collection`, `db_name`, `index_collection_name`, `close` | Internal + `Indexer` |
| Document streaming | `stream_source_documents`, `stream_excel_documents` | `Indexer.index_documents` only |
| Metadata | `get_journal_ids`, `init_metadata_index`, `save_metadata_one` | `pipeline.py` main only |

`Indexer` depended on the full `MongoDBManager` interface but only used connection + streaming. `pipeline.py` main only used connection + metadata. No overlap. The class was a grab-bag — a shallow module that didn't earn its keep behind a cohesive interface.

Deletion test: deleting `MongoDBManager` spread three independent sets of concerns to callers, none of whom needed the others' methods.

## Goal

Two focused classes. `Indexer` depends only on document store. `pipeline.py` main uses metadata store separately. Adding a metadata query doesn't touch `Indexer`. Adding a document format doesn't touch metadata code.

## Design

### `mongo_connection` (new)

**File:** `db/__init__.py`

A context manager that encapsulates connection configuration and lifecycle.

```python
@contextmanager
def mongo_connection():
    uri = os.environ["MONGODB_URI"]
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        yield client
    finally:
        client.close()
```

### `DocumentStore` (renamed from `MongoDBManager`)

**File:** `db/documents.py`

Receives a `MongoClient` via dependency injection. Owns document streaming logic. No lifecycle ownership.

```python
class DocumentStore:
    def __init__(self, client: MongoClient) -> None:
        self.client = client

    def get_collection(self, name: str) -> Collection

    @property
    def index_collection_name(self) -> str

    @property
    def db_name(self) -> str

    def stream_source_documents(self, collection_name: str, limit: int = 0) -> Iterator[Document]

    def stream_excel_documents(self, collection_name: str, limit: int = 0) -> Iterator[Document]

    def get_journal_ids(self, publisher: str | None = None) -> list[str]
```

Removed: `init_metadata_index`, `save_metadata_one`, `close()`, `_uri`, lazy client init.

### `MetadataStore` (new)

**File:** `db/metadata.py`

Receives a `MongoClient` via dependency injection. Reads env vars for collection names internally — no constructor parameters beyond the client.

```python
class MetadataStore:
    def __init__(self, client: MongoClient) -> None

    @property
    def metadata_collection(self) -> str
    # Returns os.getenv("MONGO_METADATA_COLLECTION", "journal_metadata")

    @property
    def index_collection_name(self) -> str
    # Returns os.getenv("MONGO_INDEX_COLLECTION", "search_index")



    def init_metadata_index(self) -> None
    # Uses self.metadata_collection internally

    def save_metadata_one(self, publisher_id: str, journal_id: str, metadata: dict) -> None
    # Uses self.metadata_collection internally
```

### Entry Points

Both `pipeline.py` and `index.py` use the `mongo_connection` context manager:

```python
with mongo_connection() as client:
    store = DocumentStore(client)
    meta = MetadataStore(client)
    # ... use stores ...
```

### `Indexer` (updated)

Takes `DocumentStore` instead of `MongoDBManager`. Same interface (`client`, `db_name`, `index_collection_name`, `get_collection`, `stream_*`). Only the type name changed.

### `db/__init__.py` (updated)

```python
__all__ = ["mongo_connection", "DocumentStore", "MetadataStore", "Indexer", "get_embed_model"]
```

### `parsers.py` (unchanged)

Stays where it is. Only called from `DocumentStore`'s streaming methods.

## Changed files

| File              | Action |
|-------------------|---|
| `db/manager.py`   | Deleted |
| `db/documents.py` | New. `DocumentStore` class. |
| `db/metadata.py`  | New. `MetadataStore` class. |
| `db/indexer.py`   | Import + type hint: `MongoDBManager` → `DocumentStore` |
| `db/__init__.py`  | Added `mongo_connection` context manager. Updated exports. |
| `pipeline.py`     | Uses `mongo_connection`, `DocumentStore`, `MetadataStore`. |
| `index.py`        | Uses `mongo_connection`, `DocumentStore`. |

## Unchanged

- `db/parsers.py` — untouched
- `db/embed.py` — untouched
- `agents.py`, `search.py`, `models/` — untouched

## Verification

- Imports: `from db import mongo_connection, DocumentStore, MetadataStore, Indexer, get_embed_model` ✓
- Lint: `ruff check db/ index.py pipeline.py` — clean (pre-existing `parsers.py` error excluded) ✓
- `grep MongoDBManager --include='*.py'` — zero results in scope (remains in `compare_fees.py`, `scripts/field_coverage.py` — out of scope) ✓
- `grep from db.manager --include='*.py'` — zero results ✓
