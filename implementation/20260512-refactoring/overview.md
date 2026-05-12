# Deepening Opportunities — 2026-05-12

## Glossary

- **Module** — anything with an interface and an implementation (function, class, package, slice).
- **Interface** — everything a caller must know to use the module: types, invariants, error modes, ordering, config.
- **Implementation** — the code inside.
- **Depth** — leverage at the interface. **Deep** = high leverage. **Shallow** = interface nearly as complex as the implementation.
- **Seam** — where an interface lives; a place behaviour can be altered without editing in place.
- **Adapter** — a concrete thing satisfying an interface at a seam.
- **Deletion test** — imagine deleting the module. If complexity vanishes, it was a pass-through. If complexity reappears across N callers, it was earning its keep.

---

### 1. Factorize the four extraction agents into an agent factory

**Files:** `agents.py`

**Problem:** Four agents (`basic_info_agent`, `policies_agent`, `fees_agent`, `editors_agent`) are defined as module-level singletons with nearly identical configuration. `SYSTEM_PROMPT`, `llm_model`, `journal_search_tool`, `output_retries`, `deps_type`, and `instructions` are copy-pasted across all four. `EXTRACTION_QUERIES` lives at module level with no structural relationship to the agents. Deletion test: deleting `agents.py` spreads agent construction, prompt, model config, and tool wiring into callers. The module is a pass-through — four pre-built objects with no behaviour behind the interface.

**Solution:** Create an agent factory function that takes a pass configuration (name, output type, queries) and returns a configured `Agent`. Define the four passes as data — a list of `PassConfig` structs — not as code. The factory becomes the seam for model, retries, tools, and prompt template.

**Benefits:**
- **Locality:** Model config, retry policy, tool wiring, and prompt template live in one place.
- **Leverage:** Small factory interface (pass config in, agent out). Callers don't need to know about `OpenAIChatModel`, `Tool`, `RunContext`, or prompt construction.
- **Tests:** Factory is unit-testable with a mock model. First meaningful test surface in the project.

---

### 2. Deepen the extraction pipeline into a configurable orchestrator

**Files:** `pipeline.py`

**Problem:** `process_journal` hardcodes four parallel passes with explicit task variables and merges via `**model_dump()` unpacking. The merge relies on the four extraction schemas having non-overlapping fields — a fragile, untested invariant. `run_extraction_pass` is a shallow pass-through: retrieval, deps, agent call, catch-all error handling that discards partial results. Deletion test: the function doesn't hide complexity — callers must still understand the four passes, the merge, and the error modes.

**Solution:** Extract a `JournalPipeline` module that takes a list of pass configurations and orchestrates them. Handles: (1) building concurrent tasks from pass configs, (2) gathering results, (3) explicit merge via `merge_passes()`, (4) structured error handling that preserves partial results.

**Benefits:**
- **Locality:** Concurrency, merge strategy, and error fallback concentrated in one module.
- **Leverage:** Interface is `run(journal_id) -> JournalMetadata`. Callers don't need to know about four passes, `asyncio.gather`, or `model_dump`.
- **Tests:** `merge_passes` is a pure function. Pipeline testable with mock agents returning controlled partial results. Error paths become testable scenarios instead of untested `except` blocks.

---

### 3. Split `MongoDBManager` into focused modules

**Files:** `db/manager.py`, `db/parsers.py`, `db/indexer.py`

**Problem:** `MongoDBManager` has 9 public methods spanning three unrelated responsibilities: connection management (`client`, `get_collection`, `close`), document parsing (`stream_source_documents`, `stream_excel_documents`), and metadata persistence (`save_metadata_one`, `init_metadata_index`, `get_journal_ids`). The parser functions in `parsers.py` are pure but only called from `manager.py`, creating tight coupling. `Indexer` depends on `MongoDBManager` but only needs `client`, `db_name`, `index_collection_name`, and the streaming methods. Deletion test: the class is a grab-bag of MongoDB operations with no unifying abstraction.

**Solution:** Split into three focused modules behind `db/__init__.py`:
- **Connection module** (`db/connection.py`): `MongoClient` lifecycle, `get_collection`, env-based `db_name`/`index_collection_name`
- **Document streaming module** (`db/streaming.py`): combines connection + parsers to produce `Iterator[Document]`
- **Metadata store module** (`db/metadata.py`): `save_metadata_one`, `init_metadata_index`, `get_journal_ids`

**Benefits:**
- **Locality:** Each module has a single responsibility. New document format touches only `streaming.py` and `parsers.py`. New metadata query touches only `metadata.py`.
- **Leverage:** Callers import only what they need. Each module could have an alternative adapter (e.g., test in-memory store).
- **Tests:** `parsers.py` is pure and trivially testable. Metadata store testable with `mongomock`. Streaming testable with fake DB documents.

---

### 4. Extract `JournalSourcesDeps` context management into a retrieval session

**Files:** `search.py`

**Problem:** `JournalSourcesDeps` mixes three concerns: (1) agent dependency type for pydantic_ai `deps_type`, (2) accumulated context node management via `extend_nodes`/`node_ids`, (3) node formatting via `format_nodes`/`context_instructions`. The `journal_search` tool mutates `context_nodes` in place — the deps object has side effects. Deletion test: the interface (`index`, `journal_id`, `context_nodes`, `format_nodes`, `extend_nodes`, `node_ids`, `context_instructions`) is nearly as complex as the implementation. Shallow.

**Solution:** Split at the seam:
- **`RetrievalSession`** (new): owns node accumulation, deduplication, and formatting. Pure data management — no agent framework coupling.
- **`JournalSourcesDeps`** (simplified): thin wrapper holding a `RetrievalSession`, providing `context_instructions` for pydantic_ai. `journal_search` delegates to the session.

**Benefits:**
- **Locality:** Node management logic concentrated in `RetrievalSession`. Agent deps layer is thin and framework-specific.
- **Leverage:** `RetrievalSession` has a small interface (`add_nodes`, `format`, `node_ids`). Usable outside agents — e.g., non-agentic retrieval flow.
- **Tests:** `RetrievalSession` is pure and simply stateful — easy to test dedup, formatting, extension. Current `JournalSourcesDeps` is hard to test due to pydantic_ai `RunContext` coupling.

---

### 5. Add test infrastructure with schema merge as the first test surface

**Files:** (new) `tests/`, (new) `models/merge.py`

**Problem:** Zero tests in the project. The merge logic in `pipeline.py:103-108` (`**model_dump()` unpacking into `JournalMetadata`) is untested. It relies on the four extraction schemas having non-overlapping fields. If two agents produce overlapping fields, one silently overwrites the other.

**Solution:** Create `tests/` with pytest. Extract the merge into a pure `merge_extractions()` function in `models/merge.py`. Test that: (1) non-overlapping fields merge correctly, (2) overlapping fields raise an error, (3) partial results produce valid `JournalMetadata`.

**Benefits:**
- **Locality:** Merge logic moves from the pipeline (mixed with async orchestration) into the models package (where the schemas live).
- **Leverage:** Clean interface — four extraction objects in, `JournalMetadata` out. Pipeline calls it without knowing about `model_dump`.
- **Tests:** Highest-value first test — covers the invariant that holds the entire multi-pass architecture together.
