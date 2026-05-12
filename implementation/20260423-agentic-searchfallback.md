# Implement Agent-Driven Search Fallback

## Goal

Give each extraction agent a `journal_search` tool so it can re-query the index when the initial retrieved context doesn't contain evidence for a required field. The tool is scoped to the current journal's documents only.

This feature bridges the gap between static retrieval queries and scattered document layouts: when chunks are spread across many sub-pages and spreadsheets, the agent inspects what was retrieved, detects gaps, and formulates its own search terms as a fallback.

## Architecture Overview

```
Pipeline flow (unchanged):
  build_global_index() → process_journal(j_id) → 4 parallel run_extraction_pass()

New search tool flow (inside each pass):
  1. Initial retrieval: retrieve_for_pass(EXTRACTION_QUERIES, journal_id)
  2. Agent evaluates → null fields detected
  3. Agent calls journal_search(query="waiver criteria")
  4. Tool retrieves scoped nodes → returns formatted context
  5. Pydantic AI injects context → agent retries extraction
  6. Steps 3-5 repeat until all fields filled or max retries
```

## File: `search.py` (new)

Extract indexing/retrieval logic from `pipeline.py` and add search tool definition.

### Exports

```python
# ── Indexing ──────────────────────────────────────────────

def load_with_trafilatura(directory_path: str) -> List[Document]:
    """Loads HTML documents from a directory and extracts core text using Trafilatura."""

def build_global_index(directory_path: str) -> VectorStoreIndex:
    """Loads all documents, injects journal_id metadata, and builds a global vector index."""

# ── Retrieval ────────────────────────────────────────────

def retrieve_for_query(index: VectorStoreIndex, query: str, journal_id: str, top_k: int = 3) -> list:
    """Retrieves and deduplicates nodes for a single query, filtered by journal_id."""

def retrieve_for_pass(index: VectorStoreIndex, queries: list[str], journal_id: str, top_k: int = 3) -> list:
    """Retrieves and deduplicates nodes for multiple queries, filtered by journal_id."""

def assemble_context(nodes: list) -> str:
    """Formats retrieved nodes into a context string with source citations."""

# ── Search Tool Infrastructure ───────────────────────────

@dataclass(kw_only=True)
class JournalSearchDeps:
    index: VectorStoreIndex
    journal_id: str

def journal_search(ctx: RunContext[JournalSearchDeps], query: str) -> str:
    """Agent tool: search the journal's documents and return formatted context chunks.
    
    Scopes retrieval to the current journal via journal_id from deps.
    Returns context string in the format the agent expects (with [Source:] headers).
    """
    nodes = retrieve_for_query(ctx.deps.index, query, ctx.deps.journal_id)
    return assemble_context(nodes)
```

### Design decisions

- `journal_search` is a standalone function (not a class method). Pydantic AI auto-converts sync tool functions.
- `RunContext[JournalSearchDeps]` carries the index and journal_id at runtime. No global state.
- `retrieve_for_query` is a new helper (one query) extracted from `retrieve_for_pass`. The tool only needs a single query.
- `journal_search` returns the raw context string — Pydantic AI injects it into the agent's context window automatically.
- `JournalSearchDeps` is a dataclass (not BaseModel) since it's a dependency container, not a validation target.

## File: `agents.py`

Wire the search tool onto each agent using `tools=` and `deps_type=`.

### Imports

```python
from search import (
    JournalSearchDeps,
    journal_search,
)
from pydantic_ai import RunContext
```

### Agent construction changes

```python
# Before (no deps, no tools):
basic_info_agent = Agent(
    model=llm_model,
    output_type=BasicInfoExtraction,
    system_prompt=BASE_SYSTEM_PROMPT + "\nFocus purely on basic info...",
    output_retries=3
)

# After (deps + tool):
BASE_SYSTEM_PROMPT_WITH_FALLBACK = BASE_SYSTEM_PROMPT + """
RETRIEVAL FALLBACK:
If the retrieved context does not contain evidence for a required field,
you may CALL the 'journal_search' tool to look for additional information.
Use your own query terms — the predefined queries may have missed relevant chunks.
Retrieval is scoped to documents for this journal only.
"""

basic_info_agent = Agent(
    model=llm_model,
    output_type=BasicInfoExtraction,
    system_prompt=BASE_SYSTEM_PROMPT_WITH_FALLBACK + "\nFocus purely on basic info...",
    output_retries=3,
    deps_type=JournalSearchDeps,
    tools=[journal_search],
)
```

### Changes summary

| Change | Why |
|--------|-----|
| `deps_type=JournalSearchDeps` | Binds the index and journal_id to the agent's dependency type |
| `tools=[journal_search]` | Attaches the search tool to each agent at construction |
| `BASE_SYSTEM_PROMPT_WITH_FALLBACK` | Instructs the agent when and how to use the tool |

The same pattern applies to all four agents: `basic_info_agent`, `policies_agent`, `fees_agent`, `people_agent`.

## File: `pipeline.py`

Pass deps context from the pipeline down to each agent run.

### imports

```python
from search import build_global_index, retrieve_for_pass, assemble_context, JournalSearchDeps
```

### `run_extraction_pass` changes

```python
async def run_extraction_pass(
    index: VectorStoreIndex, agent, queries: list[str], journal_id: str
):
    """Executes a single extraction pass using specific queries, filtered by journal_id."""
    logger.info(f"[{journal_id}] Running extraction pass for queries: {queries[:1]}...")
    nodes = retrieve_for_pass(index, queries, journal_id)

    if not nodes:
        logger.warning(f"[{journal_id}] No nodes retrieved for queries: {queries[:1]}")

    context_str = assemble_context(nodes)
    prompt = f"Extract metadata using the following retrieved context:\n\n{context_str}"

    # NEW: pass deps to enable tool-calling
    deps = JournalSearchDeps(index=index, journal_id=journal_id)

    try:
        result = await agent.run(
            prompt,
            deps=deps,
            model_settings=ModelSettings(timeout=300),
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
    return agent.result_type()
```

### `process_journal` changes

No changes needed — `process_journal` already passes `index` and `journal_id` to `run_extraction_pass`. The deps object is created inside `run_extraction_pass` and is scoped to a single agent run. No propagation up the call chain.

### What stays in `pipeline.py`

- `assemble_context()` — still needed for the initial retrieval in `run_extraction_pass`
- `retrieve_for_pass()` — still needed for the initial retrieval in `run_extraction_pass`
- `build_global_index()` — still needed in `__main__`

These are re-imported from `search` rather than re-defined.

### `__main__` block changes

No changes needed. `build_global_index` is imported from `search`.

## Agent Behavior During Search Tool Calls

When a agent detects a null field after initial context:

1. Agent calls `journal_search(query="waiver criteria")`
2. Tool executes: retrieves nodes filtered by `ctx.deps.journal_id`
3. Tool returns formatted context string
4. Pydantic AI injects the returned string into the agent's context window
5. Agent retries extraction with expanded context
6. Steps 1-5 repeat until all fields filled or max tool call retries exceeded

The agent can call the tool with any query it formulates. The tool always scopes to the current journal. No manual dedup or context management needed — pydantic_ai handles context accumulation.

## Edge Cases

| Case | Behavior |
|------|----------|
| `journal_id` has no documents | `journal_search` returns empty context string (0 nodes) |
| Tool call fails (e.g. index corrupted) | Pydantic AI raises an error; caught by `run_extraction_pass` try/except; returns empty schema |
| Agent never calls tool | Falls back to initial retrieval result (same as current behavior) |
| Max tool retries hit | Agent returns what it extracted; nulls remain for unretrieved fields |

## Testing

1. **Happy path**: Journal with APC on separate page from waiver. Test that agent calls `journal_search` and fills both fields.
2. **No-op**: Journal with all info on same page. Agent should not call the tool (verify via logfire traces).
3. **Empty journal**: Journal id with 0 documents. Should produce empty schema gracefully.
4. **Index corruption**: Feed invalid index to tool. Should return graceful error, not crash pipeline.
