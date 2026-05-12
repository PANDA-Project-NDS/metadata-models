# Journal Metadata Extraction System - Current Implementation

## General Overview

This system extracts structured metadata (APC costs, Editors, Impact Factors, etc.) from heterogeneous journal documents (HTML, PDF, Excel) using a Retrieval-Augmented Generation (RAG) pipeline.

The architecture supports two modes:

1. **Multi-pass mode** (default): Four parallel agents each handle a distinct schema subset. Each agent performs targeted semantic retrieval, then extracts from the retrieved context. An agentic search fallback lets agents self-correct when initial retrieval misses relevant chunks.
2. **Single-pass mode**: One agent extracts the full `JournalMetadata` schema from a single broad retrieval.

### Tech Stack

* **Frameworks**: LlamaIndex (RAG), Pydantic AI (agentic extraction)
* **Embeddings**: `BAAI/bge-small-en-v1.5` (HuggingFace, local)
* **LLM**: any 7b+ model with tool calling capability
* **HTML parsing**: Trafilatura (core content extraction)

---

## System Architecture

### Module Structure

```
pipeline.py          — Orchestrates ingestion, retrieval, and multi-pass extraction
agents.py            — Defines the four extraction agents with search tool wiring
agents_single.py     — Single-pass alternative agent for comparison
search.py            — Indexing, retrieval, search tool infrastructure
models/journal.py    — Pydantic schema definitions (modular composition)
```

### Flow — Multi-Pass Mode

```
pipeline.py process_journal()
  ┌────────────────────────────────────────────────┐
  │ await asyncio.gather(...)                       │
  │  ┌──────────────┐ ┌──────────────┐            │
  │  │ basic_info_agent │ policies_agent │           │
  │  │ (pass 1)        │ (pass 2)        │           │
  │  └──────────────┘ └──────────────┘            │
  │  ┌──────────────┐ ┌──────────────┐            │
  │  │ fees_agent      │ people_agent     │            │
  │  │ (pass 3)        │ (pass 4)        │           │
  │  └──────────────┘ └──────────────┘            │
  └────────────────────────────────────────────────┘
  → JournalMetadata(merged)

Per agent (inside run_extraction_pass):
  1. retrieve_for_pass(queries, journal_id)
  2. assemble_context(nodes) → context string
  3. agent.run(prompt, deps=...)
     └── if context insufficient:
          agent calls journal_search(query)  ← fallback
          pydantic_ai injects result → agent retries
```

---

## Design Decisions

### 1. Input Document Handling

HTML documents are processed with **Trafilatura** for core content extraction (not LlamaIndex's `SimpleDirectoryReader`). PDF and Excel formats are supported via LlamaIndex's readers. All documents are reduced to raw text chunks before embedding.

Journal IDs are derived from filename heuristics (e.g., `journal_alpha_apc.html` → `journal_alpha`).

### 2. Agentic Search Fallback

Each agent carries a `journal_search` tool attached via the `tools=` constructor parameter. When a retrieval query misses relevant chunks (common with scattered sub-pages and spreadsheets), the agent can call the tool to re-search with its own query terms. The tool is scoped to the current journal via `journal_id` metadata filters.

```python
# Tool function signature
def journal_search(
    ctx: RunContext[JournalSearchDeps],  # carries index + journal_id
    query: str                           # agent-formulated search string
) -> str:
    """Returns formatted context chunks, always scoped to one journal."""
```

`JournalSearchDeps` (a `dataclass`) is the agent dependency type. It is passed at runtime via `agent.run(deps=...)`.

### 3. Multi-Pass Architecture

Four agents run **concurrently** via `asyncio.gather`. Each pass handles a disjoint schema subset:

| Agent | Schema fields | Initial queries |
|-------|--------------|-----------------|
| `basic_info_agent` | title, publisher, ISSN, scope, impact factor | Journal title, ISSN, metrics |
| `policies_agent` | publication frequency, submission guidelines, review type | Publication frequency, submission guidelines, peer review |
| `fees_agent` | APC, waivers, discounts, membership | Article Processing Charge, APC, publication fees |
| `people_agent` | editorial board | Editorial board, Editor in Chief |

Multi-pass prevents context overloading even for 7B+ models, and avoids "lost in the middle" degradation that occurs with single-pass broad retrieval across many chunked documents.

### 4. Model Agnosticism

LLM selection is controlled via environment variables:

```python
# Ollama (local, default)
OLLAMA_BASE_URL=http://127.0.0.1:1234
OLLAMA_MODEL=qwen/qwen3

# OR OpenRouter (cloud)
OPENROUTER_MODEL=...
OPENROUTER_API_KEY=...
```

Swapping between local and cloud requires only env vars. Pydantic AI uses the OpenAI-compatible schema on both backends.

### 5. Evidence Tracking (Provenance)

Every field in the Pydantic schema is wrapped in `SourcedValue[T]` → `SourcedModel` → `Evidence`:

```
SourcedValue[str]
  └── SourcedModel
        ├── value: str           ← extracted data
        └── evidence: Evidence
              ├── quote: str       ← verbatim source text
              └── source: str      ← source filename
```

Context chunks are injected with `[Source: <filename>]` headers so the LLM can cite the source.

### 6. Pydantic Model Composition

Domain blocks (`JournalIdentity`, `Emission`, `SubmissionInfo`, etc.) are defined in `models/journal.py` and composed into four agent-specific schemas (`BasicInfoExtraction`, `PoliciesExtraction`, `FeesExtraction`, `PeopleExtraction`). These are in turn nested into the root `JournalMetadata` via multiple inheritance.

---

## Pipeline Functions

### `search.py`

**Indexing:**
- `build_global_index(directory_path) → VectorStoreIndex` — Loads HTML docs (Trafilatura), injects `journal_id` metadata, builds the global vector index.
- `load_with_trafilatura(directory_path) → List[Document]` — Walks directory, extracts text from HTML files.

**Retrieval:**
- `retrieve_for_pass(index, queries, journal_id, top_k) → list` — Queries the index for multiple phrases, deduplicates nodes, applies `journal_id` filter.
- `retrieve_for_query(index, query, journal_id, top_k) → list` — Single query variant (used by the search tool).
- `assemble_context(nodes) → str` — Wraps each chunk in `[Source: <filename>]` headers.

**Search infrastructure:**
- `JournalSearchDeps(index, journal_id)` — Dataclass carrying index + journal_id at runtime.
- `journal_search(ctx, query) → str` — Agent tool that performs retrieval scoped to one journal.

### `agents.py`

**Extraction queries (in `EXTRACTION_QUERIES`):**
Each pass has 1-3 predefined queries covering the domain vocabulary. These are the first-pass retrieval candidates.

**Agents:**
Four agents defined at module level, each with:
- `model`: LLM instance (Ollama or OpenRouter)
- `output_type`: Agent-specific Pydantic schema
- `system_prompt`: Base rules + domain-specific focus instruction
- `output_retries=3`: Retry on validation failures
- `deps_type=JournalSearchDeps`: Dependency type for tool access
- `tools=[journal_search]`: Agentic search fallback

**System prompt rules:**
1. No hallucination — null/empty if not in context
2. Verbatim evidence in `quote` field
3. Source tracking — cite source filename
4. Strict formatting — ISO currencies, ISSN format, canonical review types
5. JSON schema compliance — no extra keys
6. Retrieval fallback — call `journal_search` when context is insufficient

### `pipeline.py`

**`run_extraction_pass(index, agent, queries, journal_id)`**
1. Retrieves initial nodes via `retrieve_for_pass`
2. Assembles context string
3. Calls `agent.run(prompt, deps=JournalSearchDeps(index, journal_id))`
4. Handles `ValidationError`, `UnexpectedModelBehavior`, and general exceptions — falls back to empty schema on failure

**`process_journal(index, journal_id) → JournalMetadata`**
1. Runs all four passes concurrently
2. Merges results via `model_dump()` → dict unpacking into `JournalMetadata`

**`__main__`**
Builds index, discovers journal IDs from docstore, processes up to 5 journals, writes `extracted_metadata.json`.

---

## Schema Summary

### Modular blocks (in `models/journal.py`)

```
JournalIdentity    — title, publisher, ISSN
JournalScope       — description, journal_sections
PublicationFrequency — frequency, issues_per_year
ReviewProcess      — type (single/double/open), description
SubmissionInfo     — submission_guidelines, article_types
Pricing            — article_processing_charges, discounts
Editorial          — editors (name, role, affiliations)
Facts              — short_name, abbreviation, indexed_in
ImpactMetrics      — cite_score, impact_factor
AdditionalInformation — open_access_statement, copyright_statement, quality_assurance
```

### Agent schemas (composition)

| Agent | Block composition |
|-------|------------------|
| `BasicInfoExtraction` | JournalIdentity, JournalScope, Facts, ImpactMetrics |
| `PoliciesExtraction` | PublicationFrequency, SubmissionInfo, ReviewAndPolicy |
| `FeesExtraction` | Pricing, Membership |
| `PeopleExtraction` | Editorial |

### Root schema

`JournalMetadata` inherits from all four agent schemas.

---

## Agent Architecture

### Dependency Injection

Each agent uses `deps_type=JournalSearchDeps`. This type is set at construction time and enforced by pydantic_ai at runtime.

```python
# Construction
basic_info_agent = Agent(
    model=llm_model,
    output_type=BasicInfoExtraction,
    system_prompt=...,
    deps_type=JournalSearchDeps,
    tools=[journal_search],
)

# Runtime — passed to agent.run()
deps = JournalSearchDeps(index=index, journal_id=journal_id)
result = await basic_info_agent.run(prompt, deps=deps)
```

### Tool Function

The tool is a plain async function annotated with `RunContext[JournalSearchDeps]` as the first parameter:

```python
def create_search_tool():
    """Creates the journal_search tool bound to a specific index/journal_id pair."""
    ...

tool = create_search_tool()
```

At call time, pydantic_ai injects `ctx.deps` which contains both `index` and `journal_id`.

### Search Fallback Behavior

When the agent detects that required fields are null:

1. Agent invokes `journal_search(query="waiver policy")` with agent-formulated terms
2. Tool executes: filters by `ctx.deps.journal_id`, returns formatted chunk string
3. Pydantic AI injects the return value into context
4. Agent retries extraction

This repeats up to the model's tool call limit. If the tool is never needed, behavior is identical to the non-search mode.

---

## Key Implementation Details

### LlamaIndex Configuration

```python
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
Settings.llm = None  # LLM is handled by Pydantic AI externally
```

### Index Filter

All retrieval uses `ExactMatchFilter(key="journal_id", value=journal_id)` to ensure agents never see another journal's data.

### Context Injection Format

```
--- [Source: journal_alpha_apc.html] ---
<h2>Article Processing Charges</h2>
<p>$1,200 USD per article...</p>
```

---

## Running

```bash
# Local (Ollama)
OLLAMA_BASE_URL=http://127.0.0.1:1234 OLLAMA_MODEL=qwen/qwen3-1.7b python pipeline.py

# Cloud (OpenRouter)
OPENROUTER_MODEL=... OPENROUTER_API_KEY=... python pipeline.py
```

Results written to `extracted_metadata.json`.
