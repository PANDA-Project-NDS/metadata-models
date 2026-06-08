# Stage 5 — Langfuse Tracing

**Status**: Completed

## Goal

Replace Logfire with Langfuse as the primary observability backend for the golden metadata pipeline, with Logfire as a fallback when Langfuse credentials are absent. Leverage native Pydantic AI + LlamaIndex OTel integrations for zero-boilerplate agent and retrieval tracing.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ agents/base.py                                               │
│                                                               │
│  if LANGFUSE_PUBLIC_KEY set:                                 │
│    langfuse = get_client()                                   │
│    LlamaIndexInstrumentor().instrument()  ← LlamaIndex OTel  │
│    Agent.instrument_all()              ← Pydantic AI OTel     │
│  else:                                                       │
│    logfire.configure(send_to_logfire="if-token-present")     │
│    logfire.instrument_pydantic_ai()                          │
└─────────────────────────────────────────────────────────────┘
```

### Trace Hierarchy

```
Journal Extraction: {publisher}/{journal}  (trace, tags: golden-metadata, publisher)
├── Info Agent - Map (generation, Pydantic AI OTel, ×N chunks)
├── Info Agent - Reduce (generation, Pydantic AI OTel)
├── Info Agent - Completeness (generation, Pydantic AI OTel, may loop)
├── Info Agent - Verification (generation, Pydantic AI OTel)
├── Info Agent - Correction (generation, Pydantic AI OTel, may loop)
├── Policies Agent - Map ...
├── ... (4 passes × ~5 agents each)
├── Coverage Judge (generation, Pydantic AI OTel)
└── Evidence Judge (generation, Pydantic AI OTel)
```

All agent spans are captured automatically by Pydantic AI's OTel instrumentation. Each span includes agent name, input prompt, output model, tool calls, timing, and token usage. LlamaIndex embedding/retrieval/indexing operations are captured by `openinference-instrumentation-llama-index`.

### Trace Attributes

| Attribute | Value | Scope |
|---|---|---|
| `trace.name` | `Journal Extraction: {publisher}/{journal}` | Per journal |
| `tags` | `golden-metadata`, `{publisher}` | Per journal, propagates to children |
| `metadata.publisher` | Publisher slug | Per journal |
| `metadata.journal` | Journal slug | Per journal |

## File Changes

### 1. `pyproject.toml` — Add Langfuse dependencies

```diff
     "logfire>=4.32.1",
+    "langfuse>=2.50.0",
+    "openinference-instrumentation-llama-index>=2.0.0",
```

Logfire is kept for fallback.

### 2. `agents/base.py` — Dual-init: Langfuse or Logfire

**Remove:**
```python
import logfire
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
```

**Add:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

_langfuse_available = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))

if _langfuse_available:
    from langfuse import get_client
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from pydantic_ai.agent import Agent

    langfuse = get_client()
    LlamaIndexInstrumentor().instrument()
    Agent.instrument_all()
else:
    import logfire
    logfire.configure(send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()
    langfuse = None
```

- `Agent.instrument_all()` — hooks Pydantic AI to emit OTel spans to Langfuse
- `LlamaIndexInstrumentor().instrument()` — hooks LlamaIndex embedding/retrieval/indexing
- Logfire path unchanged when Langfuse not configured (preserves current behavior)
- `_langfuse_available` — module-level flag for conditional tracing in downstream code

### 3. `golden/agents/golden.py` — Enable `instrument=True` on all agents

Every `Agent(...)` constructor gains `instrument=True`:

```python
def make_map_agent(pass_config):
    return Agent(
        name=f"{pass_config.name} - Map",
        model=map_model,
        output_type=MapResult,
        system_prompt=MAP_PROMPT.format(...),
        instrument=True,  # <-- added
    )
```

Same for: `make_reduce_agent`, `make_completeness_agent`, `make_verification_agent`, `make_correction_agent`.

### 4. `golden/agents/judge.py` — Enable `instrument=True` on judges

```python
def make_coverage_judge():
    return Agent(
        name="Coverage Judge",
        model=judge_model,
        output_type=CoverageJudgeResult,
        system_prompt=COVERAGE_JUDGE_PROMPT,
        instrument=True,  # <-- added
    )
```

Same for `make_evidence_judge`.

### 5. `golden/main.py` — Journal-level trace wrapper + flush

Add imports:
```python
import contextlib
from agents.base import langfuse, _langfuse_available
```

Wrap journal loop:
```python
for identity in journals:
    publisher, journal = identity.publisher, identity.journal

    if _langfuse_available:
        from langfuse import propagate_attributes

        trace_ctx = langfuse.start_as_current_observation(
            as_type="trace",
            name=f"Journal Extraction: {publisher}/{journal}",
        )
        attr_ctx = propagate_attributes(
            metadata={"publisher": publisher, "journal": journal},
            tags=["golden-metadata", publisher],
        )
    else:
        trace_ctx = contextlib.nullcontext()
        attr_ctx = contextlib.nullcontext()

    with trace_ctx, attr_ctx:
        # ... all existing journal processing unchanged ...
```

Add flush before final logger:
```python
if _langfuse_available:
    langfuse.flush()
```

### 6. `implementation/20260601-golden-metadata/observability.md` — Update docs

- Logfire section → marked as fallback-only
- Langfuse section → primary, with `Agent.instrument_all()` auto-tracing
- Trace structure → matches actual implementation (Pydantic AI auto-spans + manual journal trace)
- Remove hybrid approach language; it's now Langfuse-first with Logfire fallback

## ENV Vars

Already present in `.env`:
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://...
```

No new vars needed. Existing `MAP_MODEL`, `REDUCE_MODEL`, etc. remain unchanged.

## Fallback Behavior

| Scenario | Behavior |
|---|---|
| `LANGFUSE_PUBLIC_KEY` set | Langfuse active, all agent calls traced via OTel |
| `LANGFUSE_PUBLIC_KEY` unset | Logfire active, `send_to_logfire="if-token-present"` preserves current no-op-if-no-token behavior |
| Both set | Langfuse wins, Logfire not loaded (zero overhead) |

## Design Decisions

| Decision | Rationale |
|---|---|
| `Agent.instrument_all()` + `instrument=True` | Zero-boilerplate tracing. Agent name, input, output, tool calls, timing all captured automatically. |
| `LlamaIndexInstrumentor()` | Auto-traces embedding, retrieval, indexing. No manual spans needed. |
| Manual trace wrapper in `main.py` | Groups all agent calls per journal into one trace. Provides publisher/journal metadata and tags. |
| `contextlib.nullcontext()` fallback | No deep nesting. Journal loop body stays unchanged whether Langfuse is active or not. |
| No pass-level tags | Agent names (e.g., `"Info Agent - Map"`) already identify the pass. Publisher tag provides filtering. |
| Logfire kept | Preserves existing behavior for environments without Langfuse. |
| No `@observe` decorators | Unnecessary with `Agent.instrument_all()`. Keeps agent factories clean. |
| No changes to `pipeline.py` | Agent calls remain untouched. Tracing is transparent. |

## Relevant Files

- `pyproject.toml` — add `langfuse`, `openinference-instrumentation-llama-index`
- `agents/base.py` — dual-init layer
- `golden/agents/golden.py` — `instrument=True` on 5 agents
- `golden/agents/judge.py` — `instrument=True` on 2 judges
- `golden/main.py` — journal trace wrapper + `flush()`
- `implementation/20260601-golden-metadata/observability.md` — update docs

## Impact

- **~40 lines added, ~3 lines removed** across 6 files
- **Zero changes** to `pipeline.py` — agent calls remain untouched
- **Zero runtime overhead** when Langfuse not configured (Logfire fallback)
- All agent spans, LlamaIndex operations, and journal-level traces appear in Langfuse UI
