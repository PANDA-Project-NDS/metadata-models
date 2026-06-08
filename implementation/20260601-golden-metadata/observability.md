# Observability & Logging Implementation

This document defines the observability strategy for the Golden Metadata generation pipeline, utilizing Langfuse as the primary tracing backend with Logfire as a fallback when Langfuse credentials are absent.

## 1. Standard Logging (CLI)
**Focus**: High-level health, pipeline progress, and aggregate metrics.
**Goal**: Answer *"Is the script running, where is it stuck, and how is it performing overall?"*

Standard Python `logging` (configured in `golden/main.py`). No observability SDK dependency.

| Event | Log Level | Details to Capture |
| :--- | :--- | :--- |
| **Startup** | `INFO` | Loaded models, Judge model selection, total journals discovered. |
| **Parser Fallback** | `INFO/WARN` | `[Parser] Succeeded for {journal}` OR `[Parser] Failed → Falling back to Map-Reduce`. |
| **Loop Progress** | `INFO` | `[Completeness] Round {n}: Added {x} new fields`. |
| **Verification** | `INFO` | `[Verify] Found {n} errors in {journal}`. |
| **Correction** | `DEBUG` | `[Correct] Targeting fields: {field_paths}`. |
| **Convergence** | `INFO` | `[Pipeline] {journal} converged in {x} completeness and {y} correction rounds`. |
| **Final Result** | `INFO` | `{journal}: OK (Pass: {x}/{y} fields)`. |
| **System** | `ERROR` | API timeouts, disk write failures, or schema validation errors. |

## 2. Langfuse Traces (Primary)
**Focus**: Deep-dive LLM auditing and evidence flow.
**Goal**: Answer *"Why did the agent extract this value, and why did the judge mark it as incorrect?"*

**Activation**: When `LANGFUSE_PUBLIC_KEY` is set in `.env`. All agent calls and LlamaIndex operations are traced automatically via OTel instrumentation.

**Trace Structure**: `Journal Extraction: {publisher}/{journal}`
**Tags**: `golden-metadata`, `{publisher}`
**Metadata**: `publisher`, `journal`

### Auto-Traced Spans (Pydantic AI OTel)

All agent calls are captured automatically by `Agent.instrument_all()` + `instrument=True`. Each span includes agent name, input prompt, output model, tool calls, timing, and token usage.

| Span (agent name) | Purpose |
| :--- | :--- |
| **`{Pass} - Map`** | Audit evidence collection quality per chunk |
| **`{Pass} - Reduce`** | Audit assembly logic from evidence to draft |
| **`{Pass} - Completeness`** | See what the search tool found for missing fields |
| **`Verification Agent`** | See which quotes failed verification |
| **`{Pass} - Correction`** | Audit targeted field fixes |
| **`Coverage Judge`** | Audit completeness against publisher expectations |
| **`Evidence Judge`** | Audit value correctness against source quotes |

### Auto-Traced Spans (LlamaIndex OTel)

Captured by `LlamaIndexInstrumentor()`. Includes embedding generation, vector index building, and retrieval operations from the `journal_search_tool`.

### Manual Trace Wrapper

`golden/main.py` wraps each journal in a Langfuse trace:
```python
with langfuse.start_as_current_observation(
    as_type="trace",
    name=f"Journal Extraction: {publisher}/{journal}",
):
    with propagate_attributes(
        metadata={"publisher": publisher, "journal": journal},
        tags=["golden-metadata", publisher],
    ):
        # All agent.run() calls become child spans automatically
```

## 3. Logfire (Fallback)
**Activation**: When `LANGFUSE_PUBLIC_KEY` is **not** set. Preserves existing behavior.

```python
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
```

- Same Pydantic AI auto-instrumentation
- Sends to Logfire only if logfire token is present
- Zero overhead when neither Langfuse nor Logfire tokens are set

## 4. Summary Matrix

| Question | CLI Logging | Langfuse |
| :--- | :---: | :---: |
| "What is the fallback rate for the parser?" | ✅ | ❌ |
| "Why is the Correct agent hallucinating?" | ❌ | ✅ |
| "How many journals are left to process?" | ✅ | ❌ |
| "Did the completeness agent find the ISSN?" | ❌ | ✅ |
| "Which journal is costing the most tokens?" | ❌ | ✅ (Exact) |
| "Is the script hanging on a specific journal?" | ✅ | ❌ |
| "Did coverage say metrics are PRESENT but we got null?" | ❌ | ✅ (Coverage Judge) |
| "Why is a field null despite having evidence quotes?" | ❌ | ✅ (Evidence Judge) |
| "What embedding model was used?" | ❌ | ✅ (LlamaIndex OTel) |
| "How many chunks did retrieval return?" | ❌ | ✅ (search tool span) |
