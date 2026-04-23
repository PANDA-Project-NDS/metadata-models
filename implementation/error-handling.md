# Error Handling Proposal

## Goal

Make the extraction pipeline resilient to retrieval, tool, and model failures without discarding partial work.

## Principles

- Fail locally, not globally.
- Preserve partial extraction whenever possible.
- Treat empty retrieval as normal.
- Distinguish transient failures from structural ones.
- Log enough context to debug a single journal run.

## Failure Classes

### 1. Initial retrieval failures

Examples:

- Index unavailable
- Bad metadata filter
- Retrieval backend exception

Handling:

- Log the journal ID and query set.
- Return an empty retrieval result for that pass only if no context can be built.
- Do not crash the whole journal run unless the index itself is unusable.

### 2. Tool failures

Examples:

- `journal_search` raises an exception
- Tool timeout
- Corrupted index during fallback search

Handling:

- Retry transient tool errors once or twice with a short backoff.
- If the tool still fails, keep the original retrieval context and let the agent continue.
- If the tool is unavailable, log a warning and proceed with the current pass.

### 3. Model / agent failures

Examples:

- ValidationError on schema output
- UnexpectedModelBehavior
- Malformed tool call arguments

Handling:

- Keep agent retries separate from tool retries.
- After retries are exhausted, return the best partial result available.
- Only fall back to an empty schema when no valid partial output exists.

## Recommended Control Flow

1. Run initial retrieval for the pass.
2. Build prompt from retrieved context.
3. Run the agent with `deps` enabled for fallback search.
4. If the agent calls `journal_search`, handle tool exceptions locally.
5. If fallback search fails, let the agent continue with existing context.
6. If agent output still fails validation, return the best partial schema or an empty schema for that pass only.

## Logging

Log these fields on failures:

- `journal_id`
- agent/pass name
- query or search term
- failure class
- retry count

Prefer warning logs for empty results and recoverable tool failures. Use error logs for unrecoverable index or schema issues.

## Testing Targets

- Empty journal ID returns a valid empty pass result.
- Retrieval failure does not crash unrelated passes.
- Tool timeout falls back to the original context.
- Corrupted index produces a controlled error and an empty pass result.
- One failed pass does not prevent final journal merge.

## Suggested Implementation Shape

- Keep retrieval, tool execution, and agent invocation in separate `try/except` blocks.
- Avoid one top-level catch-all that erases partial progress.
- Prefer explicit fallbacks over exception-driven control flow.
