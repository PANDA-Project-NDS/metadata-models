# Observability & Logging Plan

This document defines the observability strategy for the Golden Metadata generation pipeline, utilizing a hybrid approach of system logging (Logfire/CLI) and LLM tracing (Langfuse).

## 1. Standard Logging (Logfire / CLI)
**Focus**: High-level health, pipeline progress, and aggregate metrics.
**Goal**: Answer *"Is the script running, where is it stuck, and how is it performing overall?"*

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

## 2. Langfuse Traces
**Focus**: Deep-dive LLM auditing and evidence flow.
**Goal**: Answer *"Why did the agent extract this value, and why did the judge mark it as incorrect?"*

**Trace Structure**: `Journal Extraction: {publisher}/{journal}`
**Tags**: `publisher`, `journal_id`, `pass_id`

| Span | Input (Context) | Output (Result) | Purpose |
| :--- | :--- | :--- | :--- |
| **Map (per chunk)** | Text chunk + Schema | `MapResult` (Quotes) | Audit the "collection" quality. |
| **Reduce** | Aggregate Quotes | `BaseModel` (Draft) | Audit the "assembly" logic. |
| **Completeness** | Draft + Search results | `BaseModel` (Patch) | See what the search tool actually found. |
| **Verify** | `(path, value, evidence)` | `VerificationResult` | See exactly which quote failed verification. |
| **Correct** | State + FieldErrors | `BaseModel` (Corrected) | Audit the "fix" attempt. |
| **Coverage Judge** | Coverage doc + Metadata JSON | `CoverageJudgeResult` | Audit completeness against publisher expectations. |
| **Evidence Judge** | Evidence Triples + Metadata JSON | `EvidenceJudgeResult` | Audit value correctness against source quotes. |

## 3. Summary Matrix

| Question | Logs (Logfire/CLI) | Langfuse |
| :--- | :---: | :---: |
| "What is the fallback rate for the parser?" | ✅ | ❌ |
| "Why is the Correct agent hallucinating?" | ❌ | ✅ |
| "How many journals are left to process?" | ✅ | ❌ |
| "Did the completeness agent find the ISSN?" | ❌ | ✅ |
| "Which journal is costing the most tokens?" | ⚠️ (Aggregate) | ✅ (Exact) |
| "Is the script hanging on a specific journal?" | ✅ | ❌ |
| "Did coverage say metrics are PRESENT but we got null?" | ❌ | ✅ (Coverage Judge) |
| "Why is a field null despite having evidence quotes?" | ❌ | ✅ (Evidence Judge) |
