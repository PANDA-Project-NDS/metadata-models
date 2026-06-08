# Golden Metadata Samples — Overview

## Goal

Generate golden `JournalMetadata` JSON samples from the 38 extracted journal markdown directories in `journal-samples/*/extracted/`, with two-judge evaluation: a **Coverage Judge** comparing extraction completeness against `journal-samples/coverage.md`, and an **Evidence Judge** validating values against source quotes. Extraction runs in **evidence mode** (`WITH_EVIDENCE=1`) using a map-reduce pipeline with bounded context (~20K tokens per chunk). Golden output strips evidence to produce clean JSON.

## Architecture

```
journal-samples/{publisher}/extracted/{journal}/*.md
    │
    ▼
Phase 1: Map-Reduce Extraction (per pass, concurrent across passes)
    │
    ├─ Map: Parallel evidence collectors on ~20K token chunks
    ├─ Reduce: Assembler maps evidence → typed extraction
    ├─ Completeness: Search tool fills missing fields
    ├─ Verification: Evidence-only correctness check
    └─ Correction: Targeted fixes for flagged fields (max 2 rounds)
    │
    ▼
Merge all 4 passes → JournalMetadata
    │
    ▼
Phase 2: Two-Judge Evaluation
     │
     ├─ Coverage Judge: Compare against publisher coverage expectations (coverage.md)
     └─ Evidence Judge: Verify values against source evidence quotes
     │
     ▼
Merge per-field verdicts → write golden JSON + grading sidecar
```

## Stages

| Stage | File | Status |
|---|---|---|
| 0 — Foundations | [stage-0.md](stage-0.md) | Done |
| 1 — Evidence Types | [stage-1.md](stage-1.md) | TODO |
| 1.5 — Utility Functions | [stage-1.5.md](stage-1.5.md) | TODO |
| 2 — Golden Pipeline Agents | [stage-2.md](stage-2.md) | TODO |
| 3 — Generation Entry Point | [stage-3.md](stage-3.md) | TODO |
| 4 — Two-Judge Evaluation | [stage-4.md](stage-4.md) | TODO |

## Observability
- [observability.md](observability.md) — Hybrid logging (Logfire) and tracing (Langfuse) strategy.
- Chunking: structure-first greedy binning (see [stage-2.md](stage-2.md))

## CLI Usage

```bash
# All journals
uv run python -m golden.main

# Single publisher
uv run python -m golden.main --publisher wiley

# Single journal
uv run python -m golden.main --publisher mdpi --journal sustainability

# Force regenerate
uv run python -m golden.main --force

# Dry run (extract + judge, don't write)
uv run python -m golden.main --dry-run

# Cap journals for incremental runs / cost control
uv run python -m golden.main --limit 5

# With env
JUDGE_MODEL=Qwen3.6-32B-A3B-instruct:Q8_0 \
  uv run python -m golden.main
```

## Output Layout

```
journal-samples/golden/
  acs/
    acs-energy-letters.json
    acs-energy-letters.grading.json
    acs-nano.json
    acs-nano.grading.json
    ...
  wiley/
    advanced-energy-materials.json
    advanced-energy-materials.grading.json
    ...
  (12 publisher dirs, 76 files total: 38 .json + 38 .grading.json)
```

## Token Budget Estimate

Per journal (evidence mode, map-reduce extraction):

| Phase | Old (dump all) | New (map-reduce) |
|---|---|---|
| Map (per chunk) | — | ~20K × N chunks (parallel, not additive) |
| Reduce | — | ~3K evidence + ~2K schema = ~5K |
| Completeness | ~5K + search | ~5K + search (same) |
| Verification | — | ~2K evidence triples |
| Correction | ~100K (repair) | ~5K draft + search results |
| Coverage Judge | — | ~2K coverage + ~5K metadata = ~7K |
| Evidence Judge | — | ~3K evidence triples + ~5K metadata = ~8K |
| **Max context at once** | **262K (srep)** | **20K** |

Extraction worst case per journal: ~87K tokens. Phase 2 adds ~15K per journal for both judges. For 38 journals at concurrency 4: ~3.9M tokens total. **~97% reduction** from the old dump-all approach (170M tokens).

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Map-reduce extraction | Parallel collectors on bounded chunks (~20K tokens). Reduces context from 262K to 20K max. |
| No paging tool | Parallel calls on chunks are simpler than sequential page-turning. No state management across turns. |
| Evidence collector → assembler | Collector finds quotes (no typing pressure). Assembler maps to schema. Separation of concerns. |
| Completeness with search tool | Fills gaps after reduce. No full source text needed. |
| Verification before correctness judge | Catches errors early with evidence-only context. Only fields with evidence are checked. |
| Correction replaces repair | Targeted field fixes with search tool. Simpler than re-running full pass agents. |
| Structural parser for editors | Handles 18/38 journals without LLM. Falls back to map-reduce for remaining 20. |
| `golden/` package | Golden pipeline is a separate uv workspace package. Imports shared modules (`models/`, `agents/`, `search.py`, `db/embed.py`) from the main project. |
| Two-judge Phase 2 | Coverage Judge checks publisher-aware completeness against `coverage.md`; Evidence Judge checks value correctness against source quotes. Separates concerns. |
| `JUDGE_MODEL` env var | Controls the judge model for both judges. Falls back to `OPENAI_MODEL`. |
| Evidence mode extraction | Each value carries its evidence quote. Evidence Judge sees ~200 tokens per field. |
| `missing_with_evidence` flag | Detects fields that are null despite having evidence quotes — catches assembly gaps in Phase 1. |
| Golden output strips evidence | `strip_evidence()` produces clean JSON for direct Pydantic validation. |
| Max 2 correction rounds | Prevents infinite loops. Accepts partial results if still failing. |
| Separate grading file | Golden JSON stays clean for direct schema validation. Grading is audit-only. |
| `--limit` flag | Cost-control knob for incremental runs. |
| Phase 2 is evaluation-only | Correction happens in Phase 1 per-pass loop (verify → correct). Phase 2 judges are read-only. |
| Coverage document as judge context | Raw `coverage.md` markdown fed directly to Coverage Judge — no parsing, no structured loading. |
| Per-agent model config | Different models per stage: `MAP_MODEL`, `REDUCE_MODEL`, `COMPLETENESS_MODEL`, `VERIFICATION_MODEL`, `CORRECTION_MODEL`. All fallback to `OPENAI_MODEL`. |
| `golden/lib/` package | Utility functions (`flatten_metadata`, `merge_partial`, `get_path`, `set_path`, `strip_evidence`) live in `golden/lib/flatten.py`, shared by agents and entry point. |
| `JOURNAL_ID = publisher/journal-slug` | Stable, parseable, mirrors directory layout. |
| `golden/` in `.gitignore` | Unfinished runs shouldn't pollute history. |
| No hardcoded rubrics | Derive verification criteria from Pydantic `Field(..., description="...")` dynamically. |
| `WITH_EVIDENCE` env var at script top | Set before any model imports so `INCLUDE_EVIDENCE` is `True` at module load time. |
| `MapResult` wrapper | pydantic-ai requires `output_type` to be a single `BaseModel`, not `list[T]`. |
| Correction only merges flagged keys | Prevents silent drift on non-flagged fields. |
| Completeness scoped to pass schema | `pass_config.output_type.model_json_schema()` — only flags fields that belong to the current pass. |
| Structure-first greedy binning for chunking | `MarkdownNodeParser` decomposes on headers, greedy aggregation packs into ~20K token windows. `SentenceSplitter` as safety valve for oversized sections. Provides `header_path` metadata on 60.6% of chunks. See [stage-2.md](stage-2.md) for implementation and benchmarks. |

## Open Items Not in This Plan

- **Re-run-on-grading-update**: Once a journal is golden, the grading sidecar is the only audit. If `metadata_definition.md` changes (new field), the grading becomes stale. A `--validate-existing` mode could re-validate without re-extracting.
- **Diff mode**: No way to diff two golden runs to see what changed. Out of scope.
- **Evidence quality**: If collector produces poor evidence (too short, too long, wrong quote), verification will flag the field. Correction loop handles this.
- **Parser improvements**: Wiley dash lists, IEEE tmrb/tbdata scraped forms, Elsevier info-sci/alexandria `####` sections need parser fixes. Filename discovery needs to handle `editorial-board.md` variant.
