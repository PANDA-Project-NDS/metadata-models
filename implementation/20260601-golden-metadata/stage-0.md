# Stage 0 — Foundations

**Status**: Done

## Completed

- `agents/` package refactor (`__init__.py`, `base.py`, `config.py`, `prompts.py`, `factory.py`)
- `scripts/editor_parser.py` — working structural parser (10 format extractors, dynamic heading detection)
- `models/journal.py` — `SourcedValue` evidence propagation, `facts.indexed_in` as `SourcedValue`
- Test fixes (`search.py` threshold, `seen_node_ids`)
- All 21 tests pass (19 skipped — evals require `EVAL_MODEL`)

## Parser State

18/38 journals parseable (~1.8M tokens saved). `srep` (250K tokens) fully parsed across 3 formats. Heading levels vary by publisher (`#`/`##`/`###`/`####`). Format detection uses heuristics on first 500 chars. Deduplication by normalized `name` + `affiliations`. Filename varies (`editors.md`, `editorial_board.md`, `editorial-board.md`, `editorial_team.md`). Remaining 20 journals fall back to map-reduce extraction.

## Relevant Files

- `agents/__init__.py` — re-exports public API, backward compatible
- `agents/base.py` — LLM model, settings, logfire setup
- `agents/config.py` — PassConfig, PASSES, domain rules
- `agents/prompts.py` — system prompts, search rules, evidence instructions
- `agents/factory.py` — make_agent(), context_instructions(), journal_search_tool
- `scripts/editor_parser.py` — structural parser with 10 format extractors
- `models/journal.py` — JournalMetadata schema with SourcedValue/SourcedModel wrappers
- `search.py` — JournalSourcesDeps with format_nodes() for context text
- `tests/test_search.py` — fixed node_ids() → seen_node_ids
