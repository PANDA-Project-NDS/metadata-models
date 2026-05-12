# Factorize Extraction Agents

## Status

Planned

## Problem

Four agents (`basic_info_agent`, `policies_agent`, `fees_agent`, `editors_agent`) are defined as module-level singletons in `agents.py` with nearly identical configuration. `SYSTEM_PROMPT`, `llm_model`, `journal_search_tool`, `output_retries`, `deps_type`, and `instructions` are copy-pasted across all four. `EXTRACTION_QUERIES` lives at module level with no structural relationship to the agents. `pipeline.py` imports four named agents and hardcodes four task variables. Adding a fifth pass touches both files. Deletion test: `agents.py` is a pass-through — four pre-built objects, no behaviour behind the interface.

## Goal

One `PassConfig` dataclass, one `PASSES` list, one `make_agent()` factory. Adding a pass is one line. `pipeline.py` iterates `PASSES`, knows nothing about pass count.

## Design

### `PassConfig`

```python
@dataclass
class PassConfig:
    name: str                  # "basic_info"
    agent_label: str           # "Info Agent" (for pydantic_ai Agent.name)
    output_type: type[BaseModel]  # BasicInfoExtraction
    queries: list[str]
```

### `PASSES`

Replaces both `EXTRACTION_QUERIES` dict and the four named agent instances:

```python
PASSES: list[PassConfig] = [
    PassConfig("basic_info", "Info Agent", BasicInfoExtraction, [
        "Journal title, publisher, about this journal, mission, scope, sections",
        "ISSN, print ISSN, online ISSN, indexed in, abstracting and indexing databases",
        "Impact factor, journal metrics, citation score, cite score",
    ]),
    PassConfig("policies_submissions", "Policies Agent", PoliciesExtraction, [
        "Publication frequency, issues per year, submission guidelines, author instructions, article types accepted",
        "Peer review process, blind review, open access policy statement, copyright, quality assurance",
        "diamond open access, community owned, open to all authors",
        "publication languages, languages accepted",
    ]),
    PassConfig("fees_membership", "Fees Agent", FeesExtraction, [
        "Article Processing Charge, APC, publication fees, cost, waivers, discounts, society membership, institutional membership",
    ]),
    PassConfig("editors", "Editors Agent", EditorialExtraction, [
        "Editorial board, Editor in Chief, managing editor, editorial team",
    ]),
]
```

### `make_agent(pass_config) -> Agent`

Pulls shared configuration from module-level constants. No extra parameters:

```python
def make_agent(pass_config: PassConfig) -> Agent:
    return Agent(
        name=pass_config.agent_label,
        model=llm_model,
        output_type=pass_config.output_type,
        system_prompt=SYSTEM_PROMPT,
        instructions=context_instructions,
        output_retries=3,
        deps_type=JournalSourcesDeps,
        tools=[journal_search_tool],
    )
```

### Module-level constants (unchanged)

`SYSTEM_PROMPT`, `llm_model`, `journal_search_tool`, `context_instructions` remain at module level in `agents.py`. They're shared across all passes — no reason to parameterize them.

### Removed

- `basic_info_agent`, `policies_agent`, `fees_agent`, `editors_agent` named exports
- `EXTRACTION_QUERIES` dict

## `pipeline.py` Changes

### Imports

```python
# Before
from agents import (
    basic_info_agent,
    policies_agent,
    fees_agent,
    editors_agent,
    EXTRACTION_QUERIES,
)
from search import retrieve_for_pass, JournalSourcesDeps

# After
from agents import PASSES, make_agent
from search import retrieve_for_pass, JournalSourcesDeps
```

### `process_journal`

```python
async def process_journal(index: VectorStoreIndex, journal_id: str) -> JournalMetadata:
    import asyncio

    logger.info(f"Starting multi-pass extraction for {journal_id}...")

    agents = [make_agent(p) for p in PASSES]
    tasks = [
        run_extraction_pass(index, agent, p.queries, journal_id)
        for agent, p in zip(agents, PASSES)
    ]
    results = await asyncio.gather(*tasks)

    # Merge into Final Schema
    final_metadata = JournalMetadata()
    for result in results:
        final_metadata.__dict__.update(result.model_dump())

    return final_metadata
```

### `run_extraction_pass`

Signature unchanged. No awareness of pass count.

## Unchanged

- `SYSTEM_PROMPT`, `llm_model`, `journal_search_tool`, `context_instructions` — remain module-level in `agents.py`
- `search.py`, `models/`, `db/` — untouched
- Error handling in `run_extraction_pass` — unchanged

## Merge Note

No overlapping fields exist between the four extraction schemas. Order does not matter. Inline `__dict__.update()` loop is sufficient — no need to extract `merge_extractions()` at this point.

## Verification

- `python pipeline.py --publisher <name>` runs identically
- Adding a fifth pass is one `PassConfig` appended to `PASSES`
- `grep basic_info_agent` returns zero results (named exports removed)
- `grep EXTRACTION_QUERIES` returns zero results (dict removed)
