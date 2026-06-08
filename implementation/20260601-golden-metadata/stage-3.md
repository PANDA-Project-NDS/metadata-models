# Stage 3 — Golden Generation Entry Point

**Goal**: Implement the CLI entry point that orchestrates journal discovery, extraction, judge loop, and output.

**Depends on**: Stage 1.5, Stage 2, Stage 4 (for `golden/agents/judge.py`)

## Files to Create/Modify

| File | Action |
|---|---|
| `golden/main.py` | **Create** — main CLI entry point |
| `scripts/editor_parser.py` | **Delete** — moved to `golden/parser.py` |
| `pyproject.toml` | **Modify** — add `"golden"` to workspace members |
| `journal-samples/.gitignore` | **Modify** — add `golden/` |

**Note**: `golden/agents/judge.py` is created in [Stage 4](stage-4.md). This stage depends on Stage 4's judge module being available before running `main()`.

## `golden/main.py`

### Imports

```python
# MUST set WITH_EVIDENCE before any model imports
os.environ.setdefault("WITH_EVIDENCE", "1")

import asyncio
import json
import logging
import os
import sys
import argparse
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode

from agents import PASSES
from golden.agents.golden import (
    chunk_files,
    CollectedEvidence,
    FieldError,
    VerificationResult,
    run_extraction_pipeline,
)
from golden.agents.judge import judge_journal, load_coverage_sections
from golden.lib.flatten import strip_evidence
from models.journal import JournalMetadata
```

### Constants

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_ROOT = PROJECT_ROOT / "journal-samples"
GOLDEN_OUT = SAMPLES_ROOT / "golden"
```

### Field-to-Pass Mapping

```python
PASS_FIELDS = {
    0: {"title", "publisher", "issn", "scope", "facts", "metrics"},
    1: {"publication_frequency", "submissions", "policies", "languages", "diamond_open_access"},
    2: {"pricing", "membership"},
    3: {"editors"},
}

# Verify all JournalMetadata top-level fields are assigned to exactly one pass
_all_pass_fields = set()
for fields in PASS_FIELDS.values():
    _all_pass_fields.update(fields)
for fname in JournalMetadata.model_fields:
    if fname not in ("journal_id", "uri"):  # structural fields, no pass
        assert fname in _all_pass_fields, f"Field '{fname}' not in any pass"
```

### Discovery and Loading

```python
from dataclasses import dataclass
from typing import Generator

@dataclass(frozen=True)
class JournalIdentity:
    """Unique identifier for a journal sample."""
    publisher: str
    journal: str

def discover_journals(publisher_filter=None, journal_filter=None) -> Generator[JournalIdentity, None, None]:
    """Walk journal-samples/{publisher}/extracted/{journal}/ for directories with .md files."""
    for pub_dir in sorted(SAMPLES_ROOT.iterdir()):
        if not pub_dir.is_dir() or pub_dir.name == "golden":
            continue
        if publisher_filter and pub_dir.name != publisher_filter:
            continue
        ext_dir = pub_dir / "extracted"
        if not ext_dir.is_dir():
            continue
        for jdir in sorted(ext_dir.iterdir()):
            if not jdir.is_dir():
                continue
            if journal_filter and jdir.name != journal_filter:
                continue
            if list(jdir.glob("*.md")):
                yield JournalIdentity(pub_dir.name, jdir.name)


def build_index_from_chunks(
    chunks: list[Chunk]
) -> VectorStoreIndex:
    """Build per-journal VectorStoreIndex from chunks.

    Passes chunk.header_path as metadata for structural context in search results.
    """
    nodes = [
        TextNode(
            text=c.text,
            metadata={"source_uri": c.filename, "header_path": c.header_path},
        )
        for c in chunks
    ]
    return VectorStoreIndex(nodes=nodes)
```

### Flatten, Merge, Strip (in `lib/flatten.py`)

All utility functions live in `golden/lib/flatten.py` and are imported by both
`golden/agents/golden.py` and `golden/main.py`.

```python
import re
from typing import Any

from golden.lib.flatten import (
    flatten_metadata,
    get_path,
    set_path,
    merge_partial,
    strip_evidence,
    MAX_ROUNDS,
)
```

#### `flatten_metadata(metadata)` → `list[tuple[str, Any, str | None]]`

Recursively walk metadata to `(path, value, evidence)` triples. Walks the
model dump (dict), yielding `(dotted.path, leaf_value, evidence)` for every
leaf field. SourcedModel evidence propagates to all child leaf fields.
SourcedValue fields yield the `.value` and `.evidence` directly.

Helper `_sourced_value_paths(model)` walks Pydantic model fields to find all
`SourcedValue`-wrapped paths. Uses `get_origin()`/`get_args()` to unwrap
`Optional`/`Union` before checking for `SourcedValue`.

`_flatten_dict(obj, prefix, sv_paths, inherited_evidence, result)` recurses
the dict: detects `SourcedValue` wrappers (`{"value": ..., "evidence": ...}`),
`SourcedModel` (dict with `"evidence"` key + other fields), regular dicts,
lists (indexed), and leaf values.

#### `get_path(obj, path)` / `set_path(obj, path, value)`

Navigate nested dict/list using dotted bracket notation, e.g.
`pricing.article_processing_charges[0].fee.value`.

`set_path` handles list indices — navigates into existing list elements by
index, creates dict placeholders for missing intermediate paths.

#### `merge_partial(draft_dump, patch_dump)`

Deep merge `patch_dump` into `draft_dump` in-place:
- `None` values in patch are skipped
- Scalar: only set if draft is `None`
- List: extend draft with patch items
- Dict/model: recurse

#### `strip_evidence(obj)`

Recursively remove `'evidence'` keys from model_dump output.

#### `MAX_ROUNDS = 5`

Maximum rounds for completeness and correction loops.

### Phase 2: Two-Judge Evaluation

The Phase 2 evaluation uses two separate judges. See [stage-4.md](stage-4.md) for the full design:

- **Coverage Judge**: Compares extracted metadata against publisher-specific coverage expectations from `journal-samples/coverage.md`
- **Evidence Judge**: Validates extracted values against source evidence quotes; flags fields that are null despite having evidence (`missing_with_evidence`)

Both judges produce per-field verdicts merged into the grading sidecar.

```python
async def judge_journal(metadata, publisher: str, coverage_text: str) -> dict:
    """Run both judges and return merged grading sidecar."""
    from agents.judge import judge_journal as run_judges
    return await run_judges(metadata, publisher, coverage_text)
```

### Write Outputs

```python
def write_outputs(publisher, journal, metadata, grading):
    """Write golden JSON and grading sidecar."""
    out_dir = GOLDEN_OUT / publisher
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = strip_evidence(metadata.model_dump(mode="json", exclude_none=True))
    (out_dir / f"{journal}.json").write_text(
        json.dumps(clean, indent=2, default=str), encoding="utf-8"
    )

    (out_dir / f"{journal}.grading.json").write_text(
        json.dumps(grading, indent=2, default=str), encoding="utf-8"
    )
```

### `main()`

```python
async def main():
    parser = argparse.ArgumentParser(description="Generate golden JournalMetadata JSON")
    parser.add_argument("--publisher", default=None)
    parser.add_argument("--journal", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of journals processed (for incremental runs).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    Settings.embed_model = get_embed_model()
    Settings.llm = None
    # Eagerly initialize embedding model to avoid blocking first journal
    Settings.embed_model._model  # noqa: B018

    # Load coverage.md sections once, cached by publisher name
    coverage_sections = load_coverage_sections()
    logger.info(f"Loaded coverage for {len(coverage_sections)} publishers")

    journals = discover_journals(args.publisher, args.journal)
    if args.limit is not None:
        import itertools
        journals = itertools.islice(journals, args.limit)
    
    stats = {"ok": 0, "fail": 0, "skip": 0}

    for identity in journals:
        publisher, journal = identity.publisher, identity.journal
        out_path = GOLDEN_OUT / publisher / f"{journal}.json"
        if out_path.exists() and not args.force:
            stats["skip"] += 1
            continue

        journal_dir = SAMPLES_ROOT / publisher / "extracted" / journal

        # Chunk files and build index once — shared across all passes
        chunks_gen = chunk_files(journal_dir)
        # Convert to list here because chunks are used multiple times across passes
        chunks = list(chunks_gen) 
        if not chunks:
            logger.warning(f"No content for {publisher}/{journal}")
            stats["fail"] += 1
            continue
        journal_id = f"{publisher}/{journal}"
        logger.info(f"Processing {journal_id} ({len(chunks)} chunks) ...")

        # Build index in thread to avoid blocking the event loop during embedding
        index = await asyncio.to_thread(build_index_from_chunks, chunks)

        # Phase 1: Map-Reduce Extraction (4 passes, concurrent within journal)
        pass_results = await asyncio.gather(*[
            run_extraction_pipeline(pc, journal_id, journal_dir, index, chunks)
            for pc in PASSES
        ])

        # Merge pass results into JournalMetadata
        merged = {}
        for result, _ in pass_results:
            merged.update(result.model_dump(mode="json"))
        metadata = JournalMetadata.model_validate(merged)
        metadata.journal_id = journal_id

        # Phase 2: Two-Judge Evaluation (coverage + evidence)
        coverage_text = coverage_sections.get(publisher, "")
        grading = await judge_journal(metadata, publisher, coverage_text)

        # Write
        if not args.dry_run:
            write_outputs(publisher, journal, metadata, grading)
        stats["ok"] += 1
        logger.info(f"Done {journal_id}. OK: {stats['ok']}, Failed: {stats['fail']}, Skipped: {stats['skip']}")

    logger.info(f"Finished. OK: {stats['ok']}, Failed: {stats['fail']}, Skipped: {stats['skip']}")
```

### Entry Point

```python
if __name__ == "__main__":
    asyncio.run(main())
```

## `journal-samples/.gitignore` Addition

```
golden/
```

## Acceptance Criteria

- `uv run python -m golden.main --publisher acs --journal es-and-t --dry-run --limit 1` completes without errors
- Output directory structure: `journal-samples/golden/{publisher}/{journal}.json` + `.grading.json`
- Golden JSON validates against `JournalMetadata` schema (after stripping evidence)
- Grading sidecar contains per-field pass/fail with reasons
