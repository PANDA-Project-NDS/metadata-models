# Stage 2 — Golden Pipeline Agents

**Goal**: Implement the map/reduce/verify/correct agents and the `run_extraction_pipeline()` orchestrator.

**Depends on**: Stage 1, Stage 1.5

## Files to Create/Modify

| File | Action |
|---|---|
| `golden/agents/__init__.py` | **Create** — empty |
| `golden/agents/golden.py` | **Create** — all golden pipeline agents + orchestration |

**Note**: `golden/lib/__init__.py` and `golden/lib/flatten.py` are created in [Stage 1.5](stage-1.5.md) — this stage depends on them.

## `golden/agents/golden.py` Contents

### Per-Agent Model Config

Reuses `get_model(role)` from `agents/base.py`, which reads `{ROLE}_MODEL` env var with fallback to `OPENAI_MODEL`.

```python
from agents.base import get_model

map_model = get_model("map")
reduce_model = get_model("reduce")
completeness_model = get_model("completeness")
verification_model = get_model("verification")
correction_model = get_model("correction")
```

### Chunking: Structure-First Greedy Binning

`SentenceSplitter` splits at sentence boundaries regardless of document structure, producing no header metadata and cutting mid-section. The structure-first approach uses `MarkdownNodeParser` for structural decomposition, then greedy binning to aggregate nodes into ~20K token windows while preserving header context. `SentenceSplitter` remains as a safety valve for oversized sections.

```python
from dataclasses import dataclass
from typing import Generator
from pathlib import Path
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.tokenizer import get_tokenizer


@dataclass(frozen=True)
class Chunk:
    """A bounded piece of source text with structural header context."""
    filename: str
    text: str
    header_path: str


def chunk_files(
    journal_dir: Path,
    chunk_size: int = 20000,
    chunk_overlap: int = 2000,
) -> Generator[Chunk, None, None]:
    """Structure-first greedy binning for journal markdown files.

    1. MarkdownNodeParser decomposes into structural sections
    2. Greedy aggregation packs sections into ~chunk_size token windows
    3. SentenceSplitter handles individual sections exceeding chunk_size
    """
    mnp = MarkdownNodeParser()
    ss = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    tokenizer = get_tokenizer()

    for fpath in sorted(journal_dir.glob("*.md")):
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        yield from _greedy_bin(text, fpath.name, mnp, ss, chunk_size, tokenizer)


def _greedy_bin(
    text: str,
    filename: str,
    mnp: MarkdownNodeParser,
    ss: SentenceSplitter,
    chunk_size: int,
    tokenizer,
) -> Generator[Chunk, None, None]:
    """Greedy binning: aggregate MNP nodes into token-bounded chunks."""
    nodes = mnp.get_nodes_from_node(TextNode(text=text))
    if not nodes:
        yield Chunk(filename, text, "/")
        return

    buf_parts: list[str] = []
    buf_tokens = 0
    buf_header = "/"

    def flush():
        nonlocal buf_parts, buf_tokens, buf_header
        if buf_parts:
            yield Chunk(filename, "\n\n".join(buf_parts), buf_header)
            buf_parts, buf_tokens, buf_header = [], 0, "/"

    for node in nodes:
        node_text = node.text
        node_tokens = len(tokenizer(node_text))
        node_header = node.metadata.get("header_path", "/")

        # Case A: Giant — flush buffer, sentence-split this node
        if node_tokens > chunk_size:
            yield from flush()
            for sub in ss.get_nodes_from_node(TextNode(text=node_text)):
                yield Chunk(filename, sub.text, node_header)
            continue

        # Case B: Overflow — flush, start new
        if buf_tokens + node_tokens > chunk_size:
            yield from flush()

        # Add to buffer
        buf_parts.append(node_text)
        buf_tokens += node_tokens
        # Keep the deepest header path seen so far
        if _deeper(node_header, buf_header):
            buf_header = node_header

    yield from flush()


def _deeper(hp1: str, hp2: str) -> bool:
    """Check if hp1 has more path components than hp2."""
    return len([p for p in hp1.split("/") if p]) > len([p for p in hp2.split("/") if p])
```

**Benchmark**: 343 files, 363 chunks (vs 360 for SentenceSplitter-only). 60.6% of chunks carry meaningful header metadata. `srep/editors.md` (1MB) is the only file where strategies diverge meaningfully (17 vs 20 chunks), with greedy binning preserving `/Editors/` header context on every chunk.

### Map Agent (Collector)

System prompt instructs the agent to read a chunk of text and output relevant evidence. No tools. Output is `MapResult` (pydantic-ai requires a single `BaseModel`, not `list[T]`).

```python
MAP_PROMPT = """You are an evidence collector for journal metadata extraction.

## TARGET SCHEMA
The following JSON schema defines the fields you need to find evidence for. Collect evidence for every field that has information in the source text.
{schema}

## RULES
- Output a MapResult containing a list of CollectedEvidence via final_result
- Each item needs: field_hint (exact schema field path), quote (verbatim text), source (filename), note (brief relevance)
- The source filename is in the "[Source: filename]" marker at the top of the input
- field_hint must match a field path from the target schema above, e.g. "issn.print", "pricing.article_processing_charges[0].fee.value"
- Quotes must be EXACT, verbatim text from the page
- Do not paraphrase or summarize in quotes
- Only collect evidence for fields that exist in your target schema above. If the chunk contains information for fields NOT in your target schema, ignore it.
- If nothing relevant to your target schema, return MapResult with an empty evidence list
"""

def make_map_agent(pass_config: PassConfig) -> Agent[None, MapResult]:
    return Agent(
        name=f"{pass_config.name} - Map",
        model=map_model,
        output_type=MapResult,
        system_prompt=MAP_PROMPT.format(
            schema=json.dumps(pass_config.output_type.model_json_schema(), indent=2),
        ) + pass_config.domain_guidelines,
    )
```

### Reduce Agent (Assembler)

Maps collected evidence to the pass's typed output schema.

```python
REDUCE_PROMPT = """You assemble journal metadata from collected evidence.
You receive a list of evidence quotes with field hints and source files.

## RULES
- Map each piece of evidence to the correct schema field
- If multiple pieces of evidence support the same field, combine them
- If a field has no evidence, output null or empty list
- Do not hallucinate values not supported by evidence
- **CRITICAL: If a field has an "evidence" sub-field, populate it using the CollectedEvidence item that supports this value:**
  - `CollectedEvidence.quote` → `Evidence.quote` (verbatim text)
  - `CollectedEvidence.source` → `Evidence.source` (filename)
- Output via final_result with the complete extraction
"""

def make_reduce_agent(pass_config: PassConfig) -> Agent[None, pass_config.output_type]:
    return Agent(
        name=f"{pass_config.name} - Reduce",
        model=reduce_model,
        output_type=pass_config.output_type,
        system_prompt=REDUCE_PROMPT + pass_config.domain_guidelines,
    )
```

### Completeness Agent

Outputs the same pass schema type as the reduce agent. Receives the current draft state and fills only null scalar fields and missing list items. Uses `JournalSourcesDeps` with vector index.

```python
COMPLETENESS_PROMPT = """You complete a partially-filled journal metadata extraction.
You will receive the current state of the schema.

## RULES
- Only output fields that are currently null or empty in the current state
- For list fields, only emit items that are NOT already present
- For scalar fields, only populate fields whose value is null
- Fields already populated in the current state should NOT be repeated in your output
- Never re-emit or paraphrase existing data
- Use the Journal Search tool to find missing information
- Output via final_result with only the new fields found
"""

def make_completeness_agent(pass_config: PassConfig) -> Agent[JournalSourcesDeps, pass_config.output_type]:
    return Agent(
        name=f"{pass_config.name} - Completeness",
        model=completeness_model,
        output_type=pass_config.output_type,
        system_prompt=COMPLETENESS_PROMPT + pass_config.domain_guidelines,
        instructions=context_instructions,
        deps_type=JournalSourcesDeps,
        tools=[journal_search_tool],
    )
```

### Verification Agent

Checks extracted values against evidence quotes. No tools — evidence-only context. Only checks fields that have evidence.

```python
VERIFICATION_PROMPT = """You verify extracted journal metadata against evidence quotes.
You receive a list of (field_path, value, evidence) triples.

## RULES
- For each triple, check if the value is correctly derived from the evidence quote
- Only check fields that have non-null evidence
- If a value is not supported by the evidence, add a FieldError
- If all values are correct, return is_correct=True with empty errors
- Be strict: the value must be explicitly stated or clearly derivable from the evidence
"""

def make_verification_agent() -> Agent[None, VerificationResult]:
    return Agent(
        name="Verification Agent",
        model=verification_model,
        output_type=VerificationResult,
        system_prompt=VERIFICATION_PROMPT,
    )
```

### Correction Agent

Fixes flagged fields using search tool + critique. Targets only the fields in `errors`.

```python
CORRECTION_PROMPT = """You correct flagged fields in a journal metadata extraction.
You receive the current extraction, a list of field errors, and access to a search tool.

## RULES
- Only output the fields listed in the errors that you were able to correct
- Do NOT repeat fields that are not being corrected
- Use the Journal Search tool to find the correct values
- If you cannot find a correct value for a flagged field, omit it from your output
- Output via final_result with only the corrected fields
"""

def make_correction_agent(pass_config: PassConfig) -> Agent[JournalSourcesDeps, pass_config.output_type]:
    return Agent(
        name=f"{pass_config.name} - Correction",
        model=correction_model,
        output_type=pass_config.output_type,
        system_prompt=CORRECTION_PROMPT + pass_config.domain_guidelines,
        instructions=context_instructions,
        deps_type=JournalSourcesDeps,
        tools=[journal_search_tool],
    )
```

### Orchestration: `run_extraction_pipeline()`

```python
from models.journal import (
    BasicInfoExtraction,
    EditorialExtraction,
    FeesExtraction,
    PoliciesExtraction,
)

# Dedup key per pass output type — prevents list duplicates when the
# completeness agent re-emits existing items.
_DEDUP_KEY = {
    EditorialExtraction: "name",
    PoliciesExtraction: "type",
    FeesExtraction: "article_type",
    BasicInfoExtraction: None,
}


async def run_extraction_pipeline(
    pass_config: PassConfig,
    journal_id: str,
    journal_dir: Path,
    index: BaseIndex,
    chunks: list[Chunk],
) -> tuple[BaseModel, dict]:
    """Run map-reduce-verify-correct pipeline for one extraction pass.
    
    *chunks* and *index* are pre-built once per journal by the caller and
    shared across all passes.  Pass 3 (editors) uses the structural parser
    first and falls back to map-reduce if parser returns empty results.
    """
    from golden.lib.flatten import (
        flatten_metadata,
        merge_partial,
        MAX_ROUNDS,
        MAX_CORRECTION_ROUNDS,
    )
    dedup_key = _DEDUP_KEY.get(pass_config.output_type)

    # Editors pass: try structural parser first
    if pass_config.output_type == EditorialExtraction:
        parsed = try_parse_editors(journal_dir)
        if parsed:
            return EditorialExtraction(editors=parsed), {}
        # Fall through to map-reduce

    # Map phase: sequential collectors, one LLM call at a time per pass
    chunk_texts = [
        f"--- [Source: {c.filename}] [Section: {c.header_path}] ---\n{c.text}"
        for c in chunks
    ]

    all_evidence: list[CollectedEvidence] = []
    map_agent = make_map_agent(pass_config)

    for text in chunk_texts:
        result = (await map_agent.run(text)).output
        if result.evidence:
            all_evidence.extend(result.evidence)

    if not all_evidence:
        return pass_config.output_type(), {}

    # Reduce phase: assemble draft
    reduce_agent = make_reduce_agent(pass_config)
    ev_text = json.dumps([e.model_dump() for e in all_evidence], indent=2)
    schema_text = json.dumps(pass_config.output_type.model_json_schema(), indent=2)
    reduce_prompt = f"## Schema\n{schema_text}\n\n## Evidence ({len(all_evidence)} items)\n{ev_text}"
    draft = (await reduce_agent.run(reduce_prompt)).output

    # Completeness phase (single round)
    completeness_agent = make_completeness_agent(pass_config)
    for _ in range(MAX_ROUNDS):
        draft_json = json.dumps(draft.model_dump(mode="json"), indent=2, default=str)
        prompt = (
            f"## Current extraction state\n{draft_json}\n\n"
            f"## Schema (JSON)\n{json.dumps(pass_config.output_type.model_json_schema(), indent=2)}"
        )
        deps = JournalSourcesDeps(index=index, journal_id=journal_id)
        result = await completeness_agent.run(prompt, deps=deps)
        patch = result.output

        # Deep merge patch into draft
        before = draft.model_dump(mode="json")  # snapshot before merge
        patch_dump = patch.model_dump(mode="json", exclude_none=True)
        merge_partial(before, patch_dump, dedup_key=dedup_key)

        # Check if anything new was added (compare mutated dict vs fresh dump)
        if before == draft.model_dump(mode="json"):
            break
        draft = pass_config.output_type.model_validate(before)

    # Verification → Correction loop (max 2 rounds)
    verify_agent = make_verification_agent()
    correct_agent = make_correction_agent(pass_config)
    for attempt in range(MAX_CORRECTION_ROUNDS):
        fields = flatten_metadata(draft)
        to_check = [(p, v, e) for p, v, e in fields if e is not None]
        if not to_check:
            break
        check_text = json.dumps([
            {"field_path": p, "value": v, "evidence": e}
            for p, v, e in to_check
        ], indent=2, default=str)
        verdict = (await verify_agent.run(check_text)).output
        if verdict.is_correct or not verdict.errors:
            break
        # Correction — only merge flagged top-level keys
        deps = JournalSourcesDeps(index=index, journal_id=journal_id)
        current_json = json.dumps(draft.model_dump(mode="json"), indent=2, default=str)
        errors_json = json.dumps([e.model_dump() for e in verdict.errors], indent=2)
        correct_prompt = (
            f"## Current extraction\n{current_json}\n\n"
            f"## Errors to fix\n{errors_json}"
        )
        corrected = (await correct_agent.run(correct_prompt, deps=deps)).output
        old_dump = draft.model_dump(mode="json")
        patch_dump = corrected.model_dump(mode="json", exclude_none=True)
        merge_partial(old_dump, patch_dump, dedup_key=dedup_key, force=True)
        draft = pass_config.output_type.model_validate(old_dump)

    return draft, {}
```

### Structural Parser Fallback

```python
def try_parse_editors(journal_dir: Path) -> list[Editor] | None:
    """Try structural parser on editors files. Returns None if no editors file or parser fails."""
    from golden.parser import parse_editors

    for fname in ("editors.md", "editorial_board.md", "editorial-board.md", "editorial_team.md"):
        fpath = journal_dir / fname
        if fpath.exists():
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            parsed = parse_editors(text)
            if parsed:
                return parsed
    return None
```

## Acceptance Criteria

- `from golden.agents.golden import run_extraction_pipeline, make_map_agent, chunk_files, Chunk` works
- `from golden.models import CollectedEvidence, MapResult, FieldError, VerificationResult` works
- `chunk_files()` returns `Chunk` objects with `header_path` metadata
- `chunk_files()` splits large files, keeps small files whole, preserves structural sections
- `run_extraction_pipeline()` returns `(BaseModel, dict)` tuple
- `run_extraction_pipeline()` accepts pre-built `chunks` and `index`, does not call `chunk_files` internally
- Map agent input includes `[Section: header_path]` per chunk
- All existing tests still pass
