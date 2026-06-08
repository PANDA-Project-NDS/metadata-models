"""Pipeline orchestration for golden metadata extraction.

Manages chunking, parser whitelist, and the map-reduce-verify-correct
extraction loop.  Agent factories live in golden/agents/golden.py.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from pydantic import BaseModel
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.settings import get_tokenizer

from agents.config import PassConfig
from golden.models import (
    CollectedEvidence,
    MapResult,
    VerificationResult,
)
from golden.lib.flatten import (
    flatten_metadata,
    merge_partial,
    MAX_ROUNDS,
    MAX_CORRECTION_ROUNDS,
)
from models.journal import (
    BasicInfoExtraction,
    EditorialExtraction,
    FeesExtraction,
    PoliciesExtraction,
)
from search import JournalSourcesDeps

# Import agent factories (kept in agents/golden.py)
from golden.agents.golden import (
    make_map_agent,
    make_reduce_agent,
    make_completeness_agent,
    make_verification_agent,
    make_correction_agent,
)


# --- Chunking: Structure-First Greedy Binning ---

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
    """Structure-first greedy binning for journal markdown files."""
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
):
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

        if node_tokens > chunk_size:
            yield from flush()
            for sub in ss.get_nodes_from_node(TextNode(text=node_text)):
                yield Chunk(filename, sub.text, node_header)
            continue

        if buf_tokens + node_tokens > chunk_size:
            yield from flush()

        buf_parts.append(node_text)
        buf_tokens += node_tokens
        if _deeper(node_header, buf_header):
            buf_header = node_header

    yield from flush()


def _deeper(hp1: str, hp2: str) -> bool:
    """Check if hp1 has more path components than hp2."""
    return hp1.count("/") > hp2.count("/")


# --- Editor Parser Whitelist ---

PARSER_WHITELIST = frozenset({
    # Springer Nature — springer_inline, springer_single, bold_two_line, nmeth
    "springer_nature/srep",
    "springer_nature/npjclimataction",
    "springer_nature/nmeth",
    # SAGE — sage_table
    "sage/sgo",
    "sage/jom",
    "sage/smsa",
    "sage/smo",
    # ACS — acs_plain
    "acs/acs-energy-letters",
    "acs/es-and-t",
    "acs/acs-nano",
    "acs/jacs",
    # Copernicus — acs_plain (frontiers_block sections filtered by _is_section_heading)
    "copernicus/aerosol-research",
    "copernicus/earth-system-dynamics",
    "copernicus/mechanical-sciences",
    "copernicus/ocean-science",
    # Elsevier — elsevier_bold_section
    "elsevier/alexandria-engineering-journal",
    "elsevier/information-sciences",
    # Springer Link — elsevier_bold_section, acs_plain, heading_name
    "springer_link/bmc-biology",
    "springer_link/epj-data-science",
    "springer_link/j-solid-state-electrochem",
})


# --- Orchestration ---

# Dedup key per pass output type — prevents list duplicates when the
# completeness agent re-emits existing items.
_DEDUP_KEY = {
    EditorialExtraction: "name",
    PoliciesExtraction: "type",
    FeesExtraction: "article_type",
    BasicInfoExtraction: None,
}


def try_parse_editors(journal_dir: Path, journal_id: str):
    """Try structural parser on editors files.

    Only runs for journals in PARSER_WHITELIST.  Returns None if not
    whitelisted, no editors file found, or parser returns empty results.
    """
    if journal_id not in PARSER_WHITELIST:
        return None

    from golden.structural_parser import parse_editors

    for fname in ("editors.md", "editorial_board.md", "editorial-board.md", "editorial_team.md"):
        fpath = journal_dir / fname
        if fpath.exists():
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            parsed = parse_editors(text)
            if parsed:
                return parsed
    return None


async def run_extraction_pipeline(
    pass_config: PassConfig,
    journal_id: str,
    journal_dir: Path,
    index,
    chunks: list[Chunk],
) -> tuple[BaseModel, dict]:
    """Run map-reduce-verify-correct pipeline for one extraction pass.

    *chunks* and *index* are pre-built once per journal by the caller and
    shared across all passes.  Pass 3 (editors) uses the structural parser
    first (whitelisted journals only) and falls back to map-reduce.
    """
    dedup_key = _DEDUP_KEY.get(pass_config.output_type)

    # Editors pass: try structural parser first (whitelisted journals only)
    if pass_config.output_type == EditorialExtraction:
        parsed = try_parse_editors(journal_dir, journal_id)
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
        fields = list(flatten_metadata(draft))
        to_check = [(f.path, f.value, f.evidence) for f in fields if f.evidence is not None]
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
